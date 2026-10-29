from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from webapp.progress import StepTimer, emit_progress

from webapp.core import (
    EXPECTED_PAIRED_MANIFEST_SHA256,
    EXPECTED_NEGATIVE_EMBEDDING_SHA256,
    EXPECTED_LATENT_SCALE,
    PipelineError,
    RELEASE_ROOT,
    SAMPLE_ID_PATTERN,
    AssetPaths,
    compute_metrics,
    emit_stage,
    get_asset_paths,
    load_verified_mappings,
    load_verified_test_eeg,
    sha256_file,
    safe_data_path,
    validate_upload_file,
    verify_channel_file,
    identify_sample,
    write_mp4,
    _load_rgb,
)


EXPECTED_ARTIFACT_HASHES = {
    "visual_checkpoint": "508092943464347b69b07de5d9c049c55d0f30b80e4a7c4ff974e20ac98e3741",
    "visual_normalisation": "066ae2ee9092c3c27ca1eaa6671f5618b5bd28bf8557adec426db40040219a73",
    "visual_predictions": "a0e29127d5a820a9d974f71636c1c526d1cf5d1e74d705c37cbdcbae339d102b",
    "semantic_checkpoint": "3df77e7e36c9afffdd87c09c10b329ebdc88a0141e97348d96ff45d3bf6da377",
    "semantic_normalisation": "92025576b59b4a2bc86e5178d2b428e6f629a0a72544dc7dfa01f6112f33fea1",
    "semantic_predictions": "41d4a96e64a474c713020906ebfd5e5c2e7e23ee9e99f64d23f61a2590458fcb",
    "dynamic_checkpoint": "6d7f35c82173ce481c965f5724df1ae80caa9f4cf10ce5cb70410432ebc1463a",
    "dynamic_normalisation": "931529566d620454a768fe8c8cc4cc768171ee9a3d9cb8c8997e478143891a69",
    "dynamic_predictions": "99a7f412f8cb30f3ee39f078e5b42995493c240ff70da8d87432515ff779df33",
    "diffusion_unet": "95a5018d42862f3fe9d0e45f7f01bebe1b7cf611c19f56cbabaf6f4d74930543",
    "diffusion_scheduler": "7a33e77332bb34ae30a24bcef37f184acc73b93c99202ae3e1c353ed452a2609",
    "sd14_model_index": "c7cc314014486aa2f7f17ab8f90197fe0412c818cdefb9e15601ae58752b5001",
    "sd14_vae_config": "0e5966d4d52c8077648754ffd2c14ab5e0b26a4bdfa6af1d207c27550fa6d2af",
    "sd14_vae_weights": "1b134cded8eb78b184aefb8805b6b572f36fa77b255c483665dda931fa0130c5",
}

PREDICTION_TOLERANCES = {
    "visual": {"rtol": 1e-4, "atol": 1e-5},
    "semantic": {"rtol": 1e-4, "atol": 1e-5},
}


def _import_source(module_name: str, source: Path):
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Cannot load the existing pipeline module {source.name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise PipelineError(f"Required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise PipelineError(f"The {label} SHA-256 does not match the verified project artifact.")
    return actual


def _normalisation(path: Path, expected_hash: str, expected_shape: tuple[int, ...], label: str):
    _require_hash(path, expected_hash, f"{label} normalisation")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"mean", "std"}:
                raise PipelineError(f"{label} normalisation must contain exactly mean and std.")
            mean = np.array(archive["mean"], dtype=np.float64, copy=True)
            std = np.array(archive["std"], dtype=np.float64, copy=True)
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"Cannot read {label} normalisation: {exc}") from exc
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise PipelineError(f"{label} normalisation has an unexpected shape.")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise PipelineError(f"{label} normalisation contains invalid values.")
    return mean, std


def _load_trusted_checkpoint(torch, path: Path, expected_hash: str, label: str) -> dict:
    _require_hash(path, expected_hash, f"{label} checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise PipelineError(f"Could not safely load the {label} checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise PipelineError(f"The {label} checkpoint is not in the expected final-model format.")
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise PipelineError(f"The {label} checkpoint contains no model state dictionary.")
    checkpoint["model"] = state
    return checkpoint


def _known_prediction_reference(paths: AssetPaths, test_index: int, name: str):
    directory = {
        "visual": paths.visual_output,
        "semantic": paths.semantic_output,
        "dynamic": paths.dynamic_output,
    }[name]
    filenames = {
        "visual": "block7_predicted_visual_latents.npy",
        "semantic": "block7_predicted_text_embeddings.npy",
        "dynamic": "block7_predicted_dynamic_labels.npy",
    }
    mapping_file = directory / "prediction_mapping.csv"
    values_path = directory / filenames[name]
    expected_hash = EXPECTED_ARTIFACT_HASHES[f"{name}_predictions"]
    _require_hash(values_path, expected_hash, f"saved {name} predictions")
    rows = []
    try:
        import csv

        with mapping_file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        value = np.load(values_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"Cannot load the saved {name} prediction reference: {exc}") from exc
    if len(rows) != 200 or rows[test_index].get("sample_id") != f"sub01_session01_block07_clip{test_index + 1:03d}":
        raise PipelineError(f"Saved {name} prediction mapping is not aligned to Block 7.")
    if rows[test_index].get("prediction_index") != str(test_index):
        raise PipelineError(f"Saved {name} prediction index is misaligned.")
    return np.asarray(value[test_index])


def compare_predictions(predictions: dict, paths: AssetPaths, test_index: int | None) -> dict:
    report = {}
    if test_index is None:
        return {
            "status": "not_comparable",
            "reason": "Input did not exactly match the canonical Block 7 test set.",
        }
    for name in ("visual", "semantic"):
        expected = _known_prediction_reference(paths, test_index, name)
        actual = np.asarray(predictions[name])
        if actual.shape != expected.shape:
            raise PipelineError(f"Live {name} prediction shape differs from its stored Block 7 counterpart.")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        tolerance = PREDICTION_TOLERANCES[name]
        passed = bool(np.allclose(actual, expected, **tolerance))
        report[name] = {
            "shape": list(actual.shape),
            "max_absolute_difference": float(difference.max(initial=0)),
            "mean_absolute_difference": float(difference.mean()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "rtol": tolerance["rtol"],
            "atol": tolerance["atol"],
            "within_tolerance": passed,
        }
    saved_dynamic = _known_prediction_reference(paths, test_index, "dynamic")
    predicted_dynamic = int(predictions["dynamic_label"])
    report["dynamic"] = {
        "shape": [],
        "saved_label": int(saved_dynamic),
        "predicted_label": predicted_dynamic,
        "match": int(saved_dynamic) == predicted_dynamic,
    }
    report["status"] = "passed" if (
        report["visual"]["within_tolerance"]
        and report["semantic"]["within_tolerance"]
        and report["dynamic"]["match"]
    ) else "failed"
    return report


def predict_single_sample(
    sample: np.ndarray,
    paths: AssetPaths | None = None,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    import torch

    paths = paths or get_asset_paths()
    cpu_threads = int(os.environ.get("EEG2VIDEO_CPU_THREADS", "4"))
    if cpu_threads < 1:
        raise PipelineError("EEG2VIDEO_CPU_THREADS must be a positive integer.")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if sample.shape != (62, 400) or not np.issubdtype(sample.dtype, np.number):
        raise PipelineError("Predictor input must be one numeric EEG sample with shape (62, 400).")
    if not np.isfinite(sample).all():
        raise PipelineError("Predictor input contains NaN or infinity.")

    source = RELEASE_ROOT / "EEG2Video" / "EEG2Video_New"
    feature_source = RELEASE_ROOT / "EEG_preprocessing" / "extract_DE_PSD_features_1per2s.py"
    outputs = {}

    emit_stage(callback, "Visual prediction", "running", "Loading the original GLMNet + Seq2Seq implementation and training normalisation on CPU.")
    visual_checkpoint_path = paths.visual_output / "final_model.pt"
    visual_norm_path = paths.visual_output / "eeg_normalisation.npz"
    mean, std = _normalisation(
        visual_norm_path,
        EXPECTED_ARTIFACT_HASHES["visual_normalisation"],
        (7, 62, 100),
        "Visual predictor",
    )
    visual_module = _import_source(
        "eeg2video_original_visual_predictor",
        source / "Seq2Seq" / "my_autoregressive_transformer_glmnet.py",
    )
    checkpoint = _load_trusted_checkpoint(
        torch,
        visual_checkpoint_path,
        EXPECTED_ARTIFACT_HASHES["visual_checkpoint"],
        "visual predictor",
    )
    model = visual_module.myTransformer()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model_input, _, _ = visual_module.prepare_eeg(sample[None], mean, std)
    if tuple(model_input.shape) != (1, 7, 62, 100):
        raise PipelineError(f"Visual predictor input has unexpected shape {tuple(model_input.shape)}.")
    with torch.inference_mode():
        visual = model(model_input).cpu().numpy()[0]
    if visual.shape != (6, 4, 36, 64) or not np.isfinite(visual).all():
        raise PipelineError(f"Visual predictor returned invalid output shape/values: {visual.shape}.")
    outputs["visual"] = np.asarray(visual, dtype=np.float32)
    del model, checkpoint, model_input, mean, std, visual_module
    gc.collect()
    emit_stage(callback, "Visual prediction", "complete", "Predicted six 4x36x64 visual latents from seven EEG windows.")

    emit_stage(callback, "Semantic prediction", "running", "Extracting full_window_400fft DE features and loading the CPU float32 semantic MLP.")
    semantic_checkpoint_path = paths.semantic_output / "final_model.pt"
    semantic_norm_path = paths.semantic_output / "eeg_normalisation.npz"
    mean, std = _normalisation(
        semantic_norm_path,
        EXPECTED_ARTIFACT_HASHES["semantic_normalisation"],
        (310,),
        "Semantic predictor",
    )
    feature_module = _import_source("eeg2video_original_de_psd_features", feature_source)
    de, _, _ = feature_module.extract_features(sample[None], nfft=400)
    features = np.asarray(de.reshape(1, 310), dtype=np.float64)
    semantic_input = ((features - mean) / std).astype(np.float32)
    if semantic_input.shape != (1, 310) or not np.isfinite(semantic_input).all():
        raise PipelineError("Semantic predictor features are invalid after training normalisation.")
    semantic_module = _import_source(
        "eeg2video_original_semantic_predictor",
        source / "Semantic" / "eeg_text.py",
    )
    checkpoint = _load_trusted_checkpoint(
        torch,
        semantic_checkpoint_path,
        EXPECTED_ARTIFACT_HASHES["semantic_checkpoint"],
        "semantic predictor",
    )
    try:
        with torch.device("meta"):
            model = semantic_module.CLIP()
        model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    except (RuntimeError, TypeError, KeyError) as exc:
        raise PipelineError(f"Could not memory-efficiently initialise the semantic predictor: {exc}") from exc
    del checkpoint
    gc.collect()
    model.eval()
    with torch.inference_mode():
        semantic = model(torch.from_numpy(semantic_input)).cpu().numpy()[0].reshape(77, 768)
    if semantic.shape != (77, 768) or not np.isfinite(semantic).all():
        raise PipelineError(f"Semantic predictor returned invalid output shape/values: {semantic.shape}.")
    outputs["semantic"] = np.asarray(semantic, dtype=np.float32)
    del model, semantic_module, feature_module, semantic_input, features, mean, std
    gc.collect()
    emit_stage(callback, "Semantic prediction", "complete", "Predicted 77x768 text embeddings in CPU float32.")

    emit_stage(callback, "Dynamic prediction", "running", "Loading the original two-branch GLMNet and training-fitted coordinate normalisation on CPU.")
    dynamic_checkpoint_path = paths.dynamic_output / "final_model.pt"
    dynamic_norm_path = paths.dynamic_output / "eeg_normalisation.npz"
    mean, std = _normalisation(
        dynamic_norm_path,
        EXPECTED_ARTIFACT_HASHES["dynamic_normalisation"],
        (62, 400),
        "Dynamic predictor",
    )
    dynamic_module = _import_source(
        "eeg2video_original_dynamic_predictor",
        source / "Dynamic" / "train_dynamic_glmnet.py",
    )
    checkpoint = _load_trusted_checkpoint(
        torch,
        dynamic_checkpoint_path,
        EXPECTED_ARTIFACT_HASHES["dynamic_checkpoint"],
        "dynamic predictor",
    )
    model = dynamic_module.build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dynamic_input = ((sample.astype(np.float64) - mean) / std).astype(np.float32)[None, None]
    if dynamic_input.shape != (1, 1, 62, 400) or not np.isfinite(dynamic_input).all():
        raise PipelineError("Dynamic predictor input is invalid after training normalisation.")
    with torch.inference_mode():
        logits = model(torch.from_numpy(dynamic_input))
        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
            raise PipelineError("Dynamic predictor returned invalid logits.")
        label = int(logits.argmax(dim=1).item())
        probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    if label not in (0, 1):
        raise PipelineError(f"Dynamic predictor returned invalid label: {label}.")
    outputs["dynamic_label"] = label
    outputs["dynamic_probabilities"] = probabilities.astype(np.float32).tolist()
    outputs["model_hashes"] = {
        "visual_checkpoint": EXPECTED_ARTIFACT_HASHES["visual_checkpoint"],
        "visual_normalisation": EXPECTED_ARTIFACT_HASHES["visual_normalisation"],
        "semantic_checkpoint": EXPECTED_ARTIFACT_HASHES["semantic_checkpoint"],
        "semantic_normalisation": EXPECTED_ARTIFACT_HASHES["semantic_normalisation"],
        "dynamic_checkpoint": EXPECTED_ARTIFACT_HASHES["dynamic_checkpoint"],
        "dynamic_normalisation": EXPECTED_ARTIFACT_HASHES["dynamic_normalisation"],
    }
    del model, checkpoint, dynamic_module, dynamic_input, mean, std, logits
    gc.collect()
    emit_stage(
        callback,
        "Dynamic prediction",
        "complete",
        f"Predicted dynamic label {label} ({'fast' if label else 'slow'}).",
    )
    return outputs


def predict_known_block7_sample(test_index: int, callback: Callable[[dict], None] | None = None) -> dict:
    if not 0 <= test_index < 200:
        raise PipelineError("Test index must be between 0 and 199.")
    paths = get_asset_paths()
    verify_channel_file(paths.channel_file)
    test = load_verified_test_eeg(paths.test_eeg)
    load_verified_mappings(paths)
    result = predict_single_sample(np.asarray(test[test_index]), paths, callback)
    return result


def _validate_diffusion_assets(paths: AssetPaths) -> tuple[Path, Path]:
    diffusion = paths.diffusion_output
    provenance_file = diffusion / "provenance.json"
    if not provenance_file.is_file():
        raise PipelineError(f"Completed diffusion provenance is missing: {provenance_file}")
    try:
        provenance = json.loads(provenance_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("Cannot read the completed diffusion provenance record.") from exc
    if (
        provenance.get("status") != "complete"
        or provenance.get("completed_steps") != 24000
        or provenance.get("steps") != 24000
        or provenance.get("epochs") != 200
        or provenance.get("test_block_used") is not False
        or provenance.get("training_set") != "paper_all"
        or provenance.get("manifest_sha256") != EXPECTED_PAIRED_MANIFEST_SHA256
    ):
        raise PipelineError("The diffusion checkpoint does not have the expected training-only provenance.")

    unet_weights = diffusion / "unet" / "diffusion_pytorch_model.bin"
    scheduler_config = diffusion / "scheduler" / "scheduler_config.json"
    _require_hash(unet_weights, EXPECTED_ARTIFACT_HASHES["diffusion_unet"], "final temporal UNet")
    _require_hash(scheduler_config, EXPECTED_ARTIFACT_HASHES["diffusion_scheduler"], "final DDIM scheduler")
    if not (diffusion / "unet" / "config.json").is_file():
        raise PipelineError("Final temporal UNet config is missing.")
    if paths.sd14_model is None or not paths.sd14_model.is_dir():
        raise PipelineError(
            "The pinned Stable Diffusion 1.4 local snapshot is unavailable. Set sd14_model "
            "to its relative path in asset_config.json; the application will not download it."
        )
    if not (paths.sd14_model / "model_index.json").is_file():
        raise PipelineError("The configured Stable Diffusion directory is not a local model snapshot.")
    for path, key, label in (
        (paths.sd14_model / "model_index.json", "sd14_model_index", "Stable Diffusion model index"),
        (paths.sd14_model / "vae" / "config.json", "sd14_vae_config", "Stable Diffusion VAE config"),
        (
            paths.sd14_model / "vae" / "diffusion_pytorch_model.bin",
            "sd14_vae_weights",
            "Stable Diffusion VAE weights",
        ),
    ):
        _require_hash(path, EXPECTED_ARTIFACT_HASHES[key], label)
    _require_hash(
        paths.negative_embedding,
        EXPECTED_NEGATIVE_EMBEDDING_SHA256,
        "final negative-conditioning embedding",
    )
    if provenance.get("model_revision") != "133a221b8aa7292a167afc5127cb63fb5005638b":
        raise PipelineError("Diffusion training and Stable Diffusion VAE revisions do not match.")
    try:
        latent_scale = float(provenance.get("latent_scale", -1))
    except (TypeError, ValueError) as exc:
        raise PipelineError("Diffusion training provenance has an invalid latent scale.") from exc
    if latent_scale != EXPECTED_LATENT_SCALE:
        raise PipelineError("Diffusion training latent scale differs from the verified run.")
    return diffusion, paths.sd14_model


def _verify_recorded_noise(
    paths: AssetPaths,
    test_index: int,
    seed: int,
    dynamic_label: int,
    beta: float,
    dana_timestep: int,
    alpha: float,
    first_ddim_timestep: int,
    noise_hash: str,
    live_initial_latent,
    shared,
    diverse,
    init_noise_sigma: float,
) -> dict:
    import torch

    sample_id = f"sub01_session01_block07_clip{test_index + 1:03d}"
    completion_file = paths.generation_output / "clips" / sample_id / "complete.json"
    if not completion_file.is_file():
        raise PipelineError(f"Completed generation record is missing for {sample_id}.")
    try:
        completion = json.loads(completion_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read completed generation record for {sample_id}.") from exc
    expected = {
        "sample_id": sample_id,
        "seed": seed,
        "dynamic_label": dynamic_label,
        "beta": beta,
        "dana_timestep": dana_timestep,
        "first_ddim_timestep": first_ddim_timestep,
        "noise_sha256": noise_hash,
        "protocol": "released500",
        "ground_truth_conditioning": False,
    }
    if any(completion.get(key) != value for key, value in expected.items()):
        raise PipelineError(
            f"Live DANA noise or initial latent differs from the recorded final run for {sample_id}; "
            "diffusion was not started."
        )
    try:
        recorded_alpha = float(completion.get("dana_alpha_cumprod"))
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Completed DANA alpha is invalid for {sample_id}.") from exc
    if not np.isclose(recorded_alpha, alpha, rtol=0, atol=1e-8):
        raise PipelineError(f"Live DANA alpha differs from the recorded final run for {sample_id}.")
    saved_visual = _known_prediction_reference(paths, test_index, "visual")
    saved_latent = torch.from_numpy(np.array(saved_visual, copy=True))
    saved_latent = saved_latent.permute(1, 0, 2, 3)[None].to(
        device=live_initial_latent.device,
        dtype=torch.float32,
    )
    saved_latent = (
        alpha**0.5 * saved_latent
        + (1 - alpha) ** 0.5 * (beta**0.5 * shared + (1 - beta) ** 0.5 * diverse)
    ).to(live_initial_latent.dtype)
    saved_latent = saved_latent * init_noise_sigma
    saved_hash = hashlib.sha256(saved_latent.float().cpu().numpy().tobytes()).hexdigest()
    if saved_hash != completion.get("initial_latent_sha256"):
        raise PipelineError(f"Reconstructed historical initial latent is invalid for {sample_id}.")
    difference = (live_initial_latent.float() - saved_latent.float()).abs()
    return {
        "sha256": saved_hash,
        "matches_live_initial_latent": saved_hash
        == hashlib.sha256(live_initial_latent.float().cpu().numpy().tobytes()).hexdigest(),
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(difference**2)).item()),
        "note": (
            "The live initial latent is formed from freshly recomputed predictor outputs; "
            "the comparison uses the saved Block 7 visual prediction and the exact verified DANA noise."
        ),
    }


def _run_single_sample_diffusion(
    predictions: dict,
    paths: AssetPaths,
    test_index: int | None,
    user_seed: int,
    run_dir: Path,
    callback: Callable[[dict], None] | None,
) -> tuple[np.ndarray, dict]:
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler

    if not torch.cuda.is_available():
        raise PipelineError("Live video generation requires the verified CUDA GPU; no CPU fallback is used.")
    if not 0 <= user_seed <= 2**31 - 1:
        raise PipelineError("Seed must be an integer between 0 and 2,147,483,647.")
    if importlib.util.find_spec("xformers") is None:
        raise PipelineError(
            "The verified xFormers attention package is unavailable in the worker environment. "
            "No alternate attention implementation will be substituted."
        )
    diffusion, sd14_model = _validate_diffusion_assets(paths)
    free_vram, total_vram = torch.cuda.mem_get_info()
    if free_vram < int(4.5 * 1024**3):
        raise PipelineError(
            f"Only {free_vram / 1024**3:.2f} GiB GPU memory is free; the verified 6 GB CUDA "
            "generation path requires at least 4.5 GiB free. Close GPU workloads and retry."
        )
    device = torch.device("cuda")
    dtype = torch.float16
    seed = 42 + test_index if test_index is not None else user_seed
    if not 0 <= seed <= 2**31 - 1:
        raise PipelineError("The Block 7-derived deterministic seed exceeds the supported range.")

    generation = RELEASE_ROOT / "EEG2Video" / "EEG2Video_New" / "Generation"
    if str(generation) not in sys.path:
        sys.path.insert(0, str(generation))
    try:
        from vendor.unet import UNet3DConditionModel
    except (ImportError, RuntimeError) as exc:
        raise PipelineError(f"Cannot import the release's vendored temporal UNet: {exc}") from exc

    emit_stage(
        callback,
        "Diffusion preparation",
        "running",
        f"Loading the verified local UNet and scheduler on CUDA; seed {seed}, 100 DDIM steps.",
    )
    started = time.perf_counter()
    net = UNet3DConditionModel.from_pretrained(
        str(diffusion),
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    try:
        net.enable_xformers_memory_efficient_attention()
    except (ImportError, RuntimeError, ValueError) as exc:
        del net
        torch.cuda.empty_cache()
        raise PipelineError(f"Verified xFormers attention could not be enabled: {exc}") from exc
    if not all(torch.isfinite(parameter).all().item() for parameter in net.parameters()):
        del net
        torch.cuda.empty_cache()
        raise PipelineError("The final UNet contains non-finite parameters.")

    scheduler = DDIMScheduler.from_pretrained(
        str(diffusion),
        subfolder="scheduler",
        clip_sample=False,
        steps_offset=1,
        set_alpha_to_one=False,
        local_files_only=True,
    )
    negative_array = np.load(paths.negative_embedding, allow_pickle=False)
    if negative_array.shape != (1, 77, 768) or not np.isfinite(negative_array).all():
        raise PipelineError("Verified negative embedding must be finite with shape (1, 77, 768).")
    negative = torch.from_numpy(negative_array).to(device=device, dtype=dtype)

    # Preserve the final run's released500 DANA noise schedule and RNG order.
    released_alphas = torch.cumprod(
        1 - torch.linspace(0.0001, 0.02, 500, dtype=torch.float32),
        dim=0,
    )
    scheduler.set_timesteps(100, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.from_numpy(np.array(predictions["visual"], copy=True))
    latent = latent.permute(1, 0, 2, 3)[None].to(device=device, dtype=torch.float32)
    label = int(predictions["dynamic_label"])
    beta = 0.2 if label == 1 else 0.3
    dana_timestep = int(torch.randint(0, 500, (1,), generator=generator, device=device).item())
    alpha = float(released_alphas[dana_timestep])
    diverse = torch.randn(
        (1, 6, 4, 36, 64),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).permute(0, 2, 1, 3, 4)
    shared = torch.randn(
        (1, 1, 4, 36, 64),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).permute(0, 2, 1, 3, 4)
    noise_bytes = torch.cat([shared.flatten(), diverse.flatten()]).cpu().numpy().tobytes()
    noise_hash = hashlib.sha256(noise_bytes).hexdigest()
    latent = (
        alpha**0.5 * latent.float()
        + (1 - alpha) ** 0.5 * (beta**0.5 * shared + (1 - beta) ** 0.5 * diverse)
    ).to(dtype)
    latent = latent * scheduler.init_noise_sigma
    initial_latent_hash = hashlib.sha256(latent.float().cpu().numpy().tobytes()).hexdigest()
    conditioning = torch.from_numpy(np.array(predictions["semantic"], copy=True))[None].to(
        device=device,
        dtype=dtype,
    )
    first_ddim_timestep = int(scheduler.timesteps[0])
    recorded_initial_latent = None
    if test_index is not None:
        recorded_initial_latent = _verify_recorded_noise(
            paths,
            test_index,
            seed,
            label,
            beta,
            dana_timestep,
            alpha,
            first_ddim_timestep,
            noise_hash,
            latent,
            shared,
            diverse,
            float(scheduler.init_noise_sigma),
        )

    emit_stage(callback, "Diffusion preparation", "complete", "Model, scheduler, DANA noise and conditioning are ready.")
    emit_stage(callback, "Diffusion generation", "running", f"DANA timestep {dana_timestep}; starting 100 DDIM steps.")
    emit_progress(callback, "Diffusion generation", 0, 100, f"DANA timestep {dana_timestep}; DDIM step 0/100.")
    timer = StepTimer(100)
    try:
        with torch.inference_mode():
            for step_index, timestep in enumerate(scheduler.timesteps, start=1):
                model_input = scheduler.scale_model_input(latent, timestep)
                unconditional = net(
                    model_input,
                    timestep,
                    encoder_hidden_states=negative,
                ).sample
                conditional = net(
                    model_input,
                    timestep,
                    encoder_hidden_states=conditioning,
                ).sample
                predicted = unconditional + 12.5 * (conditional - unconditional)
                latent = scheduler.step(predicted, timestep, latent, eta=0.0, generator=generator).prev_sample
                if not torch.isfinite(latent).all().item():
                    raise PipelineError(f"Non-finite latents during DDIM step {step_index}/100.")
                snapshot = timer.complete_step()
                emit_progress(
                    callback,
                    "Diffusion generation",
                    snapshot["completed"],
                    snapshot["total"],
                    f"DANA timestep {dana_timestep}; DDIM step {step_index}/100.",
                    elapsed_seconds=snapshot["elapsed_seconds"],
                    eta_seconds=snapshot["eta_seconds"],
                    average_step_seconds=snapshot["average_step_seconds"],
                )
                if step_index % 10 == 0:
                    emit_stage(
                        callback,
                        "Diffusion generation",
                        "running",
                        f"DANA timestep {dana_timestep}; DDIM step {step_index}/100.",
                    )
    except torch.cuda.OutOfMemoryError as exc:
        del net
        torch.cuda.empty_cache()
        raise PipelineError(
            "CUDA ran out of memory during the verified 512x288, six-frame generation. "
            "The application did not lower resolution, frame count, or DDIM steps."
        ) from exc

    del net, scheduler, negative, conditioning, model_input, unconditional, conditional, predicted
    torch.cuda.empty_cache()
    emit_stage(callback, "Diffusion generation", "complete", "Completed all 100 DDIM steps with the released500 DANA schedule.")

    emit_stage(callback, "Video decoding", "running", "Decoding six frames with the pinned VAE on CPU.")
    final_media = _import_source(
        "eeg2video_original_final_media",
        RELEASE_ROOT / "EEG2Video" / "EEG2Video_New" / "FinalEvaluation" / "final_media.py",
    )
    vae = AutoencoderKL.from_pretrained(
        str(sd14_model),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval()
    # Count completed frames by observing the original per-frame decode call; the call itself is unchanged.
    original_decode = vae.decode
    decoded_frames = {"count": 0}
    decode_started = time.monotonic()
    emit_progress(callback, "Video decoding", 0, 6, "Decoding frame 0/6.", unit="frame")

    def counted_decode(*args, **kwargs):
        output = original_decode(*args, **kwargs)
        decoded_frames["count"] += 1
        emit_progress(
            callback,
            "Video decoding",
            decoded_frames["count"],
            6,
            f"Decoded frame {decoded_frames['count']}/6.",
            unit="frame",
            elapsed_seconds=time.monotonic() - decode_started,
        )
        return output

    vae.decode = counted_decode
    rgb = final_media.decode(
        vae,
        latent[0].permute(1, 0, 2, 3).float().cpu().numpy(),
        EXPECTED_LATENT_SCALE,
        device="cpu",
    )
    del vae, latent, shared, diverse
    gc.collect()
    torch.cuda.empty_cache()
    if rgb.shape != (6, 288, 512, 3) or rgb.dtype != np.uint8 or not np.isfinite(rgb).all():
        raise PipelineError(f"VAE decoding returned invalid RGB output: {rgb.shape}, {rgb.dtype}.")
    emit_stage(callback, "Video decoding", "complete", "Decoded six uint8 RGB frames at 512x288.")
    metadata = {
        "seed": seed,
        "known_block7_seed": test_index is not None,
        "dynamic_label": label,
        "beta": beta,
        "dana_schedule": "released500",
        "dana_timestep": dana_timestep,
        "dana_alpha_cumprod": alpha,
        "first_ddim_timestep": first_ddim_timestep,
        "ddim_steps": 100,
        "guidance_scale": 12.5,
        "fps": 3,
        "frames": 6,
        "height": 288,
        "width": 512,
        "dtype": "torch.float16",
        "attention": "xformers",
        "device": str(device),
        "noise_sha256": noise_hash,
        "initial_latent_sha256": initial_latent_hash,
        "historical_initial_latent_comparison": recorded_initial_latent,
        "duration_seconds": time.perf_counter() - started,
        "diffusion_unet_sha256": EXPECTED_ARTIFACT_HASHES["diffusion_unet"],
        "scheduler_sha256": EXPECTED_ARTIFACT_HASHES["diffusion_scheduler"],
        "negative_embedding_sha256": EXPECTED_NEGATIVE_EMBEDDING_SHA256,
        "base_model_revision": "133a221b8aa7292a167afc5127cb63fb5005638b",
        "gpu_total_bytes": int(total_vram),
        "gpu_free_bytes_before_generation": int(free_vram),
        "torch_version": torch.__version__,
        "diffusers_version": importlib.metadata.version("diffusers"),
    }
    return rgb, metadata


def _existing_generated_comparison(paths: AssetPaths, sample_id: str, generated: np.ndarray) -> dict | None:
    directory = paths.generation_output / "clips" / sample_id
    record_path = directory / "complete.json"
    generated_path = directory / "generated_rgb.npy"
    if not record_path.is_file() or not generated_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read existing generation record for {sample_id}.") from exc
    if record.get("sample_id") != sample_id or record.get("ground_truth_conditioning") is not False:
        raise PipelineError(f"Existing generation provenance is invalid for {sample_id}.")
    if sha256_file(generated_path) != record.get("generated_sha256"):
        raise PipelineError(f"Existing generated output hash is invalid for {sample_id}.")
    existing = _load_rgb(generated_path, "Existing generated")
    difference = np.abs(generated.astype(np.float32) - existing.astype(np.float32))
    return {
        "existing_run_seed": record.get("seed"),
        "pixel_identical": bool(np.array_equal(generated, existing)),
        "mean_absolute_difference_0_255": float(difference.mean()),
        "max_absolute_difference_0_255": float(difference.max(initial=0)),
        "existing_generated_sha256": record["generated_sha256"],
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_live(
    upload_path: str | Path,
    selected_index: int,
    sampling_rate_hz: float,
    channels_confirmed: bool,
    unknown_seed: int,
    run_dir: Path,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    paths = get_asset_paths()
    emit_stage(callback, "EEG validation", "running", "Checking the .npy input, sampling rate and recorded channel order.")
    verify_channel_file(paths.channel_file)
    uploaded = validate_upload_file(upload_path, sampling_rate_hz, channels_confirmed)
    canonical = None
    test_index: int | None = None
    eeg_mapping = paired_mapping = None
    if paths.test_eeg.is_file():
        canonical = load_verified_test_eeg(paths.test_eeg)
        eeg_rows, paired_rows = load_verified_mappings(paths)
        eeg_mapping, paired_mapping = eeg_rows, paired_rows
    elif uploaded.shape == (200, 62, 400):
        raise PipelineError("Cannot verify a full 200-sample upload because canonical Block 7 EEG is unavailable.")

    if canonical is not None:
        sample, matched_index = identify_sample(uploaded, canonical, selected_index)
        if matched_index >= 0:
            test_index = matched_index
    else:
        if uploaded.shape == (200, 62, 400):
            raise PipelineError("The full test array could not be verified against canonical Block 7 EEG.")
        sample = uploaded[0] if uploaded.shape == (1, 62, 400) else uploaded
    emit_stage(callback, "EEG validation", "complete", f"Validated a single (62, 400) signal at {sampling_rate_hz:g} Hz.")
    if test_index is not None:
        sample_id = eeg_mapping[1200 + test_index]["sample_id"]
        if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise PipelineError("The canonical sample mapping produced an unexpected Block 7 ID.")
        emit_stage(callback, "Sample identification", "complete", f"Exact match: {sample_id} (test index {test_index}).")
    else:
        sample_id = None
        emit_stage(
            callback,
            "Sample identification",
            "complete",
            "No exact canonical Block 7 match. A user-provided seed will be used; no reference video or historical metrics will be shown.",
        )

    started = time.perf_counter()
    prediction = predict_single_sample(sample, paths, callback)
    verification = compare_predictions(prediction, paths, test_index)
    prediction["verification"] = verification
    if test_index is not None and verification.get("status") != "passed":
        raise PipelineError(
            "Live predictor outputs did not meet the numerical agreement check against the saved "
            "Block 7 predictions. Diffusion generation was not started."
        )
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir / "visual_latents.npy", prediction["visual"], allow_pickle=False)
    np.save(predictions_dir / "semantic_embeddings.npy", prediction["semantic"], allow_pickle=False)
    np.save(
        predictions_dir / "dynamic_label.npy",
        np.asarray(prediction["dynamic_label"], dtype=np.int64),
        allow_pickle=False,
    )

    rgb, generation_metadata = _run_single_sample_diffusion(
        prediction,
        paths,
        test_index,
        unknown_seed,
        run_dir,
        callback,
    )
    generated_rgb_path = run_dir / "generated_rgb.npy"
    np.save(generated_rgb_path, rgb, allow_pickle=False)
    generated_mp4 = run_dir / "generated_live.mp4"
    write_mp4(generated_mp4, rgb)

    reference_video = None
    metrics: dict | None = None
    reference_hash = None
    comparison = None
    if sample_id is not None and paired_mapping is not None:
        emit_stage(callback, "Evaluation", "running", "Loading the matched reference only after live prediction and generation have completed.")
        paired = paired_mapping[1200 + test_index]
        reference_path = safe_data_path(paths.data_root, paired["rgb_array"])
        completion_path = paths.generation_output / "clips" / sample_id / "complete.json"
        if completion_path.is_file():
            existing_completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
            reference_hash = existing_completion.get("reference_sha256")
            if existing_completion.get("ground_truth_conditioning") is not False:
                raise PipelineError("Existing completion record does not confirm ground-truth-free conditioning.")
        if not reference_path.is_file():
            raise PipelineError(f"Verified ground-truth RGB is missing for {sample_id}.")
        # Reference data is intentionally first loaded after prediction and diffusion have completed.
        reference = _load_rgb(reference_path, "Ground-truth")
        actual_reference_hash = sha256_file(reference_path)
        if reference_hash is not None and reference_hash != actual_reference_hash:
            raise PipelineError(f"Ground-truth RGB hash differs from the completed record for {sample_id}.")
        reference_hash = actual_reference_hash
        metrics = compute_metrics(reference, rgb)
        metrics.update(
            {
                "source": "Computed from this live generation and the verified reference; not the historical metric row.",
                "sample_id": sample_id,
                "test_index": test_index,
                "temporal_difference_note": "Descriptive adjacent-frame pixel-change difference; not motion accuracy.",
            }
        )
        ground_truth_mp4 = run_dir / "ground_truth.mp4"
        write_mp4(ground_truth_mp4, reference)
        reference_video = str(ground_truth_mp4)
        comparison = _existing_generated_comparison(paths, sample_id, rgb)
    else:
        emit_stage(
            callback,
            "Evaluation",
            "not_available",
            "No verified reference is associated with this unmatched input; no metrics were computed.",
        )

    if metrics is not None:
        emit_stage(callback, "Evaluation", "complete", "Computed metrics for this new output; historical metrics were not substituted.")
    prediction_metadata = {
        "shapes": {
            "visual": list(prediction["visual"].shape),
            "semantic": list(prediction["semantic"].shape),
            "dynamic": [],
        },
        "dynamic_label": prediction["dynamic_label"],
        "dynamic_probabilities": prediction["dynamic_probabilities"],
        "verification": verification,
        "model_hashes": prediction["model_hashes"],
        "prediction_hashes": {
            "visual": sha256_file(predictions_dir / "visual_latents.npy"),
            "semantic": sha256_file(predictions_dir / "semantic_embeddings.npy"),
            "dynamic": sha256_file(predictions_dir / "dynamic_label.npy"),
        },
    }
    result = {
        "mode": "Live EEG Inference — New Model Prediction and Generation",
        "run_id": run_dir.name,
        "sample_id": sample_id,
        "test_index": test_index,
        "seed": generation_metadata["seed"],
        "ground_truth_video": reference_video,
        "generated_video": str(generated_mp4),
        "download_video": str(generated_mp4),
        "generated_rgb": str(generated_rgb_path),
        "metrics": metrics,
        "prediction": prediction_metadata,
        "generation": generation_metadata,
        "comparison_to_existing_generated_video": comparison,
        "total_inference_seconds": time.perf_counter() - started,
        "message": (
            "A new result was generated from EEG-predicted conditions."
            if sample_id
            else "A new result was generated from the supplied EEG. No Block 7 identity or reference was inferred."
        ),
    }
    _write_json(run_dir / "run_metadata.json", result)
    _write_json(run_dir / "result.json", result)
    return result
