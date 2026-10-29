from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


RELEASE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHANNEL_FILE_SHA256 = "c79c149c3c36812856f3991d821b0fd3465ead0734319e5b1b47dae979e7b727"
EXPECTED_TEST_EEG_SHA256 = "0d5fb7505a04a49a4d3bf6c6e857c74bf702ee6bd8722b7a7ac5b40be3cfec25"
EXPECTED_SAMPLE_MAPPING_SHA256 = "25d2fc715241c9c74f6b3956f45b4e1a70fd74bea7964e8b8ac69c0a01803ca7"
EXPECTED_PAIRED_MANIFEST_SHA256 = "a5a258cb228247d29aed8afd38bbf4780aa51df70b3a8b9233baf525e7f4f6b8"
EXPECTED_NEGATIVE_EMBEDDING_SHA256 = "dfa6fb14a799325a26b424cb3e8f3e6aec45a84d070a2c2c31373c1cb988716b"
EXPECTED_LATENT_SCALE = 0.18215
EXPECTED_LOCAL_CHANNELS = ["PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
SAMPLE_ID_PATTERN = re.compile(r"^sub01_session01_block07_clip\d{3}$")

STAGES = (
    "EEG validation",
    "Sample identification",
    "Visual prediction",
    "Semantic prediction",
    "Dynamic prediction",
    "Diffusion preparation",
    "Diffusion generation",
    "Video decoding",
    "Evaluation",
)


class PipelineError(RuntimeError):
    """An expected, user-actionable failure in the local pipeline."""


@dataclass(frozen=True)
class AssetPaths:
    data_root: Path
    model_root: Path
    cache_root: Path
    channel_file: Path
    test_eeg: Path
    sample_mapping: Path
    paired_manifest: Path
    generation_output: Path
    metrics_csv: Path
    diffusion_output: Path
    visual_output: Path
    semantic_output: Path
    dynamic_output: Path
    negative_embedding: Path
    sd14_model: Path | None


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _configured_asset_path(root: Path, value: object, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"Asset configuration field {key!r} must be a non-empty relative path.")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        if value != ".":
            raise PipelineError(f"Asset configuration field {key!r} must stay inside EEG2VIDEO_ASSETS_DIR.")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PipelineError(f"Asset configuration field {key!r} escapes EEG2VIDEO_ASSETS_DIR.") from exc
    return resolved


def get_asset_paths() -> AssetPaths:
    configured_root = os.environ.get("EEG2VIDEO_ASSETS_DIR")
    if not configured_root:
        raise PipelineError(
            "EEG2VIDEO_ASSETS_DIR is not set. Point it to the private runtime asset directory; "
            "the application does not fall back to another folder."
        )
    asset_root = Path(configured_root).expanduser().resolve()
    if not asset_root.is_dir():
        raise PipelineError(f"Configured EEG2VIDEO_ASSETS_DIR does not exist: {asset_root}")

    config_value = os.environ.get("EEG2VIDEO_ASSET_CONFIG")
    config_path = (
        Path(config_value).expanduser().resolve()
        if config_value
        else asset_root / "asset_config.json"
    )
    if not config_path.is_file():
        raise PipelineError(
            f"Asset configuration is missing: {config_path}. Copy "
            "webapp/asset_config.example.json to the configured asset directory as asset_config.json "
            "and edit only relative paths."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read asset configuration {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise PipelineError("Asset configuration must contain a JSON object.")

    required_keys = {
        "data_root",
        "model_root",
        "channel_file",
        "test_eeg",
        "sample_mapping",
        "paired_manifest",
        "generation_output",
        "metrics_csv",
        "diffusion_output",
        "visual_output",
        "semantic_output",
        "dynamic_output",
        "negative_embedding",
    }
    missing_keys = sorted(required_keys - set(config))
    if missing_keys:
        raise PipelineError("Asset configuration is missing fields: " + ", ".join(missing_keys))

    data_root = _configured_asset_path(asset_root, config["data_root"], "data_root")
    model_root = _configured_asset_path(asset_root, config["model_root"], "model_root")
    data_keys = {
        "channel_file",
        "test_eeg",
        "sample_mapping",
        "paired_manifest",
        "generation_output",
        "metrics_csv",
        "negative_embedding",
    }
    resolved = {
        key: _configured_asset_path(data_root if key in data_keys else model_root, config[key], key)
        for key in required_keys - {"data_root", "model_root"}
    }
    sd14_value = config.get("sd14_model")
    sd14_model = None if sd14_value is None else _configured_asset_path(model_root, sd14_value, "sd14_model")
    cache_root = _env_path("EEG2VIDEO_CACHE_DIR", RELEASE_ROOT / "webapp" / ".cache")
    return AssetPaths(
        data_root=data_root,
        model_root=model_root,
        cache_root=cache_root,
        channel_file=resolved["channel_file"],
        test_eeg=resolved["test_eeg"],
        sample_mapping=resolved["sample_mapping"],
        paired_manifest=resolved["paired_manifest"],
        generation_output=resolved["generation_output"],
        metrics_csv=resolved["metrics_csv"],
        diffusion_output=resolved["diffusion_output"],
        visual_output=resolved["visual_output"],
        semantic_output=resolved["semantic_output"],
        dynamic_output=resolved["dynamic_output"],
        negative_embedding=resolved["negative_embedding"],
        sd14_model=sd14_model,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit_stage(
    callback: Callable[[dict], None] | None,
    stage: str,
    status: str,
    detail: str,
) -> None:
    if callback:
        callback({"type": "stage", "stage": stage, "status": status, "detail": detail})


def safe_data_path(data_root: Path, relative_path: str) -> Path:
    value = relative_path.replace("\\", "/")
    candidate = Path(value)
    if candidate.is_absolute() or any(part in ("", ".", "..") for part in candidate.parts):
        raise PipelineError(f"Manifest contains an unsafe local path: {relative_path!r}")
    resolved = (data_root / candidate).resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as exc:
        raise PipelineError(f"Manifest path escapes the configured data directory: {relative_path!r}") from exc
    return resolved


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise PipelineError(f"Cannot read required mapping file: {path.name}") from exc


def verify_channel_file(path: Path) -> list[str]:
    if not path.is_file():
        raise PipelineError(
            "The recorded 62-channel order file is missing. Set EEG2VIDEO_ASSETS_DIR and "
            "check its asset_config.json channel_file path."
        )
    if sha256_file(path) != EXPECTED_CHANNEL_FILE_SHA256:
        raise PipelineError("The channel-order file does not match the verified training file.")
    names = [line.split()[-1] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != 62 or names[50:62] != EXPECTED_LOCAL_CHANNELS:
        raise PipelineError("The channel-order file has an unexpected layout.")
    return names


def validate_upload_file(path: str | Path, sampling_rate_hz: float, channels_confirmed: bool) -> np.ndarray:
    upload = Path(path)
    if upload.suffix.lower() != ".npy":
        raise PipelineError("Upload must be a NumPy .npy file.")
    if not upload.is_file():
        raise PipelineError("The uploaded EEG file is no longer available. Upload it again.")
    size = upload.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise PipelineError(f"Upload is {size / 1024 / 1024:.1f} MiB; the limit is 64 MiB.")
    if size <= 0:
        raise PipelineError("The uploaded file is empty.")
    try:
        value = np.load(upload, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"Could not load a valid, non-pickled .npy array: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise PipelineError("The upload must contain one NumPy array.")
    if not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)):
        raise PipelineError(f"EEG values must use a real numeric dtype; received {value.dtype}.")
    if not np.isfinite(value).all():
        raise PipelineError("EEG contains NaN or infinity.")
    allowed_shapes = ((62, 400), (1, 62, 400), (200, 62, 400))
    if value.shape not in allowed_shapes:
        raise PipelineError(
            f"Unsupported EEG shape {value.shape}. Expected (62, 400), (1, 62, 400), "
            "or the verified complete test array (200, 62, 400). No reshape or resampling is applied."
        )
    if sampling_rate_hz != 200:
        raise PipelineError("Sampling rate must be exactly 200 Hz; this application does not resample EEG.")
    if not channels_confirmed:
        raise PipelineError("Confirm that the input channels use the recorded project channel order.")
    if value.dtype.itemsize > 8:
        raise PipelineError(f"Unsupported numeric dtype width: {value.dtype}.")
    return value


def load_verified_test_eeg(path: Path) -> np.ndarray:
    if not path.is_file():
        raise PipelineError(f"Canonical Block 7 EEG is missing: {path}")
    if sha256_file(path) != EXPECTED_TEST_EEG_SHA256:
        raise PipelineError("Canonical Block 7 EEG hash differs from the verified project file.")
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"Cannot load canonical Block 7 EEG: {exc}") from exc
    if value.shape != (200, 62, 400) or not np.issubdtype(value.dtype, np.floating):
        raise PipelineError(f"Canonical Block 7 EEG has unexpected shape/dtype: {value.shape}, {value.dtype}.")
    return value


def identify_sample(uploaded: np.ndarray, canonical: np.ndarray, selected_index: int) -> tuple[np.ndarray, int]:
    if uploaded.shape == (200, 62, 400):
        if not 0 <= selected_index < 200:
            raise PipelineError("Test index must be between 0 and 199.")
        if not np.array_equal(uploaded, canonical):
            raise PipelineError(
                "A full-array upload is accepted only when it exactly matches the verified complete "
                "Block 7 test array. No approximate array matching is performed."
            )
        sample = uploaded[selected_index]
        return sample, selected_index

    sample = uploaded[0] if uploaded.shape == (1, 62, 400) else uploaded
    matches = [index for index in range(200) if np.array_equal(sample, canonical[index])]
    if len(matches) > 1:
        raise PipelineError("The EEG matches multiple Block 7 samples; exact identification is ambiguous.")
    return sample, matches[0] if matches else -1


def load_verified_mappings(paths: AssetPaths) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    for path, digest, label in (
        (paths.sample_mapping, EXPECTED_SAMPLE_MAPPING_SHA256, "EEG sample mapping"),
        (paths.paired_manifest, EXPECTED_PAIRED_MANIFEST_SHA256, "paired-video manifest"),
    ):
        if not path.is_file():
            raise PipelineError(f"Required {label} is missing: {path}")
        if sha256_file(path) != digest:
            raise PipelineError(f"The {label} does not match the verified run.")
    eeg_rows = _read_csv(paths.sample_mapping)
    paired_rows = _read_csv(paths.paired_manifest)
    if len(eeg_rows) != 1400 or len(paired_rows) != 1400:
        raise PipelineError("Expected 1,400 rows in each verified sample mapping.")
    for index, (eeg_row, paired_row) in enumerate(zip(eeg_rows, paired_rows)):
        sample_id = f"sub01_session01_block{index // 200 + 1:02d}_clip{index % 200 + 1:03d}"
        split = "train" if index < 1200 else "test"
        split_index = index if index < 1200 else index - 1200
        expected = (sample_id, split, str(split_index))
        for row in (eeg_row, paired_row):
            if (row.get("sample_id"), row.get("split"), row.get("split_index")) != expected:
                raise PipelineError(f"Sample mapping order or split is invalid at row {index}.")
        if eeg_row["sample_id"] != paired_row["sample_id"]:
            raise PipelineError(f"EEG/video mappings disagree at row {index}.")
    return eeg_rows, paired_rows


def _load_rgb(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise PipelineError(f"{label} RGB array is missing: {path}")
    try:
        frames = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise PipelineError(f"Cannot load {label} RGB array: {exc}") from exc
    if frames.shape != (6, 288, 512, 3) or frames.dtype != np.uint8:
        raise PipelineError(f"{label} RGB array must be uint8 with shape (6, 288, 512, 3).")
    return frames


def compute_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float | bool | None]:
    from skimage.metrics import structural_similarity

    for value in (reference, generated):
        if value.shape != (6, 288, 512, 3) or value.dtype != np.uint8:
            raise PipelineError("Metrics require uint8 RGB arrays with shape (6, 288, 512, 3).")
    ref = reference.astype(np.float64) / 255.0
    gen = generated.astype(np.float64) / 255.0
    mse = float(np.mean((ref - gen) ** 2))
    return {
        "rgb_mse_01": mse,
        "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
        "ssim_rgb_7x7": float(
            np.mean(
                [
                    structural_similarity(
                        reference[index],
                        generated[index],
                        channel_axis=-1,
                        data_range=255,
                        win_size=7,
                        gaussian_weights=False,
                        use_sample_covariance=True,
                    )
                    for index in range(6)
                ]
            )
        ),
        "temporal_difference_mae_255": float(np.abs(np.diff(gen, axis=0) - np.diff(ref, axis=0)).mean() * 255),
    }


def _metric_record(path: Path, sample_id: str, test_index: int) -> dict[str, str]:
    rows = _read_csv(path)
    matches = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matches) != 1 or matches[0].get("test_index") != str(test_index):
        raise PipelineError(f"No unique completed evaluation metric row for {sample_id}.")
    return matches[0]


def _check_recorded_metrics(actual: dict[str, float | bool | None], recorded: dict[str, str]) -> None:
    for key, value in actual.items():
        if key == "psnr_db" and value is None:
            if recorded.get(key, "") not in ("", "None", "null"):
                raise PipelineError("Saved PSNR record disagrees with an exact RGB match.")
            continue
        try:
            saved = float(recorded[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"Saved evaluation record has no valid {key} value.") from exc
        if not np.isclose(float(value), saved, rtol=1e-9, atol=1e-12):
            raise PipelineError(f"Saved {key} metric does not agree with the completed RGB arrays.")


def write_mp4(path: Path, frames: np.ndarray) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        frames,
        fps=3,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def run_precomputed(
    upload_path: str | Path,
    selected_index: int,
    sampling_rate_hz: float,
    channels_confirmed: bool,
    run_dir: Path,
    callback: Callable[[dict], None] | None = None,
) -> dict:
    emit_stage(callback, "EEG validation", "running", "Checking the .npy upload, signal shape, sampling rate and channel-order confirmation.")
    paths = get_asset_paths()
    verify_channel_file(paths.channel_file)
    uploaded = validate_upload_file(upload_path, sampling_rate_hz, channels_confirmed)
    canonical = load_verified_test_eeg(paths.test_eeg)
    eeg_rows, paired_rows = load_verified_mappings(paths)
    emit_stage(callback, "EEG validation", "complete", f"Validated a 62-channel, 400-sample signal at {sampling_rate_hz:g} Hz.")

    emit_stage(callback, "Sample identification", "running", "Comparing signal values to the verified Block 7 test array; filename is not used.")
    sample, test_index = identify_sample(uploaded, canonical, selected_index)
    if test_index < 0:
        emit_stage(callback, "Sample identification", "failed", "No exact Block 7 match was found.")
        raise PipelineError("Precomputed mode requires an exact match to one of the known Block 7 EEG samples.")
    mapping = eeg_rows[1200 + test_index]
    paired = paired_rows[1200 + test_index]
    sample_id = mapping["sample_id"]
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise PipelineError("The verified mapping produced an unexpected Block 7 sample identifier.")
    if not np.array_equal(sample, canonical[test_index]):
        raise PipelineError("The selected EEG values changed during sample identification.")
    emit_stage(callback, "Sample identification", "complete", f"Exact match: {sample_id} (test index {test_index}).")
    for stage in ("Visual prediction", "Semantic prediction", "Dynamic prediction", "Diffusion preparation", "Diffusion generation"):
        emit_stage(callback, stage, "not_run", "Not run: this mode retrieves an existing evaluation result.")

    result_folder = paths.generation_output / "clips" / sample_id
    completion_file = result_folder / "complete.json"
    generated_file = result_folder / "generated_rgb.npy"
    if not completion_file.is_file():
        raise PipelineError(f"No completed generated result exists for {sample_id}.")
    try:
        completion = json.loads(completion_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read the completion record for {sample_id}.") from exc
    if completion.get("sample_id") != sample_id or completion.get("ground_truth_conditioning") is not False:
        raise PipelineError(f"Completion provenance is invalid for {sample_id}.")
    if completion.get("seed") != 42 + test_index or completion.get("protocol") != "released500":
        raise PipelineError(f"Completion seed or DANA protocol is invalid for {sample_id}.")
    dynamic_label = completion.get("dynamic_label")
    expected_beta = 0.2 if dynamic_label == 1 else 0.3 if dynamic_label == 0 else None
    if expected_beta is None or completion.get("beta") != expected_beta:
        raise PipelineError(f"Completion dynamic label or beta is invalid for {sample_id}.")
    reference_file = safe_data_path(paths.data_root, paired["rgb_array"])

    # Ground truth is loaded only after an exact EEG-to-sample match and generated-result lookup.
    generated = _load_rgb(generated_file, "Precomputed generated")
    if sha256_file(generated_file) != completion.get("generated_sha256"):
        raise PipelineError(f"The saved generated RGB hash does not match its completion record for {sample_id}.")
    reference = _load_rgb(reference_file, "Ground-truth")
    if sha256_file(reference_file) != completion.get("reference_sha256"):
        raise PipelineError(f"The ground-truth RGB hash does not match its completion record for {sample_id}.")

    emit_stage(callback, "Video decoding", "running", "Encoding separate playback files from the matched RGB arrays.")
    metric_row = _metric_record(paths.metrics_csv, sample_id, test_index)
    emit_stage(callback, "Evaluation", "running", "Verifying the recorded per-clip metrics against the completed RGB arrays.")
    measured = compute_metrics(reference, generated)
    _check_recorded_metrics(measured, metric_row)
    metrics = {key: (None if key == "psnr_db" and value == "" else value) for key, value in metric_row.items() if key in (
        "rgb_mse_01", "psnr_db", "ssim_rgb_7x7", "temporal_difference_mae_255"
    )}
    for key, value in metrics.items():
        if value is not None:
            metrics[key] = float(value)
    metrics.update(
        {
            "source": "Existing completed evaluation record; no model inference was run.",
            "sample_id": sample_id,
            "test_index": test_index,
            "temporal_difference_note": "Descriptive adjacent-frame pixel-change difference; not motion accuracy.",
        }
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_mp4 = run_dir / "ground_truth.mp4"
    generated_mp4 = run_dir / "generated_precomputed.mp4"
    write_mp4(ground_truth_mp4, reference)
    write_mp4(generated_mp4, generated)
    emit_stage(callback, "Video decoding", "complete", "Converted the existing six-frame RGB arrays to separate 3-fps MP4 files.")
    emit_stage(callback, "Evaluation", "complete", "Retrieved the saved per-clip metrics and verified them against the RGB arrays.")
    result = {
        "mode": "Precomputed Reconstruction — Existing Evaluation Result",
        "sample_id": sample_id,
        "test_index": test_index,
        "seed": completion.get("seed"),
        "ground_truth_video": str(ground_truth_mp4),
        "generated_video": str(generated_mp4),
        "download_video": str(generated_mp4),
        "metrics": metrics,
        "message": "Existing result retrieved. Visual, semantic, dynamic and diffusion inference were not run.",
        "completion_record": completion,
        "eeg_sample_mapping": mapping,
    }
    result_path = run_dir / "result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(result_path)
    return result
