from __future__ import annotations

import html
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import gradio as gr

from webapp.core import PipelineError, RELEASE_ROOT, STAGES, get_asset_paths
from webapp.progress import PipelineProgress

PROGRESS_REFRESH_SECONDS = 1.0
APP_CSS = """
:root {
  color-scheme: dark;
  --e2v-bg: #151719;
  --e2v-panel: #202326;
  --e2v-panel-raised: #262a2e;
  --e2v-border: #383d42;
  --e2v-text: #f0f2f3;
  --e2v-muted: #a7afb5;
  --e2v-accent: #f28a36;
  --e2v-success: #55c99a;
  --e2v-warning: #f0b45a;
}
body, .gradio-container {
  background: var(--e2v-bg) !important;
  color: var(--e2v-text) !important;
}
.gradio-container {
  max-width: 1760px !important;
  padding: 18px 26px 26px !important;
}
#e2v-header { align-items: center; margin-bottom: 12px; }
#e2v-header h1 { margin: 0; font-size: 30px; line-height: 1.15; }
#e2v-header p { color: var(--e2v-muted); margin: 4px 0 0; font-size: 15px; }
#e2v-readiness { text-align: right; }
.e2v-ready-badge {
  display: inline-block; border: 1px solid var(--e2v-border); border-radius: 999px;
  padding: 7px 12px; font-size: 13px; font-weight: 650; white-space: nowrap;
}
.e2v-ready { color: var(--e2v-success); }
.e2v-limited, .e2v-unavailable { color: var(--e2v-warning); }
#e2v-readiness-warning { margin: 0 0 10px; color: var(--e2v-warning); font-size: 13px; }
#e2v-dashboard { gap: 18px; align-items: flex-start; }
#e2v-inputs {
  background: var(--e2v-panel); border: 1px solid var(--e2v-border);
  border-radius: 12px; padding: 16px;
}
#e2v-inputs h3, #e2v-results h3 { margin-top: 0; }
#e2v-inputs .wrap, #e2v-results .wrap { min-width: 0; }
#e2v-results { min-width: 0; }
#e2v-video-row { gap: 14px; }
#e2v-ground-truth, #e2v-generated-video { min-width: 0; }
#e2v-ground-truth .video-container, #e2v-generated-video .video-container {
  width: 100%; aspect-ratio: 16 / 9; min-height: 0 !important;
}
#e2v-ground-truth .empty, #e2v-generated-video .empty {
  width: 100%; height: auto !important; min-height: 0 !important;
  aspect-ratio: 16 / 9; display: flex; flex-direction: column;
  justify-content: center; gap: 8px; background: #090a0b;
}
#e2v-ground-truth .empty::after, #e2v-generated-video .empty::after {
  content: "Waiting for a reconstruction"; color: #a7afb5;
  font-size: 13px; text-align: center;
}
#e2v-ground-truth video, #e2v-generated-video video {
  width: 100%; height: auto !important; aspect-ratio: 16 / 9;
  object-fit: contain; background: #090a0b;
}
#e2v-download .empty { display: none; }
.e2v-status {
  color: var(--e2v-text); background: var(--e2v-panel-raised);
  border-left: 3px solid var(--e2v-success); border-radius: 5px;
  padding: 8px 11px; font-size: 14px;
}
.e2v-status-error { border-left-color: #ef7777; }
.e2v-identity { color: var(--e2v-muted); margin-left: 8px; }
.e2v-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.e2v-metric-card {
  background: var(--e2v-panel); border: 1px solid var(--e2v-border);
  border-radius: 9px; padding: 10px 12px; min-width: 0;
}
.e2v-metric-label, .e2v-metric-unit { display: block; color: var(--e2v-muted); }
.e2v-metric-label { font-size: 12px; line-height: 1.35; min-height: 2.5em; }
.e2v-metric-value { display: block; font-size: 21px; line-height: 1.2; margin: 4px 0; }
.e2v-metric-unit { font-size: 11px; }
.e2v-diagnostics { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }
#e2v-diagnostics { margin-top: 14px; }
#e2v-download { margin-top: 7px; }
#e2v-progress { margin-top: 12px; }
#e2v-seed-settings { margin: 4px 0 8px; }
#e2v-sampling-rate { display: none; }
@media (max-width: 850px) {
  .gradio-container { padding: 14px !important; }
  #e2v-dashboard { flex-direction: column; }
  #e2v-inputs, #e2v-results { width: 100% !important; }
  #e2v-video-row { flex-direction: column; }
}
@media (max-width: 560px) {
  #e2v-header h1 { font-size: 26px; }
  #e2v-readiness { text-align: left; }
  .e2v-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .e2v-metric-label { min-height: 0; }
}
"""


try:
    ASSETS = get_asset_paths()
    ASSET_CONFIG_ERROR = None
except PipelineError as exc:
    ASSETS = None
    ASSET_CONFIG_ERROR = str(exc)
WORKER_PYTHON = os.environ.get("EEG2VIDEO_WORKER_PYTHON", sys.executable)
UPLOAD_LIMIT_BYTES = 64 * 1024 * 1024
RUN_LOCK = threading.Lock()
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _validate_cache_root() -> Path:
    configured = os.environ.get("EEG2VIDEO_CACHE_DIR")
    cache = (
        Path(configured).expanduser().resolve()
        if configured
        else RELEASE_ROOT / "webapp" / ".cache"
    )
    asset_roots = []
    configured_assets = os.environ.get("EEG2VIDEO_ASSETS_DIR")
    if configured_assets:
        asset_roots.append(Path(configured_assets).expanduser().resolve())
    if ASSETS is not None:
        asset_roots.extend((ASSETS.data_root.resolve(), ASSETS.model_root.resolve()))
    for root in set(asset_roots):
        if cache == root or cache.is_relative_to(root) or root.is_relative_to(cache):
            raise RuntimeError("EEG2VIDEO_CACHE_DIR must not overlap the private, read-only asset directory.")
    return cache


CACHE_ROOT = _validate_cache_root()
RUNS_ROOT = CACHE_ROOT / "runs"


def _cleanup_stale_runs(retention_days: int = 7) -> int:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    removed = 0
    for candidate in RUNS_ROOT.iterdir():
        if not candidate.is_dir() or not UUID_PATTERN.fullmatch(candidate.name):
            continue
        modified = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc)
        if modified >= cutoff:
            continue
        resolved = candidate.resolve()
        if resolved.parent != RUNS_ROOT.resolve():
            continue
        shutil.rmtree(resolved)
        removed += 1
    return removed


def _worker_probe() -> dict:
    if not WORKER_PYTHON:
        return {"available": False, "reason": "Set EEG2VIDEO_WORKER_PYTHON to the worker environment executable."}
    executable = Path(WORKER_PYTHON).expanduser()
    if not executable.is_file():
        return {"available": False, "reason": "Configured worker Python executable does not exist."}
    script = (
        "import importlib.util,json;"
        "mods=['numpy','torch','diffusers','xformers','imageio','imageio_ffmpeg','skimage'];"
        "print(json.dumps({m:importlib.util.find_spec(m) is not None for m in mods}))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", script],
            cwd=RELEASE_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": f"Worker dependency probe failed: {exc}"}
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": completed.stderr.strip()[-1000:] or "Worker dependency probe failed.",
        }
    try:
        modules = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"available": False, "reason": "Worker returned an invalid dependency probe result."}
    return {"available": True, "modules": modules}


def _preflight() -> dict:
    if ASSETS is None:
        reason = ASSET_CONFIG_ERROR or "Private asset configuration is unavailable."
        return {
            "precomputed_ready": False,
            "live_ready": False,
            "missing_precomputed": [reason],
            "missing_live": [reason],
            "worker_modules": {},
        }
    required_precomputed = {
        "canonical test EEG": ASSETS.test_eeg,
        "EEG sample mapping": ASSETS.sample_mapping,
        "paired video manifest": ASSETS.paired_manifest,
        "channel order file": ASSETS.channel_file,
        "completed generation metrics": ASSETS.metrics_csv,
        "final generation directory": ASSETS.generation_output,
    }
    required_live = {
        "canonical test EEG": ASSETS.test_eeg,
        "EEG sample mapping": ASSETS.sample_mapping,
        "paired video manifest": ASSETS.paired_manifest,
        "channel order file": ASSETS.channel_file,
        "visual predictor checkpoint": ASSETS.visual_output / "final_model.pt",
        "visual normalisation": ASSETS.visual_output / "eeg_normalisation.npz",
        "semantic predictor checkpoint": ASSETS.semantic_output / "final_model.pt",
        "semantic normalisation": ASSETS.semantic_output / "eeg_normalisation.npz",
        "dynamic predictor checkpoint": ASSETS.dynamic_output / "final_model.pt",
        "dynamic normalisation": ASSETS.dynamic_output / "eeg_normalisation.npz",
        "visual Block 7 predictions": ASSETS.visual_output / "block7_predicted_visual_latents.npy",
        "visual prediction mapping": ASSETS.visual_output / "prediction_mapping.csv",
        "semantic Block 7 predictions": ASSETS.semantic_output / "block7_predicted_text_embeddings.npy",
        "semantic prediction mapping": ASSETS.semantic_output / "prediction_mapping.csv",
        "dynamic Block 7 predictions": ASSETS.dynamic_output / "block7_predicted_dynamic_labels.npy",
        "dynamic prediction mapping": ASSETS.dynamic_output / "prediction_mapping.csv",
        "final diffusion UNet": ASSETS.diffusion_output / "unet" / "diffusion_pytorch_model.bin",
        "final DDIM scheduler": ASSETS.diffusion_output / "scheduler" / "scheduler_config.json",
        "completed diffusion provenance": ASSETS.diffusion_output / "provenance.json",
        "final negative embedding": ASSETS.negative_embedding,
    }
    missing_precomputed = [label for label, path in required_precomputed.items() if not path.exists()]
    missing_live = [label for label, path in required_live.items() if not path.is_file()]
    if ASSETS.sd14_model is None or not (
        (ASSETS.sd14_model / "model_index.json").is_file()
        and (ASSETS.sd14_model / "vae" / "config.json").is_file()
        and any((ASSETS.sd14_model / "vae").glob("diffusion_pytorch_model.*"))
    ):
        missing_live.append("pinned local Stable Diffusion VAE")
    probe = _worker_probe()
    if not probe.get("available"):
        missing_precomputed.append("worker Python environment")
        missing_live.append("worker Python environment")
        modules = {}
    else:
        modules = probe["modules"]
        for module in ("numpy", "imageio", "imageio_ffmpeg", "skimage"):
            if not modules.get(module):
                missing_precomputed.append(f"worker package {module}")
        for module in ("numpy", "torch", "diffusers", "xformers", "imageio", "imageio_ffmpeg", "skimage"):
            if not modules.get(module):
                missing_live.append(f"worker package {module}")
    return {
        "precomputed_ready": not missing_precomputed,
        "live_ready": not missing_live,
        "missing_precomputed": missing_precomputed,
        "missing_live": missing_live,
        "worker_modules": modules,
    }


def _preflight_markdown(info: dict) -> str:
    demo = "READY" if info["precomputed_ready"] else "NOT READY"
    live = "READY" if info["live_ready"] else "NOT READY"
    lines = [
        f"- **Precomputed demonstration:** {demo}",
        f"- **Live inference and generation:** {live}",
        f"- **Asset root:** `{ASSETS.data_root if ASSETS is not None else 'not configured'}`",
        f"- **Cache root:** `{CACHE_ROOT}` (ignored; stale UUID run folders older than 7 days are cleaned on startup)",
        f"- **Worker Python:** `{WORKER_PYTHON or 'not configured'}`",
    ]
    modules = info.get("worker_modules", {})
    if modules:
        module_status = ", ".join(
            f"{name}: {'available' if available else 'missing'}"
            for name, available in sorted(modules.items())
        )
        lines.append(f"- **Worker packages:** {module_status}")
    if info["missing_precomputed"]:
        lines.append("- **Precomputed requirements missing:** " + ", ".join(info["missing_precomputed"]))
    if info["missing_live"]:
        lines.append("- **Live requirements missing:** " + ", ".join(info["missing_live"]))
    lines.append(
        "\nFile presence is an initial readiness check only. The worker verifies artifact hashes, mappings, "
        "model provenance, CUDA and local-only model loading before use."
    )
    return "\n".join(lines)


def _status_text(
    stages: dict[str, dict],
    run_id: str,
    message: str | None = None,
    result: dict | None = None,
) -> str:
    del stages, run_id
    status_class = "e2v-status e2v-status-error" if message and message.startswith("Error:") else "e2v-status"
    parts = [f'<div class="{status_class}">']
    if message:
        parts.append(html.escape(message).replace("\n", "<br>"))
    if result:
        test_index = result.get("test_index")
        sample_id = result.get("sample_id")
        identity = []
        if test_index is not None:
            identity.append(f"Sample {html.escape(str(test_index))}")
        if sample_id:
            identity.append(f"Clip {html.escape(str(sample_id))}")
        if identity:
            parts.append(f'<span class="e2v-identity">{" · ".join(identity)}</span>')
    if not message and not result:
        parts.append("Waiting for a reconstruction request.")
    parts.append("</div>")
    return "".join(parts)


def _json_outputs(result: dict | None) -> dict | None:
    if not result:
        return None
    payload = {
        "mode": result.get("mode"),
        "sample_id": result.get("sample_id"),
        "test_index": result.get("test_index"),
        "seed": result.get("seed"),
        "metrics": result.get("metrics"),
    }
    if result.get("prediction"):
        payload["prediction"] = result["prediction"]
    if result.get("generation"):
        payload["generation"] = result["generation"]
    if result.get("comparison_to_existing_generated_video") is not None:
        payload["comparison_to_existing_generated_video"] = result["comparison_to_existing_generated_video"]
    return payload


def _metric_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "—"
        return f"{value:.4g}"
    return html.escape(str(value))


def _metrics_html(result: dict | None) -> str:
    metrics = result.get("metrics") or {} if result else {}
    cards = (
        ("RGB MSE", "rgb_mse_01", "[0, 1]"),
        ("PSNR", "psnr_db", "dB"),
        ("RGB SSIM", "ssim_rgb_7x7", ""),
        ("Temporal-difference MAE", "temporal_difference_mae_255", "[0, 255]"),
    )
    rendered = "".join(
        '<div class="e2v-metric-card">'
        f'<span class="e2v-metric-label">{label}</span>'
        f'<strong class="e2v-metric-value">{_metric_value(metrics.get(key))}</strong>'
        f'<span class="e2v-metric-unit">{unit or "&nbsp;"}</span>'
        "</div>"
        for label, key, unit in cards
    )
    return f'<div class="e2v-metrics">{rendered}</div>'


def _diagnostics_html(stages: dict, result: dict | None, run_id: str) -> str:
    stage_data = html.escape(json.dumps(stages, indent=2, ensure_ascii=False))
    result_data = html.escape(
        json.dumps(
            {"run_id": run_id, "result": _json_outputs(result)},
            indent=2,
            ensure_ascii=False,
        )
    )
    return (
        '<details><summary>Detailed Progress · Full pipeline stages</summary>'
        f'<pre class="e2v-diagnostics">{stage_data}</pre></details>'
        '<details><summary>Raw evaluation and run metadata</summary>'
        f'<pre class="e2v-diagnostics">{result_data}</pre></details>'
    )


def run_job(mode: str, uploaded_file, selected_index, sampling_rate_hz, channels_confirmed, unknown_seed):
    if not RUN_LOCK.acquire(blocking=False):
        yield (
            None,
            None,
            None,
            _metrics_html(None),
            _status_text({}, "", "A reconstruction job is already active. Wait for it to finish."),
            _diagnostics_html({}, None, ""),
            "",
        )
        return
    run_id = str(uuid.uuid4())
    run_dir = RUNS_ROOT / run_id
    stages = {name: {"status": "not_started", "detail": ""} for name in STAGES}
    view = PipelineProgress(mode)
    result = None
    error_message = None
    process = None
    try:
        if not uploaded_file:
            raise ValueError("Upload one .npy EEG file before starting.")
        upload_path = Path(str(uploaded_file)).expanduser()
        if upload_path.suffix.lower() != ".npy":
            raise ValueError("Upload must be a NumPy .npy file.")
        if not upload_path.is_file():
            raise ValueError("The uploaded EEG file is unavailable. Upload it again.")
        if upload_path.stat().st_size > UPLOAD_LIMIT_BYTES:
            raise ValueError("The upload exceeds the 64 MiB limit.")
        if sampling_rate_hz is None or float(sampling_rate_hz) != 200.0:
            raise ValueError("Sampling rate must be exactly 200 Hz; no resampling is applied.")
        if not channels_confirmed:
            raise ValueError("Confirm that the input uses the verified project channel order.")
        try:
            selected_index_value = float(selected_index)
        except (TypeError, ValueError):
            raise ValueError("Test index must be an integer from 0 to 199.") from None
        if not selected_index_value.is_integer() or not 0 <= selected_index_value < 200:
            raise ValueError("Test index must be an integer from 0 to 199.")
        try:
            unknown_seed_value = float(unknown_seed)
        except (TypeError, ValueError):
            raise ValueError("Unknown-sample seed must be an integer from 0 to 2,147,483,647.") from None
        if not unknown_seed_value.is_integer() or not 0 <= unknown_seed_value <= 2**31 - 1:
            raise ValueError("Unknown-sample seed must be an integer from 0 to 2,147,483,647.")
        preflight = _preflight()
        if mode == "precomputed" and not preflight["precomputed_ready"]:
            raise ValueError("Precomputed mode is unavailable: " + ", ".join(preflight["missing_precomputed"]))
        if mode == "live" and not preflight["live_ready"]:
            raise ValueError("Live mode is unavailable: " + ", ".join(preflight["missing_live"]))

        executable = Path(WORKER_PYTHON).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        cached_upload = run_dir / "input.npy"
        shutil.copyfile(upload_path, cached_upload)
        command = [
            str(executable),
            "-m",
            "webapp.worker",
            "--mode",
            mode,
            "--upload",
            str(cached_upload.resolve()),
            "--index",
            str(int(selected_index_value)),
            "--sampling-rate",
            str(float(sampling_rate_hz)),
            "--unknown-seed",
            str(int(unknown_seed_value)),
            "--run-dir",
            str(run_dir.resolve()),
        ]
        if channels_confirmed:
            command.append("--channels-confirmed")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(RELEASE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        stderr_path = run_dir / "worker.stderr.log"
        yield (
            None,
            None,
            None,
            _metrics_html(None),
            _status_text(stages, run_id, "Starting the local worker…"),
            _diagnostics_html(stages, None, run_id),
            view.render(),
        )
        with stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                cwd=RELEASE_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if process.stdout is None:
                raise RuntimeError("Worker stdout could not be opened.")
            # A reader thread keeps the generator responsive so elapsed time refreshes between worker events.
            lines: queue.Queue = queue.Queue()

            def read_worker_output(stream=process.stdout) -> None:
                for raw in stream:
                    lines.put(raw)
                lines.put(None)

            threading.Thread(target=read_worker_output, daemon=True).start()
            while True:
                try:
                    line = lines.get(timeout=PROGRESS_REFRESH_SECONDS)
                except queue.Empty:
                    yield (
                        None,
                        None,
                        None,
                        _metrics_html(result),
                        _status_text(
                            stages,
                            run_id,
                            f"Error: {error_message}" if error_message else None,
                            result,
                        ),
                        _diagnostics_html(stages, result, run_id),
                        view.render(),
                    )
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "stage":
                    stages[event["stage"]] = {"status": event["status"], "detail": event["detail"]}
                    view.apply(event)
                elif event.get("type") == "progress":
                    view.apply(event)
                elif event.get("type") == "result":
                    result = event.get("result")
                elif event.get("type") == "error":
                    error_message = event.get("message", "The worker failed.")
                yield (
                    result.get("ground_truth_video") if result else None,
                    result.get("generated_video") if result else None,
                    result.get("download_video") if result else None,
                    _metrics_html(result),
                    _status_text(
                        stages,
                        run_id,
                        f"Error: {error_message}" if error_message else None,
                        result,
                    ),
                    _diagnostics_html(stages, result, run_id),
                    view.render(),
                )
            return_code = process.wait()
        if return_code != 0 or result is None:
            log_tail = ""
            if stderr_path.is_file():
                log_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-1200:].strip()
            message = error_message or f"Worker exited with code {return_code} without a result."
            view.fail(message)
            stages.update({name: state for name, state in view.snapshot().items() if name in stages})
            if log_tail:
                message += "\n\nWorker diagnostic tail:\n```\n" + log_tail + "\n```"
            yield (
                None,
                None,
                None,
                _metrics_html(None),
                _status_text(stages, run_id, f"Error: {message}"),
                _diagnostics_html(stages, result, run_id),
                view.render(),
            )
            return
        view.succeed()
        completion_message = (
            "Existing reconstruction loaded — no model inference performed."
            if mode == "precomputed"
            else "Live generation completed."
        )
        yield (
            result.get("ground_truth_video"),
            result.get("generated_video"),
            result.get("download_video"),
            _metrics_html(result),
            _status_text(stages, run_id, completion_message, result),
            _diagnostics_html(stages, result, run_id),
            view.render(),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        view.fail(str(exc))
        yield (
            None,
            None,
            None,
            _metrics_html(None),
            _status_text(stages, run_id, f"Error: {exc}"),
            _diagnostics_html(stages, None, run_id),
            view.render(),
        )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
        RUN_LOCK.release()


def run_precomputed(uploaded_file, selected_index, sampling_rate_hz, channels_confirmed, unknown_seed):
    yield from run_job("precomputed", uploaded_file, selected_index, sampling_rate_hz, channels_confirmed, unknown_seed)


def run_live(uploaded_file, selected_index, sampling_rate_hz, channels_confirmed, unknown_seed):
    yield from run_job("live", uploaded_file, selected_index, sampling_rate_hz, channels_confirmed, unknown_seed)


def build_app():
    preflight = _preflight()
    _cleanup_stale_runs()
    if preflight["precomputed_ready"] and preflight["live_ready"]:
        readiness_label, readiness_class = "● System Ready", "e2v-ready"
        readiness_warning = ""
    elif preflight["precomputed_ready"] or preflight["live_ready"]:
        readiness_label, readiness_class = "● Limited Availability", "e2v-limited"
        unavailable = []
        if not preflight["precomputed_ready"]:
            unavailable.append("existing-result retrieval")
        if not preflight["live_ready"]:
            unavailable.append("live generation")
        readiness_warning = (
            f"Unavailable: {', '.join(unavailable)}. "
            "Open Advanced Details / Diagnostics for preflight checks."
        )
    else:
        readiness_label, readiness_class = "● System Unavailable", "e2v-unavailable"
        readiness_warning = (
            "Required assets or worker dependencies are missing; both actions are disabled. "
            "Open Advanced Details / Diagnostics for preflight checks."
        )
    theme = gr.themes.Base(primary_hue="orange", secondary_hue="teal", neutral_hue="slate")
    with gr.Blocks(
        title="EEG2Video — Local Demonstration",
        theme=theme,
        css=APP_CSS,
        elem_id="e2v-app",
    ) as app:
        with gr.Row(elem_id="e2v-header"):
            with gr.Column(scale=8):
                gr.Markdown(
                    "# EEG2Video\n"
                    "Reconstructing Video from Brain Signals"
                )
            with gr.Column(scale=2, elem_id="e2v-readiness"):
                gr.HTML(f'<span class="e2v-ready-badge {readiness_class}">{readiness_label}</span>')
        if readiness_warning:
            gr.Markdown(readiness_warning, elem_id="e2v-readiness-warning")

        with gr.Row(elem_id="e2v-dashboard"):
            with gr.Column(scale=3, elem_id="e2v-inputs"):
                gr.Markdown("### EEG Input")
                upload = gr.File(
                    label="Upload EEG",
                    file_types=[".npy"],
                    type="filepath",
                    height=110,
                )
                gr.Markdown("62 channels · 400 samples · 200 Hz", elem_classes="e2v-input-note")
                test_index = gr.Slider(
                    minimum=0,
                    maximum=199,
                    value=0,
                    step=1,
                    label="Sample index",
                    info="Used for full test arrays; exact signal matching identifies single-sample files.",
                )
                channels_confirmed = gr.Checkbox(
                    label="Confirm verified 62-channel order",
                    value=False,
                )
                sampling_rate = gr.Number(
                    value=200,
                    precision=0,
                    visible=False,
                    elem_id="e2v-sampling-rate",
                )
                with gr.Accordion("Advanced Settings", open=False, elem_id="e2v-seed-settings"):
                    unknown_seed = gr.Number(
                        label="Seed for an unmatched live sample",
                        value=42,
                        precision=0,
                        info="Known Block 7 samples retain the established seed: 42 + test index.",
                    )
                precomputed_button = gr.Button(
                    "View Existing Result",
                    variant="secondary",
                    interactive=preflight["precomputed_ready"],
                )
                live_button = gr.Button(
                    "Generate Live Video",
                    variant="primary",
                    interactive=preflight["live_ready"],
                )
            with gr.Column(scale=7, elem_id="e2v-results"):
                with gr.Row(elem_id="e2v-video-row"):
                    ground_truth = gr.Video(
                        label="Ground Truth",
                        autoplay=False,
                        interactive=False,
                        elem_id="e2v-ground-truth",
                    )
                    generated = gr.Video(
                        label="EEG-Generated Video",
                        autoplay=False,
                        interactive=False,
                        elem_id="e2v-generated-video",
                    )
                download = gr.File(
                    label="Download generated MP4",
                    interactive=False,
                    height=58,
                    elem_id="e2v-download",
                )
                gr.Markdown("### Evaluation Metrics")
                metrics = gr.HTML(_metrics_html(None), label="Evaluation metrics")
                status = gr.HTML(
                    _status_text({}, ""),
                    label="Reconstruction status",
                )
                progress_view = gr.HTML(
                    '<div class="e2v-progress-empty">Processing progress will appear here.</div>',
                    label="Processing progress",
                    elem_id="e2v-progress",
                )
        with gr.Accordion("Advanced Details / Diagnostics", open=False, elem_id="e2v-diagnostics"):
            gr.Markdown(_preflight_markdown(preflight))
            gr.Markdown(
                "**Scientific scope:** Precomputed mode retrieves an evaluated result and performs no model "
                "inference. Live mode runs the three EEG predictors and a new 100-step diffusion generation. "
                "A reference video is shown only when the sample is correctly paired; unmatched samples have "
                "no reference video or historical metrics. RGB MSE is on [0, 1], PSNR in dB, RGB SSIM uses "
                "the recorded 7×7 protocol, and temporal-difference MAE is on [0, 255]. The temporal metric "
                "is descriptive, not motion accuracy; single-subject Block 7 results are not a blind test or "
                "population-level evidence."
            )
            stage_status = gr.HTML(
                _diagnostics_html({}, None, ""),
                label="Pipeline and evaluation diagnostics",
            )
        inputs = [upload, test_index, sampling_rate, channels_confirmed, unknown_seed]
        outputs = [ground_truth, generated, download, metrics, status, stage_status, progress_view]
        precomputed_button.click(
            run_precomputed,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            concurrency_id="eeg2video-reconstruction",
        )
        live_button.click(
            run_live,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
            concurrency_id="eeg2video-reconstruction",
        )
    return app


def main() -> None:
    app = build_app()
    port = int(os.environ.get("EEG2VIDEO_PORT", "7860"))
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=False,
        allowed_paths=[str(CACHE_ROOT)],
        max_file_size="64mb",
    )


if __name__ == "__main__":
    main()
