from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

from webapp.core import PipelineError, emit_stage, get_asset_paths, run_precomputed


def _validate_run_directory(path: Path) -> Path:
    paths = get_asset_paths()
    run_root = (paths.cache_root / "runs").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise PipelineError("Run output must be isolated under the configured application cache.") from exc
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", resolved.name):
        raise PipelineError("Run directory must use a UUID run identifier.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _emit(value: dict) -> None:
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local EEG2Video Gradio worker.")
    parser.add_argument("--mode", choices=("precomputed", "live"), required=True)
    parser.add_argument("--upload", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--channels-confirmed", action="store_true")
    parser.add_argument("--unknown-seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = _validate_run_directory(args.run_dir)
    active_stage = {"name": None, "status": None}

    def progress(event: dict) -> None:
        if event.get("type") == "stage":
            active_stage["name"] = event.get("stage")
            active_stage["status"] = event.get("status")
        _emit(event)

    try:
        if args.mode == "precomputed":
            result = run_precomputed(
                args.upload,
                args.index,
                args.sampling_rate,
                args.channels_confirmed,
                run_dir,
                progress,
            )
        else:
            from webapp.inference import run_live

            result = run_live(
                args.upload,
                args.index,
                args.sampling_rate,
                args.channels_confirmed,
                args.unknown_seed,
                run_dir,
                progress,
            )
        _emit({"type": "result", "result": result})
        return 0
    except Exception as exc:
        if active_stage["name"] and active_stage["status"] == "running":
            emit_stage(progress, active_stage["name"], "failed", str(exc))
        error_file = run_dir / "error.log"
        error_file.write_text(traceback.format_exc(), encoding="utf-8")
        _emit(
            {
                "type": "error",
                "message": str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}",
                "log": str(error_file),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
