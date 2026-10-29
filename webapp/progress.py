"""Structured progress events, timing and HTML rendering for the live pipeline.

This module has no heavy dependencies so it can be imported by both the CUDA worker
and the Gradio UI process.
"""

from __future__ import annotations

import html
import time
from collections import deque
from typing import Callable

# Stages in their actual execution order, with their display labels.
DISPLAY_STAGES: tuple[tuple[str, str], ...] = (
    ("EEG validation", "EEG Validation"),
    ("Sample identification", "Sample Identification"),
    ("Visual prediction", "Visual Prediction"),
    ("Semantic prediction", "Semantic Prediction"),
    ("Dynamic prediction", "Dynamic Prediction"),
    ("Diffusion preparation", "Diffusion Model Preparation"),
    ("Diffusion generation", "Video Diffusion Generation"),
    ("Video decoding", "Video Decoding"),
    ("Evaluation", "Evaluation and Finalisation"),
)
STAGE_NAMES = tuple(name for name, _ in DISPLAY_STAGES)
STAGE_LABELS = dict(DISPLAY_STAGES)

_FINISHED = ("complete", "failed", "not_run", "not_available")


def emit_progress(
    callback: Callable[[dict], None] | None,
    stage: str,
    completed: int,
    total: int,
    message: str,
    unit: str = "step",
    elapsed_seconds: float | None = None,
    eta_seconds: float | None = None,
    average_step_seconds: float | None = None,
) -> None:
    if callback:
        callback(
            {
                "type": "progress",
                "stage": stage,
                "status": "running",
                "completed": completed,
                "total": total,
                "unit": unit,
                "percent": percent(completed, total),
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "average_step_seconds": average_step_seconds,
                "message": message,
            }
        )


def percent(completed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * min(max(completed, 0), total) / total, 1)


class StepTimer:
    """Measures completed steps with a monotonic clock and estimates remaining time.

    The estimate is the mean of the most recent `window` step durations multiplied by
    the number of remaining steps. The first step is excluded as warm-up, and no estimate
    is produced until `min_samples` durations have been recorded.
    """

    def __init__(self, total: int, window: int = 10, min_samples: int = 3, clock=time.monotonic):
        self.total = total
        self.min_samples = min_samples
        self._clock = clock
        self._durations: deque[float] = deque(maxlen=window)
        self.completed = 0
        self._started = self._last = clock()

    def complete_step(self) -> dict:
        now = self._clock()
        if self.completed > 0:
            self._durations.append(now - self._last)
        self._last = now
        self.completed += 1
        average = None
        eta = None
        if len(self._durations) >= self.min_samples:
            average = sum(self._durations) / len(self._durations)
            eta = average * (self.total - self.completed)
        elif self.completed >= self.total:
            eta = 0.0
        return {
            "completed": self.completed,
            "total": self.total,
            "elapsed_seconds": now - self._started,
            "eta_seconds": eta,
            "average_step_seconds": average,
        }


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class PipelineProgress:
    """UI-side state built from worker events; rendered to HTML on demand."""

    def __init__(self, mode: str, clock=time.monotonic):
        self.mode = mode
        self._clock = clock
        self.started = clock()
        self.finished_at: float | None = None
        self.outcome: str | None = None  # None while running, then "success" or "failed"
        self.error: str | None = None
        self.stages = {name: {"status": "not_started", "detail": "", "start": None, "end": None} for name in STAGE_NAMES}
        self.progress: dict[str, dict] = {}
        self.progress_received: dict[str, float] = {}

    def apply(self, event: dict) -> None:
        now = self._clock()
        stage = event.get("stage")
        if stage not in self.stages:
            if event.get("type") == "stage" and stage:
                self.stages[stage] = {"status": "not_started", "detail": "", "start": None, "end": None}
            else:
                return
        state = self.stages[stage]
        if event.get("type") == "stage":
            status = event.get("status", "not_started")
            if status == "running" and state["start"] is None:
                state["start"] = now
            if status in _FINISHED:
                if state["start"] is None:
                    state["start"] = now
                state["end"] = now
            state["status"] = status
            state["detail"] = event.get("detail", "")
        elif event.get("type") == "progress":
            if state["status"] != "running":
                state["status"] = "running"
                state["start"] = state["start"] or now
            state["detail"] = event.get("message", state["detail"])
            self.progress[stage] = event
            self.progress_received[stage] = now

    def fail(self, message: str | None) -> None:
        now = self._clock()
        for state in self.stages.values():
            if state["status"] == "running":
                state["status"] = "failed"
                state["end"] = now
        self.outcome = "failed"
        self.error = message
        self.finished_at = now

    def succeed(self) -> None:
        self.outcome = "success"
        self.finished_at = self._clock()

    def active_stage(self) -> str | None:
        for name in STAGE_NAMES:
            if self.stages[name]["status"] == "running":
                return name
        return None

    def completed_count(self) -> tuple[int, int]:
        counted = [s for s in self.stages.values() if s["status"] not in ("not_run", "not_available")]
        return sum(1 for s in counted if s["status"] == "complete"), len(counted)

    def snapshot(self) -> dict:
        return {
            name: {"status": state["status"], "detail": state["detail"]}
            for name, state in self.stages.items()
        }

    def render(self) -> str:
        now = self._clock()
        end = self.finished_at if self.finished_at is not None else now
        elapsed = format_duration(end - self.started)
        if self.mode == "precomputed":
            if self.outcome == "success":
                headline = "✓ Existing reconstruction loaded — no model inference performed"
            elif self.outcome == "failed":
                headline = "Existing reconstruction unavailable"
            else:
                headline = "Retrieving existing reconstruction…"
            parts = [_STYLE, '<div class="e2v-progress">', f"<b>{headline}</b>"]
            if self.outcome == "failed" and self.error:
                parts.append(f'<div class="e2v-error">{html.escape(self.error)}</div>')
            if self.outcome is None:
                parts.append(f'<span class="e2v-elapsed">Elapsed {elapsed}</span>')
            parts.append("</div>")
            return "".join(parts)

        if self.outcome == "success":
            headline, colour = "✓ Generation complete", "#55c99a"
        elif self.outcome == "failed":
            headline, colour = "✖ Generation failed", "#ef7777"
        else:
            headline, colour = "● Processing", "#f28a36"
        parts = [
            _STYLE,
            '<div class="e2v-progress">',
            f'<div class="e2v-progress-head"><b style="color:{colour}">{headline}</b>'
            f'<span class="e2v-elapsed">Elapsed {elapsed}</span></div>',
        ]
        if self.error and self.outcome == "failed":
            parts.append(f'<div class="e2v-error">{html.escape(self.error)}</div>')
        parts.append(self._detail_panel(now, end))
        parts.append("</div>")
        return "".join(parts)

    def _detail_panel(self, now: float, end: float) -> str:
        stage = self.active_stage()
        if stage is None:
            measured = [name for name in STAGE_NAMES if name in self.progress]
            stage = max(measured, key=self.progress_received.__getitem__) if measured else None
            if stage is None and self.outcome == "failed":
                measured = [n for n in STAGE_NAMES if n in self.progress]
                stage = max(measured, key=self.progress_received.__getitem__) if measured else None
        if stage is None:
            if self.outcome == "success":
                return '<div class="e2v-detail">All processing stages completed.</div>'
            if self.outcome == "failed":
                return ""
            return (
                '<div class="e2v-bar e2v-indeterminate"></div>'
                '<div class="e2v-detail">Preparing EEG validation and sample identification…</div>'
            )
        state = self.stages[stage]
        title = "Video diffusion" if stage == "Diffusion generation" else STAGE_LABELS[stage]
        title = html.escape(title)
        detail = html.escape(state["detail"])
        event = self.progress.get(stage)
        if event is None:
            bar = self._bar(None) if state["status"] == "running" else ""
            return (
                f'{bar}<div class="e2v-detail"><b>{title}</b>'
                f'<span class="e2v-note">{detail or "In progress"} · Elapsed {format_duration(end - self.started)}</span>'
                "</div>"
            )
        completed, total = event["completed"], event["total"]
        unit = html.escape(str(event.get("unit", "step"))).capitalize()
        percent_complete = event.get("percent")
        if percent_complete is None:
            percent_complete = 100.0 * completed / total if total else 0.0
        lines = [f"{completed}/{total} {unit.lower()}s · {percent_complete:.0f}%"]
        if state["status"] == "running":
            eta = event.get("eta_seconds")
            lines.append(
                f"ETA ~{format_duration(eta)}" if eta is not None else "Estimating ETA…"
            )
        return (
            f'{self._bar(percent_complete)}<div class="e2v-detail"><b>{title}</b>'
            f'<span class="e2v-detail-line">{" · ".join(lines)} · Elapsed {format_duration(end - self.started)}</span>'
            f'<span class="e2v-note">{detail}</span></div>'
        )

    @staticmethod
    def _bar(pct: float | None, extra: str = "") -> str:
        if pct is None:
            return f'<div class="e2v-bar e2v-indeterminate {extra}"><div></div></div>'
        return f'<div class="e2v-bar {extra}"><div style="width:{pct:.1f}%"></div></div>'


_STYLE = (
    "<style>"
    ".e2v-progress{font-size:13px;color:#f0f2f3}"
    ".e2v-progress-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}"
    ".e2v-elapsed,.e2v-note{color:#a7afb5;font-size:12px}"
    ".e2v-error{background:#3a2022;color:#ffd9d9;padding:7px 9px;border-radius:6px;margin:6px 0;overflow-wrap:anywhere}"
    ".e2v-bar{height:5px;border-radius:4px;background:#383d42;overflow:hidden;margin:6px 0}"
    ".e2v-bar>div{height:100%;background:#f28a36;transition:width .3s}"
    ".e2v-indeterminate>div{width:35%;animation:e2v-slide 1.2s ease-in-out infinite}"
    "@keyframes e2v-slide{0%{margin-left:-35%}100%{margin-left:100%}}"
    ".e2v-detail{display:flex;gap:8px;flex-wrap:wrap;align-items:baseline}"
    ".e2v-detail-line{color:#d5d9dc}"
    ".e2v-note{overflow-wrap:anywhere}"
    "</style>"
)
