import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from webapp.progress import PipelineProgress, StepTimer, emit_progress, format_duration, percent


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ProgressMathTests(unittest.TestCase):
    def test_percent(self):
        self.assertEqual(percent(1, 100), 1.0)
        self.assertEqual(percent(25, 100), 25.0)
        self.assertEqual(percent(100, 100), 100.0)
        self.assertEqual(percent(150, 100), 100.0)
        self.assertEqual(percent(0, 0), 0.0)

    def test_event_format_is_strict_json(self):
        events = []
        emit_progress(events.append, "Diffusion generation", 50, 100, "half", eta_seconds=12.5)
        event = events[0]
        self.assertEqual(event["type"], "progress")
        self.assertEqual(event["percent"], 50.0)
        self.assertEqual(event["total"], 100)
        json.dumps(event, allow_nan=False)

    def test_eta_needs_samples_then_tracks_average(self):
        clock = FakeClock()
        timer = StepTimer(100, window=10, min_samples=3, clock=clock)
        clock.now += 5.0  # first step includes warm-up and is excluded from the average
        first = timer.complete_step()
        self.assertIsNone(first["eta_seconds"])
        for _ in range(2):
            clock.now += 2.0
            snapshot = timer.complete_step()
        self.assertIsNone(snapshot["eta_seconds"])  # only two durations recorded
        clock.now += 2.0
        snapshot = timer.complete_step()
        self.assertEqual(snapshot["completed"], 4)
        self.assertAlmostEqual(snapshot["average_step_seconds"], 2.0)
        self.assertAlmostEqual(snapshot["eta_seconds"], 2.0 * 96)
        clock.now += 4.0  # a slower step moves the moving average and the estimate
        snapshot = timer.complete_step()
        self.assertAlmostEqual(snapshot["average_step_seconds"], 2.5)
        self.assertAlmostEqual(snapshot["eta_seconds"], 2.5 * 95)

    def test_eta_is_zero_on_final_step_and_window_is_bounded(self):
        clock = FakeClock()
        timer = StepTimer(20, window=5, min_samples=3, clock=clock)
        for index in range(20):
            clock.now += 10.0 if index < 10 else 1.0
            snapshot = timer.complete_step()
        self.assertEqual(snapshot["completed"], 20)
        self.assertEqual(snapshot["eta_seconds"], 0.0)
        self.assertAlmostEqual(snapshot["average_step_seconds"], 1.0)

    def test_format_duration(self):
        self.assertEqual(format_duration(None), "—")
        self.assertEqual(format_duration(59), "59s")
        self.assertEqual(format_duration(125), "2m 05s")
        self.assertEqual(format_duration(3725), "1h 02m 05s")


class PipelineViewTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.view = PipelineProgress("live", clock=self.clock)

    def stage(self, name, status, detail=""):
        self.view.apply({"type": "stage", "stage": name, "status": status, "detail": detail})

    def test_stage_transitions_and_counts(self):
        self.stage("EEG validation", "running")
        self.assertEqual(self.view.active_stage(), "EEG validation")
        self.clock.now += 3
        self.stage("EEG validation", "complete")
        self.stage("Sample identification", "complete")
        self.stage("Visual prediction", "running")
        done, total = self.view.completed_count()
        self.assertEqual((done, total), (2, 9))
        self.assertEqual(self.view.stages["Semantic prediction"]["status"], "not_started")
        page = self.view.render()
        self.assertIn("Visual Prediction", page)
        self.assertNotIn("Semantic Prediction", page)
        self.assertNotIn("Pending", page)

    def test_diffusion_progress_and_estimating_message(self):
        self.stage("Video diffusion", "running")  # unknown names are tracked but not counted as pipeline stages
        self.stage("Diffusion generation", "running", "start")
        self.view.apply({"type": "progress", "stage": "Diffusion generation", "completed": 2, "total": 100,
                         "unit": "step", "percent": 2.0, "eta_seconds": None, "average_step_seconds": None,
                         "message": "step 2"})
        page = self.view.render()
        self.assertIn("2/100 steps", page)
        self.assertIn("Estimating ETA", page)
        self.view.apply({"type": "progress", "stage": "Diffusion generation", "completed": 50, "total": 100,
                         "unit": "step", "percent": 50.0, "eta_seconds": 100.0, "average_step_seconds": 2.0,
                         "message": "step 50"})
        page = self.view.render()
        self.assertIn("50/100 steps", page)
        self.assertIn("width:50.0%", page)
        self.assertIn("ETA ~1m 40s", page)

    def test_unmeasured_stage_is_indeterminate(self):
        self.stage("Semantic prediction", "running", "loading")
        page = self.view.render()
        self.assertIn("e2v-indeterminate", page)
        self.assertIn("Semantic Prediction", page)
        self.assertIn("loading", page)
        self.assertNotIn("Estimating remaining", page)

    def test_success_keeps_full_diffusion_progress(self):
        self.stage("Diffusion generation", "running")
        self.view.apply({"type": "progress", "stage": "Diffusion generation", "completed": 100, "total": 100,
                         "unit": "step", "percent": 100.0, "eta_seconds": 0.0, "average_step_seconds": 1.0,
                         "message": "done"})
        self.stage("Diffusion generation", "complete")
        self.view.succeed()
        page = self.view.render()
        self.assertIn("Generation complete", page)
        self.assertIn("100/100 steps", page)
        self.assertNotIn("Estimated remaining", page)

    def test_failure_marks_active_stage_and_stops_eta(self):
        self.stage("EEG validation", "complete")
        self.stage("Diffusion generation", "running")
        self.view.apply({"type": "progress", "stage": "Diffusion generation", "completed": 40, "total": 100,
                         "unit": "step", "percent": 40.0, "eta_seconds": 60.0, "average_step_seconds": 1.0,
                         "message": "step 40"})
        self.view.fail("CUDA ran out of memory")
        self.assertEqual(self.view.stages["Diffusion generation"]["status"], "failed")
        self.assertEqual(self.view.stages["EEG validation"]["status"], "complete")
        page = self.view.render()
        self.assertIn("failed", page.lower())
        self.assertIn("CUDA ran out of memory", page)
        self.assertIn("40/100 steps", page)
        self.assertNotIn("Estimated remaining", page)
        self.assertNotIn("Estimating remaining", page)

    def test_precomputed_has_no_diffusion_progress(self):
        view = PipelineProgress("precomputed", clock=self.clock)
        view.apply({"type": "stage", "stage": "Diffusion generation", "status": "not_run", "detail": "n/a"})
        page = view.render()
        self.assertIn("Retrieving existing reconstruction", page)
        self.assertNotIn('class="e2v-bar', page)
        self.assertNotIn("DDIM", page)

    def test_html_escapes_details(self):
        self.stage("Visual prediction", "running", "<script>alert(1)</script>")
        self.assertNotIn("<script>", self.view.render())


class RunJobCallbackTests(unittest.TestCase):
    """Drives the real generator callbacks with a fake worker subprocess."""

    class FakeProcess:
        def __init__(self, lines, delay_before=None):
            self.stdout = self._iterate(lines, delay_before)
            self.returncode = 0

        @staticmethod
        def _iterate(lines, delay_before):
            for index, line in enumerate(lines):
                if delay_before == index:
                    time.sleep(0.25)
                yield line + "\n"

        def wait(self):
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

    def setUp(self):
        try:
            from webapp import app
        except ImportError as exc:
            self.skipTest(f"Gradio unavailable: {exc}")
        self.app = app
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.upload = Path(self.tmp.name) / "eeg.npy"
        np.save(self.upload, np.zeros((62, 400), dtype=np.float32))
        patches = [
            mock.patch.object(app, "_preflight", return_value={
                "precomputed_ready": True, "live_ready": True, "missing_precomputed": [], "missing_live": []}),
            mock.patch.object(app, "RUNS_ROOT", Path(self.tmp.name) / "runs"),
            mock.patch.object(app, "PROGRESS_REFRESH_SECONDS", 0.05),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    @staticmethod
    def events(live_ok=True):
        stage = lambda name, status, detail="": json.dumps(
            {"type": "stage", "stage": name, "status": status, "detail": detail})
        progress = lambda done: json.dumps(
            {"type": "progress", "stage": "Diffusion generation", "status": "running", "completed": done,
             "total": 100, "unit": "step", "percent": float(done), "elapsed_seconds": done * 1.0,
             "eta_seconds": None if done < 4 else (100 - done) * 1.0, "average_step_seconds": None if done < 4 else 1.0,
             "message": f"step {done}"})
        lines = [stage("EEG validation", "running"), stage("EEG validation", "complete"),
                 stage("Diffusion generation", "running")]
        lines += [progress(n) for n in (1, 2, 3, 4, 50, 100)]
        if live_ok:
            lines += [stage("Diffusion generation", "complete"),
                      json.dumps({"type": "result", "result": {
                          "ground_truth_video": "gt.mp4", "generated_video": "gen.mp4",
                          "download_video": "gen.mp4", "metrics": {"psnr_db": 1.0},
                          "sample_id": "clip008", "test_index": 7, "message": "ok"}})]
        else:
            lines += [json.dumps({"type": "stage", "stage": "Diffusion generation", "status": "failed",
                                  "detail": "boom"}),
                      json.dumps({"type": "error", "message": "boom"})]
        return lines

    def drive(self, mode, lines, returncode=0, delay_before=None):
        process = self.FakeProcess(lines, delay_before)
        process.returncode = returncode
        with mock.patch.object(self.app.subprocess, "Popen", return_value=process):
            callback = self.app.run_live if mode == "live" else self.app.run_precomputed
            return list(callback(str(self.upload), 0, 200, True, 42))

    def test_callbacks_are_generator_functions(self):
        self.assertTrue(inspect.isgeneratorfunction(self.app.run_live))
        self.assertTrue(inspect.isgeneratorfunction(self.app.run_precomputed))

    def test_metric_cards_keep_zero_distinct_from_missing_values(self):
        rendered = self.app._metrics_html({
            "metrics": {"rgb_mse_01": 0.0, "psnr_db": None, "ssim_rgb_7x7": 0.0}
        })
        self.assertIn("<strong class=\"e2v-metric-value\">0</strong>", rendered)
        self.assertEqual(rendered.count("<strong class=\"e2v-metric-value\">—</strong>"), 2)

    def test_registered_callbacks_keep_seven_outputs_and_diagnostics_start_collapsed(self):
        with mock.patch.object(self.app, "_preflight", return_value={
            "precomputed_ready": True, "live_ready": True, "missing_precomputed": [], "missing_live": [],
            "worker_modules": {},
        }), mock.patch.object(self.app, "_cleanup_stale_runs"):
            demo = self.app.build_app()
        dependencies = demo.config["dependencies"]
        self.assertEqual(len(dependencies), 2)
        self.assertTrue(all(len(dependency["outputs"]) == 7 for dependency in dependencies))
        diagnostics = next(
            item for item in demo.config["components"]
            if item["props"].get("elem_id") == "e2v-diagnostics"
        )
        self.assertFalse(diagnostics["props"]["open"])

    def test_every_yield_has_seven_values_matching_outputs(self):
        updates = self.drive("live", self.events())
        self.assertGreater(len(updates), 8)
        for update in updates:
            self.assertEqual(len(update), 7)
            self.assertFalse(inspect.isgenerator(update))
        videos, generated, download, metrics, status, stages, html = updates[-1]
        self.assertEqual((videos, generated, download), ("gt.mp4", "gen.mp4", "gen.mp4"))
        self.assertIn(">1<", metrics)
        self.assertIn("Generation complete", html)
        self.assertIn("100/100 steps", html)
        self.assertIn("Sample 7", status)
        self.assertIn("Clip clip008", status)
        self.assertIn("Detailed Progress", stages)
        self.assertIn("Diffusion generation", stages)

    def test_progress_arrives_before_final_result(self):
        updates = self.drive("live", self.events())
        mid = [u for u in updates if "50/100 steps" in u[6]]
        self.assertTrue(mid)
        self.assertIsNone(mid[0][1])  # no video is presented before the result event
        self.assertTrue(any("Estimating ETA" in u[6] for u in updates))

    def test_idle_ticks_refresh_elapsed_time(self):
        updates = self.drive("live", self.events(), delay_before=3)
        self.assertGreater(len(updates), 10)

    def test_failure_is_reported_without_video(self):
        updates = self.drive("live", self.events(live_ok=False), returncode=2)
        for update in updates:
            self.assertEqual(len(update), 7)
        final = updates[-1]
        self.assertIsNone(final[0])
        self.assertIsNone(final[1])
        self.assertIn("failed", final[6].lower())
        self.assertIn("boom", final[6])
        self.assertIn("&quot;status&quot;: &quot;failed&quot;", final[5])

    def test_crash_without_events_marks_running_stage_failed(self):
        lines = [json.dumps({"type": "stage", "stage": "Visual prediction", "status": "running", "detail": ""})]
        updates = self.drive("live", lines, returncode=1)
        self.assertIn("failed", updates[-1][6].lower())
        self.assertIn("&quot;status&quot;: &quot;failed&quot;", updates[-1][5])

    def test_precomputed_unaffected(self):
        lines = [json.dumps({"type": "stage", "stage": "EEG validation", "status": "complete", "detail": ""}),
                 json.dumps({"type": "stage", "stage": "Diffusion generation", "status": "not_run", "detail": "x"}),
                 json.dumps({"type": "result", "result": {
                     "ground_truth_video": "gt.mp4", "generated_video": "gen.mp4",
                     "download_video": "gen.mp4", "metrics": {}, "message": "ok"}})]
        updates = self.drive("precomputed", lines)
        for update in updates:
            self.assertEqual(len(update), 7)
        self.assertIn("no model inference performed", updates[-1][6])
        self.assertIn("Existing reconstruction loaded", updates[-1][4])
        self.assertEqual(updates[-1][1], "gen.mp4")

    def test_outputs_declared_in_order(self):
        with mock.patch.object(self.app, "_preflight", return_value={
            "precomputed_ready": True, "live_ready": True, "missing_precomputed": [], "missing_live": [],
            "worker_modules": {},
        }), mock.patch.object(self.app, "_cleanup_stale_runs"):
            demo = self.app.build_app()
        handlers = {f.fn.__name__: f for f in demo.fns.values()}
        self.assertEqual(set(handlers), {"run_precomputed", "run_live"})
        for handler in handlers.values():
            self.assertEqual(
                [type(c).__name__ for c in handler.outputs],
                ["Video", "Video", "File", "HTML", "HTML", "HTML", "HTML"],
            )


if __name__ == "__main__":
    unittest.main()
