import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from webapp.core import (
    PipelineError,
    get_asset_paths,
    identify_sample,
    safe_data_path,
    validate_upload_file,
)


class UploadValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, value, name="input.npy"):
        path = self.root / name
        np.save(path, value, allow_pickle=True)
        return path

    def test_accepts_single_and_singleton_batch_without_reshape(self):
        sample = np.zeros((62, 400), dtype=np.float64)
        single = self.save(sample)
        batch = self.save(sample[None], "batch.npy")
        self.assertEqual(validate_upload_file(single, 200, True).shape, (62, 400))
        self.assertEqual(validate_upload_file(batch, 200, True).shape, (1, 62, 400))

    def test_accepts_exact_supported_full_test_shape(self):
        path = self.save(np.zeros((200, 62, 400), dtype=np.float32))
        self.assertEqual(validate_upload_file(path, 200, True).shape, (200, 62, 400))

    def test_rejects_unsupported_shape_instead_of_reshaping(self):
        path = self.save(np.zeros((62, 401), dtype=np.float32))
        with self.assertRaisesRegex(PipelineError, "No reshape or resampling"):
            validate_upload_file(path, 200, True)

    def test_rejects_nonfinite_values(self):
        sample = np.zeros((62, 400), dtype=np.float32)
        sample[0, 0] = np.nan
        with self.assertRaisesRegex(PipelineError, "NaN or infinity"):
            validate_upload_file(self.save(sample), 200, True)

    def test_rejects_complex_values(self):
        path = self.save(np.zeros((62, 400), dtype=np.complex64))
        with self.assertRaisesRegex(PipelineError, "real numeric dtype"):
            validate_upload_file(path, 200, True)

    def test_rejects_object_arrays_without_pickle_loading(self):
        path = self.root / "object.npy"
        np.save(path, np.array([{"not": "EEG"}], dtype=object), allow_pickle=True)
        with self.assertRaisesRegex(PipelineError, "non-pickled"):
            validate_upload_file(path, 200, True)

    def test_rejects_non_npy_and_corrupt_files(self):
        text_path = self.root / "input.txt"
        text_path.write_text("not an EEG array", encoding="utf-8")
        with self.assertRaisesRegex(PipelineError, r"\.npy"):
            validate_upload_file(text_path, 200, True)
        corrupt_path = self.root / "corrupt.npy"
        corrupt_path.write_bytes(b"not a NumPy file")
        with self.assertRaisesRegex(PipelineError, "non-pickled"):
            validate_upload_file(corrupt_path, 200, True)

    def test_rejects_wrong_sampling_rate_and_unconfirmed_channels(self):
        path = self.save(np.zeros((62, 400), dtype=np.float32))
        with self.assertRaisesRegex(PipelineError, "exactly 200 Hz"):
            validate_upload_file(path, 250, True)
        with self.assertRaisesRegex(PipelineError, "Confirm"):
            validate_upload_file(path, 200, False)


class SampleIdentificationTests(unittest.TestCase):
    def test_identifies_only_exact_signal_values(self):
        canonical = np.zeros((200, 62, 400), dtype=np.float32)
        canonical[51].fill(0.25)
        sample, index = identify_sample(canonical[51], canonical, 0)
        self.assertEqual(index, 51)
        np.testing.assert_array_equal(sample, canonical[51])
        changed = canonical[51].copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(1))
        _, unmatched = identify_sample(changed, canonical, 51)
        self.assertEqual(unmatched, -1)

    def test_full_test_upload_requires_exact_canonical_values(self):
        canonical = np.zeros((200, 62, 400), dtype=np.float32)
        sample, index = identify_sample(canonical.copy(), canonical, 199)
        self.assertEqual(index, 199)
        np.testing.assert_array_equal(sample, canonical[199])
        altered = canonical.copy()
        altered[199, 0, 0] = 1
        with self.assertRaisesRegex(PipelineError, "exactly matches"):
            identify_sample(altered, canonical, 199)

    def test_rejects_ambiguous_exact_matches(self):
        canonical = np.zeros((200, 62, 400), dtype=np.float32)
        with self.assertRaisesRegex(PipelineError, "ambiguous"):
            identify_sample(canonical[0], canonical, 0)

    def test_manifest_paths_cannot_escape_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(safe_data_path(root, "outputs/sample.npy"), root / "outputs" / "sample.npy")
            for value in ("../outside.npy", r"outputs\..\outside.npy", r"C:\outside.npy"):
                with self.subTest(path=value), self.assertRaises(PipelineError):
                    safe_data_path(root, value)


class AssetConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "asset_config.json"
        self.config = {
            "data_root": ".",
            "model_root": ".",
            "channel_file": "SEED-DV/channel_62_pos.locs",
            "test_eeg": "outputs/test_eeg.npy",
            "sample_mapping": "outputs/sample_mapping.csv",
            "paired_manifest": "outputs/paired_samples.csv",
            "generation_output": "outputs/final",
            "metrics_csv": "outputs/final/metrics.csv",
            "diffusion_output": "outputs/diffusion",
            "visual_output": "outputs/visual",
            "semantic_output": "outputs/semantic",
            "dynamic_output": "outputs/dynamic",
            "negative_embedding": "FinalEvaluation/final_negative.npy",
            "sd14_model": "models/sd14",
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_requires_an_explicit_asset_root(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(PipelineError, "EEG2VIDEO_ASSETS_DIR is not set"):
                get_asset_paths()

    def test_resolves_configured_paths_under_private_asset_root(self):
        with patch.dict(
            os.environ,
            {
                "EEG2VIDEO_ASSETS_DIR": str(self.root),
                "EEG2VIDEO_ASSET_CONFIG": str(self.config_path),
            },
            clear=True,
        ):
            paths = get_asset_paths()
        self.assertEqual(paths.data_root, self.root.resolve())
        self.assertEqual(paths.test_eeg, self.root / "outputs" / "test_eeg.npy")
        self.assertEqual(paths.sd14_model, self.root / "models" / "sd14")

    def test_rejects_absolute_or_parent_relative_asset_paths(self):
        for unsafe in ("../outside.npy", str(self.root.parent / "outside.npy")):
            with self.subTest(path=unsafe):
                self.config["test_eeg"] = unsafe
                self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
                with patch.dict(
                    os.environ,
                    {
                        "EEG2VIDEO_ASSETS_DIR": str(self.root),
                        "EEG2VIDEO_ASSET_CONFIG": str(self.config_path),
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(PipelineError, "must stay inside"):
                        get_asset_paths()

    def test_missing_asset_configuration_is_actionable(self):
        self.config_path.unlink()
        with patch.dict(os.environ, {"EEG2VIDEO_ASSETS_DIR": str(self.root)}, clear=True):
            with self.assertRaisesRegex(PipelineError, "asset_config.example.json"):
                get_asset_paths()

    def test_cache_cannot_overlap_asset_root_even_when_config_is_missing(self):
        from webapp import app

        with patch.dict(
            os.environ,
            {
                "EEG2VIDEO_ASSETS_DIR": str(self.root),
                "EEG2VIDEO_CACHE_DIR": str(self.root / "cache"),
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                app._validate_cache_root()


if __name__ == "__main__":
    unittest.main()
