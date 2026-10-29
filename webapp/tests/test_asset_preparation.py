import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.prepare_webapp_assets import PlanItem, _copy_item, _safe_relative


class AssetPreparationTests(unittest.TestCase):
    def test_rejects_paths_that_escape_asset_roots(self):
        for value in ("../outside.npy", r"outputs\..\outside.npy", r"C:\outside.npy"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                _safe_relative(value)

    def test_copies_and_verifies_a_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"verified test asset")
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            item = PlanItem(
                source=str(source),
                destination="nested/copy.bin",
                label="test",
                expected_sha256=expected_hash,
            )
            result = _copy_item(item, root / "assets")
            self.assertEqual(result["status"], "COPIED")
            self.assertEqual(
                result["actual_destination_sha256"],
                expected_hash,
            )

    def test_never_overwrites_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "assets" / "copy.bin"
            source.write_bytes(b"new source")
            destination.parent.mkdir()
            destination.write_bytes(b"existing destination")
            item = PlanItem(
                source=str(source),
                destination="copy.bin",
                label="test",
                expected_sha256=None,
            )
            result = _copy_item(item, root / "assets")
            self.assertEqual(result["status"], "CONFLICT")
            self.assertEqual(destination.read_bytes(), b"existing destination")


if __name__ == "__main__":
    unittest.main()
