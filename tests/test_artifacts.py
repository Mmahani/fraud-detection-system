"""Unit tests for the artifact manifest."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud.artifacts import (
    DEFAULT_MANIFEST_PATH,
    build_manifest,
    validate_manifest,
)


class ArtifactManifestTests(unittest.TestCase):
    def test_default_manifest_exists_and_validates(self) -> None:
        self.assertTrue(DEFAULT_MANIFEST_PATH.is_file())
        result = validate_manifest(check_hashes=False)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_build_and_validate_temporary_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest = build_manifest(manifest_path, strict=True, hash_files=False)
            self.assertEqual(manifest["summary"]["missing_required"], [])
            self.assertGreater(manifest["summary"]["present"], 0)

            result = validate_manifest(manifest_path, check_hashes=False)
            self.assertTrue(result["valid"])
            self.assertEqual(result["errors"], [])

    def test_missing_manifest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                validate_manifest(manifest_path, check_hashes=False)


if __name__ == "__main__":
    unittest.main()
