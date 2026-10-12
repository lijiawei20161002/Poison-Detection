"""Integration checks for the public export's data and history boundaries."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("build_release", Path(__file__).with_name("build_release.py"))
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseChecks(unittest.TestCase):
    def fixture(self, base):
        source = base / "source"
        source.mkdir()
        (source / "README.md").write_text("Code-only fixture.\n")
        (source / "release_files.txt").write_text("README.md\nrelease_files.txt\n")
        return source

    def test_only_allowlisted_files_and_no_history_are_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            source = self.fixture(base)
            for name in [".git/objects/private", "data/train.jsonl", "cache/prompts.json", "_restricted_baseline.py"]:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("private fixture")
            dest = release.build(source, base / "public")
            self.assertEqual(release.verify(dest), 2)
            self.assertFalse((dest / ".git").exists())
            (dest / "unexpected.jsonl").write_text("private fixture")
            with self.assertRaises(ValueError):
                release.verify(dest)

    def test_sensitive_paths_cannot_be_added_to_allowlist(self):
        for name in ["data/train.jsonl", "coling2027/results/run/train.jsonl.gz",
                     "coling2027/_restricted_zscore.py", "../outside.py", "/absolute.py",
                     "model.safetensors", "poison_detection/data/cache/prompt.txt"]:
            self.assertFalse(release.approved_path(name), name)
        self.assertTrue(release.approved_path("poison_detection/data/loader.py"))

    def test_symlink_cannot_export_unreviewed_content(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            source = self.fixture(base)
            (base / "secret").write_text("private fixture")
            (source / "README.md").unlink()
            (source / "README.md").symlink_to(base / "secret")
            with self.assertRaises(ValueError):
                release.build(source, base / "public")
            self.assertFalse((base / "public").exists())

    def test_destination_and_integrity_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            source = self.fixture(base)
            with self.assertRaises(ValueError):
                release.build(source, source / "nested")
            dest = release.build(source, base / "public")
            with self.assertRaises(FileExistsError):
                release.build(source, dest)
            (dest / "README.md").write_text("changed")
            with self.assertRaises(ValueError):
                release.verify(dest)


if __name__ == "__main__":
    unittest.main()
