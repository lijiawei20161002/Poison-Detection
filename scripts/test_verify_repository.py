"""Exercise leaks that survive deleting sensitive files from the latest commit."""
from pathlib import Path
import subprocess
import tempfile
import unittest

from verify_repository import verify_repository


class RepositoryChecks(unittest.TestCase):
    def fixture(self, root):
        self.run_git(root, "init", "-q", "--initial-branch=main")
        self.run_git(root, "config", "user.name", "Release test")
        self.run_git(root, "config", "user.email", "test@localhost")
        self.run_git(root, "config", "commit.gpgsign", "false")
        self.run_git(root, "config", "core.hooksPath", str(root / "no-hooks"))
        (root / "README.md").write_text("Synthetic code-only fixture.\n")
        (root / "release_files.txt").write_text("README.md\nrelease_files.txt\n")
        self.commit(root)

    def run_git(self, root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT)

    def commit(self, root):
        self.run_git(root, "add", "--all")
        self.run_git(root, "commit", "-qm", "Synthetic fixture")

    def test_clean_repository_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            self.assertEqual(verify_repository(root)["commits"], 1)

    def test_deleted_data_still_in_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            secret = root / "data" / "train.jsonl"
            secret.parent.mkdir()
            secret.write_text('{}\n')
            self.commit(root)
            secret.unlink()
            self.commit(root)
            with self.assertRaisesRegex(ValueError, "Excluded historical path"):
                verify_repository(root)

    def test_baseline_inside_allowlisted_historical_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            with (root / "release_files.txt").open("a") as stream:
                stream.write("core.py\n")
            code = root / "core.py"
            code.write_text("class ReleasedUnigramZScore:\n    def fit(self): pass\n")
            self.commit(root)
            code.write_text("# Excluded baseline removed.\n")
            self.commit(root)
            with self.assertRaisesRegex(ValueError, "Excluded translated baseline"):
                verify_repository(root)

    def test_remote_tracking_history_is_also_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            clean = self.run_git(root, "rev-parse", "HEAD").decode().strip()
            self.run_git(root, "checkout", "-qb", "legacy")
            (root / "private.json").write_text('{}\n')
            self.commit(root)
            legacy = self.run_git(root, "rev-parse", "HEAD").decode().strip()
            self.run_git(root, "update-ref", "refs/remotes/origin/legacy", legacy)
            self.run_git(root, "checkout", "-q", "main")
            self.run_git(root, "branch", "-D", "legacy")
            self.assertEqual(self.run_git(root, "rev-parse", "HEAD").decode().strip(), clean)
            with self.assertRaisesRegex(ValueError, "Excluded historical path"):
                verify_repository(root)


if __name__ == "__main__":
    unittest.main()
