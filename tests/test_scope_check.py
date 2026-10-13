"""Tests for read-only changed-file scope checks."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.scope_check import check_file_scope


class FileScopeTests(unittest.TestCase):
    def check_output(self, output: bytes, permitted: list[str]):
        with patch("auto_coding.scope_check.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=output)
            result = check_file_scope("repo", permitted)
        run.assert_called_once_with(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd="repo", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, shell=False, check=False, timeout=10,
        )
        return result

    def test_allowed_tracked_modification_and_untracked_file(self) -> None:
        result = self.check_output(
            b" M src/app.py\0?? tests/new test.py\0",
            ["src/app.py", "tests/new test.py"],
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.changed_files, ("src/app.py", "tests/new test.py"))
        self.assertEqual(result.out_of_scope_paths, ())
        self.assertIsNone(result.error_code)

    def test_disallowed_modified_and_untracked_files(self) -> None:
        result = self.check_output(
            b" M private.txt\0?? extra.py\0",
            ["src/app.py"],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.out_of_scope_paths, ("private.txt", "extra.py"))

    def test_staged_changes_and_deletions(self) -> None:
        result = self.check_output(
            b"M  staged.py\0D  deleted.py\0",
            ["staged.py", "deleted.py"],
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.changed_files, ("staged.py", "deleted.py"))

    def test_renames_and_copies_check_both_paths(self) -> None:
        for status in (b"R ", b"C "):
            with self.subTest(status=status):
                output = status + b" new name.py\0old name.py\0"
                for permitted, outside in (
                    (["new name.py"], "old name.py"),
                    (["old name.py"], "new name.py"),
                ):
                    result = self.check_output(output, permitted)
                    self.assertFalse(result.passed)
                    self.assertEqual(result.changed_files, ("new name.py", "old name.py"))
                    self.assertEqual(result.out_of_scope_paths, (outside,))
                self.assertTrue(
                    self.check_output(output, ["new name.py", "old name.py"]).passed
                )

    def test_case_insensitive_matching_does_not_expand_other_paths(self) -> None:
        result = self.check_output(
            b" M SRC/APP.PY\0?? src/other.py\0",
            ["src/app.py"],
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.out_of_scope_paths, ("src/other.py",))

    def test_clean_status_passes_with_empty_scope(self) -> None:
        result = self.check_output(b"", [])
        self.assertTrue(result.passed)
        self.assertEqual(result.changed_files, ())

    def test_git_failures_and_timeouts_fail_closed(self) -> None:
        with patch("auto_coding.scope_check.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stdout=b" M allowed.py\0"
            )
            failed = check_file_scope("repo", ["allowed.py"])
            run.side_effect = subprocess.TimeoutExpired("git", 10, output=b"secret")
            timed_out = check_file_scope("repo", ["allowed.py"])
            run.side_effect = FileNotFoundError("secret path")
            missing = check_file_scope("repo", ["allowed.py"])

        for result, code in (
            (failed, "GIT_FAILED"),
            (timed_out, "GIT_TIMEOUT"),
            (missing, "GIT_LAUNCH_FAILED"),
        ):
            self.assertFalse(result.passed)
            self.assertEqual(result.changed_files, ())
            self.assertEqual(result.error_code, code)
            self.assertNotIn("secret", repr(result))

    def test_malformed_output_fails_closed(self) -> None:
        for output in (b" M file.py", b"R  new.py\0", b"Q  file.py\0",
                       b"?? \0", b" M bad-\xff.py\0"):
            with self.subTest(output=output):
                result = self.check_output(output, ["file.py", "new.py"])
                self.assertFalse(result.passed)
                self.assertEqual(result.error_code, "MALFORMED_STATUS")
                self.assertEqual(result.changed_files, ())

    def test_invalid_permitted_list_fails_closed(self) -> None:
        with patch("auto_coding.scope_check.subprocess.run") as run:
            result = check_file_scope("repo", "file.py")
        run.assert_not_called()
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "INVALID_PERMITTED_FILES")

    def test_inspected_repository_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            before = tuple(path.name for path in Path(directory).iterdir())
            with patch("auto_coding.scope_check.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=b" M keep.txt\0"
                )
                result = check_file_scope(directory, ["keep.txt"])
            self.assertTrue(result.passed)
            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(tuple(path.name for path in Path(directory).iterdir()), before)


if __name__ == "__main__":
    unittest.main()
