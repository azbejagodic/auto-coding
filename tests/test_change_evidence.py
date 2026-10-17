"""Tests for read-only collection of pending Git change evidence."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding import change_evidence
from auto_coding.change_evidence import collect_change_evidence


class ChangeEvidenceTests(unittest.TestCase):
    def test_git_timeout_override_reaches_scope_and_every_evidence_query(self):
        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        (self.repository / "new.bin").write_bytes(b"\0\xff")
        run = subprocess.run
        with patch("auto_coding.change_evidence.subprocess.run", wraps=run) as execute:
            result = collect_change_evidence(self.repository, ["tracked.txt", "new.bin"], 100000,
                                             timeout_seconds=7)
        self.assertTrue(result.passed, result)
        self.assertEqual(len(execute.call_args_list), 4)
        self.assertEqual([call.kwargs["timeout"] for call in execute.call_args_list], [7] * 4)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Evidence Tests")
        self.git("config", "user.email", "evidence@example.invalid")
        (self.repository / "tracked.txt").write_bytes(b"original\n")
        (self.repository / "delete.txt").write_bytes(b"delete me\n")
        (self.repository / "old name.txt").write_bytes(b"rename me\n")
        self.git("add", ".")
        self.git("commit", "-qm", "initial")

    def git(self, *arguments):
        return subprocess.run(
            ["git", *arguments], cwd=self.repository, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            check=True, timeout=10,
        ).stdout

    def collect(self, permitted, max_bytes=1_000_000):
        return collect_change_evidence(self.repository, permitted, max_bytes)

    def test_modified_tracked_file_and_stable_fingerprint(self):
        sentinel = "D11_CONTENT_SENTINEL_8f31c07a"
        (self.repository / "tracked.txt").write_bytes(f"{sentinel}\n".encode("ascii"))
        first = self.collect(["tracked.txt"])
        second = self.collect(["tracked.txt"])
        self.assertTrue(first.passed)
        self.assertEqual(first.changed_paths, ("tracked.txt",))
        self.assertIn(sentinel.encode("ascii"), first.tracked_diff)
        self.assertEqual(first.untracked_files, ())
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(len(first.fingerprint), 64)
        self.assertNotIn(sentinel, repr(first))

    def test_staged_and_unstaged_changes_are_both_in_diff(self):
        (self.repository / "tracked.txt").write_bytes(b"staged\n")
        self.git("add", "tracked.txt")
        (self.repository / "tracked.txt").write_bytes(b"unstaged\n")
        result = self.collect(["tracked.txt"])
        self.assertTrue(result.passed)
        self.assertIn(b"+unstaged", result.tracked_diff)
        self.assertNotIn(b"+staged", result.tracked_diff)

    def test_untracked_binary_file_is_complete_and_changes_fingerprint(self):
        path = self.repository / "new file.bin"
        path.write_bytes(b"\x00\xff\x80private bytes")
        first = self.collect(["new file.bin"])
        self.assertTrue(first.passed)
        self.assertEqual(first.changed_paths, ("new file.bin",))
        self.assertEqual(first.untracked_files[0].path, "new file.bin")
        self.assertEqual(first.untracked_files[0].contents, b"\x00\xff\x80private bytes")
        self.assertNotIn("private bytes", repr(first))
        self.assertNotIn("private bytes", repr(first.untracked_files[0]))
        path.write_bytes(b"\x00\xff\x80different bytes")
        second = self.collect(["new file.bin"])
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_deleted_and_renamed_tracked_files(self):
        (self.repository / "delete.txt").unlink()
        self.git("mv", "old name.txt", "new name.txt")
        result = self.collect(["delete.txt", "old name.txt", "new name.txt"])
        self.assertTrue(result.passed, result.error_code)
        self.assertEqual(set(result.changed_paths), {"delete.txt", "old name.txt", "new name.txt"})
        self.assertIn(b"deleted file mode", result.tracked_diff)
        self.assertIn(b"new file mode", result.tracked_diff)
        self.assertEqual(result.untracked_files, ())

    def test_out_of_scope_change_stops_collection(self):
        (self.repository / "unapproved.txt").write_bytes(b"secret")
        result = self.collect(["tracked.txt"])
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "OUT_OF_SCOPE")
        self.assertEqual(result.tracked_diff, b"")
        self.assertEqual(result.untracked_files, ())

    def test_unsafe_path_is_rejected(self):
        (self.repository / "bad[1].txt").write_bytes(b"unsafe")
        result = self.collect(["bad[1].txt"])
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "UNSAFE_PATH")

    def test_symlink_is_rejected(self):
        target = self.repository / "tracked.txt"
        link = self.repository / "link.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        result = self.collect(["link.txt"])
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "UNSAFE_PATH")

    def test_evidence_size_limit_and_invalid_limit(self):
        (self.repository / "new.bin").write_bytes(b"x" * 1000)
        oversized = self.collect(["new.bin"], 100)
        self.assertFalse(oversized.passed)
        self.assertEqual(oversized.error_code, "EVIDENCE_TOO_LARGE")
        self.assertEqual(self.collect(["new.bin"], 0).error_code, "INVALID_MAX_BYTES")
        self.assertEqual(self.collect(["new.bin"], True).error_code, "INVALID_MAX_BYTES")

    def test_oversized_raw_tracked_diff_is_rejected(self):
        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        result = self.collect(["tracked.txt"], 16)
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "EVIDENCE_TOO_LARGE")

    def test_oversized_untracked_path_listing_is_rejected(self):
        (self.repository / "untracked.txt").write_bytes(b"x")
        result = self.collect(["untracked.txt"], 8)
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "EVIDENCE_TOO_LARGE")

    def test_oversized_full_tracked_diff_is_rejected(self):
        (self.repository / "tracked.txt").write_bytes(b"x" * 1000)
        result = self.collect(["tracked.txt"], 256)
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "EVIDENCE_TOO_LARGE")

    def test_git_temporary_files_use_external_directory_when_system_temp_is_in_repo(self):
        in_repository_temp = self.repository / "configured-temp"
        in_repository_temp.mkdir()
        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        original_temporary_file = tempfile.TemporaryFile
        used_directories = []

        def record_temporary_file(*args, **kwargs):
            used_directories.append(Path(kwargs["dir"]).resolve(strict=True))
            return original_temporary_file(*args, **kwargs)

        with patch("auto_coding.change_evidence.tempfile.gettempdir",
                   return_value=str(in_repository_temp)), \
             patch("auto_coding.change_evidence.tempfile.TemporaryFile",
                   side_effect=record_temporary_file):
            result = self.collect(["tracked.txt"])

        self.assertTrue(result.passed, result.error_code)
        self.assertEqual(len(used_directories), 3)
        self.assertTrue(all(not directory.is_relative_to(self.repository)
                            for directory in used_directories))

    def test_collection_fails_closed_without_external_temp_directory(self):
        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        with patch("auto_coding.change_evidence._external_temporary_directory",
                   return_value=None), \
             patch("auto_coding.change_evidence.tempfile.TemporaryFile") as temporary_file:
            result = self.collect(["tracked.txt"])

        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "GIT_LAUNCH_FAILED")
        temporary_file.assert_not_called()

    def test_git_temporary_outputs_are_closed_after_success_failure_and_timeout(self):
        original_temporary_file = tempfile.TemporaryFile
        outputs = []

        def tracked_temporary_file(*args, **kwargs):
            output = original_temporary_file(*args, **kwargs)
            outputs.append(output)
            return output

        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        with patch("auto_coding.change_evidence.tempfile.TemporaryFile",
                   side_effect=tracked_temporary_file):
            result = self.collect(["tracked.txt"])
        self.assertTrue(result.passed)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.closed for output in outputs))

        outputs.clear()
        original_run = subprocess.run

        def fail_evidence_git(*args, **kwargs):
            if args[0][1] == "status":
                return original_run(*args, **kwargs)
            return subprocess.CompletedProcess(args[0], 1)

        with patch("auto_coding.change_evidence.tempfile.TemporaryFile",
                   side_effect=tracked_temporary_file), \
             patch("auto_coding.change_evidence.subprocess.run",
                   side_effect=fail_evidence_git):
            result = self.collect(["tracked.txt"])
        self.assertEqual(result.error_code, "GIT_FAILED")
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].closed)

        outputs.clear()

        def timeout_evidence_git(*args, **kwargs):
            if args[0][1] == "status":
                return original_run(*args, **kwargs)
            raise subprocess.TimeoutExpired(args[0], 10)

        with patch("auto_coding.change_evidence.tempfile.TemporaryFile",
                   side_effect=tracked_temporary_file), \
             patch("auto_coding.change_evidence.subprocess.run",
                   side_effect=timeout_evidence_git):
            result = self.collect(["tracked.txt"])
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].closed)

    def test_git_capture_uses_files_and_reads_only_limit_plus_one(self):
        calls = []
        read_sizes = []
        original_temporary_file = tempfile.TemporaryFile

        class SpyFile:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def read(self, size=-1):
                read_sizes.append(size)
                return self.wrapped.read(size)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.wrapped.close()
                return False

        def spy_temporary_file(*args, **kwargs):
            return SpyFile(original_temporary_file(*args, **kwargs))

        original_run = subprocess.run

        def inspect_run(*args, **kwargs):
            if args[0][1] != "status":
                calls.append(kwargs["stdout"])
            return original_run(*args, **kwargs)

        (self.repository / "tracked.txt").write_bytes(b"changed\n")
        with patch("auto_coding.change_evidence.tempfile.TemporaryFile",
                   side_effect=spy_temporary_file), \
             patch("auto_coding.change_evidence.subprocess.run", side_effect=inspect_run):
            self.assertTrue(self.collect(["tracked.txt"]).passed)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(hasattr(output, "read") and output != subprocess.PIPE
                            for output in calls))
        self.assertEqual(read_sizes, [1_000_001] * 3)

    def test_git_failure_stops_collection(self):
        (self.repository / "tracked.txt").write_bytes(b"changed")
        with patch("auto_coding.change_evidence._git", return_value=(None, "GIT_FAILED")):
            result = self.collect(["tracked.txt"])
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "GIT_FAILED")

    def test_collection_does_not_modify_repository(self):
        (self.repository / "tracked.txt").write_bytes(b"changed")
        (self.repository / "new.bin").write_bytes(b"\x00\xff")
        before_status = self.git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        before_tracked = (self.repository / "tracked.txt").read_bytes()
        before_untracked = (self.repository / "new.bin").read_bytes()
        result = self.collect(["tracked.txt", "new.bin"])
        self.assertTrue(result.passed)
        self.assertEqual(self.git("status", "--porcelain=v1", "-z", "--untracked-files=all"),
                         before_status)
        self.assertEqual((self.repository / "tracked.txt").read_bytes(), before_tracked)
        self.assertEqual((self.repository / "new.bin").read_bytes(), before_untracked)


if __name__ == "__main__":
    unittest.main()
