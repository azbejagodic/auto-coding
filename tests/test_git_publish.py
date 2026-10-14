"""Tests for read-only Git publishing preflight."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.git_publish import check_publish_preflight


class GitPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.repository = self.parent / "repo"
        self.git_directory = self.repository / ".git"
        self.git_directory.mkdir(parents=True)
        self.branch = "feature/preflight"
        self.head = "a" * 40
        self.origin = "https://example.invalid/repo.git"
        self.remote_args = ("ls-remote", "--exit-code", "--refs", "origin",
                            f"refs/heads/{self.branch}")
        self.index_args = ("diff", "--cached", "--quiet", "--exit-code",
                           "--no-ext-diff", "--no-textconv", "HEAD", "--")
        self.responses = {
            ("rev-parse", "--show-toplevel"): (0, f"{self.repository}\n".encode()),
            ("rev-parse", "--absolute-git-dir"): (0, f"{self.git_directory}\n".encode()),
            ("symbolic-ref", "--quiet", "HEAD"): (0, f"refs/heads/{self.branch}\n".encode()),
            ("rev-parse", "--verify", "HEAD^{commit}"): (0, f"{self.head}\n".encode()),
            ("remote", "get-url", "--all", "origin"): (0, f"{self.origin}\n".encode()),
            ("remote", "get-url", "--push", "--all", "origin"): (0, f"{self.origin}\n".encode()),
            self.index_args: (0, b""),
            self.remote_args: (0, f"{self.head}\trefs/heads/{self.branch}\n".encode()),
        }
        self.outputs = []
        self.calls = []
        clean_environment = {
            key: value for key, value in os.environ.items()
            if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
                           "GIT_INDEX_FILE", "GIT_NAMESPACE")
        }
        environment_patch = patch.dict(os.environ, clean_environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def execute(self, command, **kwargs):
        self.assertEqual(command[:2], ["git", "--no-optional-locks"])
        arguments = tuple(command[2:])
        self.calls.append(arguments)
        self.assertEqual(kwargs["cwd"], self.repository)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertFalse(kwargs["shell"])
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["timeout"], 3)
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["env"]["GCM_INTERACTIVE"], "Never")
        output = kwargs["stdout"]
        self.assertNotEqual(output, subprocess.PIPE)
        if isinstance(output.name, str):
            self.assertFalse(Path(output.name).resolve().is_relative_to(self.repository))
        self.outputs.append(output)
        response = self.responses[arguments]
        if isinstance(response, Exception):
            raise response
        code, contents = response
        output.write(contents)
        return subprocess.CompletedProcess(command, code)

    def check(self, **changes):
        inputs = {
            "repository_directory": self.repository, "expected_branch": self.branch,
            "expected_head": self.head, "expected_origin_url": self.origin,
            "timeout_seconds": 3,
        }
        with patch("auto_coding.git_publish.subprocess.run", side_effect=self.execute):
            result = check_publish_preflight(**(inputs | changes))
        self.assertTrue(all(output.closed for output in self.outputs))
        return result

    def test_correct_state_passes_with_exact_read_only_commands(self):
        result = self.check()
        self.assertTrue(result.passed)
        self.assertIsNone(result.error_code)
        self.assertEqual(self.calls, list(self.responses))

    def test_wrong_repository_root_fails(self):
        self.responses[("rev-parse", "--show-toplevel")] = (0, f"{self.parent}\n".encode())
        self.assertEqual(self.check().error_code, "REPOSITORY_ROOT_MISMATCH")
        self.assertEqual(len(self.calls), 1)

    def test_wrong_branch_and_detached_head_fail(self):
        arguments = ("symbolic-ref", "--quiet", "HEAD")
        self.responses[arguments] = (0, b"refs/heads/other\n")
        self.assertEqual(self.check().error_code, "BRANCH_MISMATCH")
        self.responses[arguments] = (1, b"")
        self.assertEqual(self.check().error_code, "GIT_FAILED")
        self.assertNotIn(self.remote_args, self.calls)

    def test_changed_local_head_fails(self):
        self.responses[("rev-parse", "--verify", "HEAD^{commit}")] = (0, b"b" * 40 + b"\n")
        self.assertEqual(self.check().error_code, "HEAD_MISMATCH")
        self.assertNotIn(self.remote_args, self.calls)

    def test_wrong_fetch_or_push_origin_fails(self):
        for arguments in (
            ("remote", "get-url", "--all", "origin"),
            ("remote", "get-url", "--push", "--all", "origin"),
        ):
            with self.subTest(arguments=arguments):
                original = self.responses[arguments]
                self.responses[arguments] = (0, b"https://other.invalid/private\n")
                result = self.check()
                self.assertEqual(result.error_code, "ORIGIN_MISMATCH")
                self.assertNotIn("private", repr(result))
                self.responses[arguments] = original

    def test_remote_sha_mismatch_and_inaccessible_remote_fail(self):
        self.responses[self.remote_args] = (0, b"b" * 40 + f"\trefs/heads/{self.branch}\n".encode())
        self.assertEqual(self.check().error_code, "REMOTE_HEAD_MISMATCH")
        self.responses[self.remote_args] = (2, b"")
        self.assertEqual(self.check().error_code, "GIT_FAILED")

    def test_preexisting_staged_changes_stop_before_remote_query(self):
        self.responses[self.index_args] = (1, b"")
        self.assertEqual(self.check().error_code, "STAGED_CHANGES")
        self.assertNotIn(self.remote_args, self.calls)

    def test_unstaged_and_untracked_changes_are_allowed_without_modification(self):
        (self.repository / "changed.txt").write_bytes(b"unstaged")
        (self.repository / "untracked.txt").write_bytes(b"untracked")
        before = self.snapshot(self.repository)
        self.assertTrue(self.check().passed)
        self.assertEqual(self.snapshot(self.repository), before)

    def test_recognized_git_operations_fail_closed(self):
        for marker in (
            "MERGE_HEAD", "MERGE_AUTOSTASH", "CHERRY_PICK_HEAD", "REVERT_HEAD",
            "REBASE_HEAD", "rebase-merge", "rebase-apply", "sequencer",
            "BISECT_LOG", "BISECT_START",
        ):
            with self.subTest(marker=marker):
                path = self.git_directory / marker
                path.write_bytes(b"operation")
                self.assertEqual(self.check().error_code, "GIT_OPERATION_IN_PROGRESS")
                self.assertEqual(path.read_bytes(), b"operation")
                path.unlink()

    def test_existing_index_lock_is_preserved(self):
        lock = self.git_directory / "index.lock"
        lock.write_bytes(b"existing lock")
        self.assertEqual(self.check().error_code, "INDEX_LOCK_PRESENT")
        self.assertEqual(lock.read_bytes(), b"existing lock")
        self.assertNotIn(self.index_args, self.calls)

    def test_git_launch_failure_timeout_and_nonzero_exit_are_sanitized(self):
        arguments = ("rev-parse", "--show-toplevel")
        for response, error in (
            (FileNotFoundError("private executable"), "GIT_IO_FAILED"),
            (subprocess.TimeoutExpired("private command", 3, output=b"private"), "GIT_TIMEOUT"),
            ((7, b"private output"), "GIT_FAILED"),
        ):
            with self.subTest(error=error):
                self.responses[arguments] = response
                result = self.check()
                self.assertFalse(result.passed)
                self.assertEqual(result.error_code, error)
                self.assertNotIn("private", repr(result))

    def test_malformed_incomplete_and_ambiguous_metadata_fail(self):
        arguments = ("rev-parse", "--show-toplevel")
        for output in (b"", b"no newline", b"one\ntwo\n", b"nul\0path\n", b"\xff\n"):
            with self.subTest(output=output):
                self.responses[arguments] = (0, output)
                self.assertEqual(self.check().error_code, "MALFORMED_GIT_OUTPUT")

    def test_malformed_head_origin_index_and_remote_fail(self):
        for arguments, output in (
            (("rev-parse", "--verify", "HEAD^{commit}"), b"short\n"),
            (("remote", "get-url", "--all", "origin"), b"one\ntwo\n"),
            (self.index_args, b"unexpected output"),
            (self.remote_args, b"short\trefs/heads/feature/preflight\n"),
            (self.remote_args, b"a" * 40 + b"\trefs/heads/other\n"),
            (self.remote_args, (f"{self.head}\trefs/heads/{self.branch}\n" * 2).encode()),
        ):
            with self.subTest(arguments=arguments, output=output):
                original = self.responses[arguments]
                self.responses[arguments] = (0, output)
                self.assertEqual(self.check().error_code, "MALFORMED_GIT_OUTPUT")
                self.responses[arguments] = original

    def test_output_capture_reads_only_fixed_limit_plus_one_and_closes(self):
        original_temporary_file = tempfile.TemporaryFile
        reads = []

        class SpyFile:
            def __init__(self, *args, **kwargs):
                self.file = original_temporary_file(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.file.close()

            def read(self, size):
                reads.append(size)
                return self.file.read(size)

            def __getattr__(self, name):
                return getattr(self.file, name)

        self.responses[("rev-parse", "--show-toplevel")] = (0, b"x" * (64 * 1024 + 1))
        with patch("auto_coding.git_publish.tempfile.TemporaryFile", SpyFile):
            self.assertEqual(self.check().error_code, "GIT_OUTPUT_TOO_LARGE")
        self.assertEqual(reads, [64 * 1024 + 1])

    def test_invalid_expected_state_and_timeouts_do_not_launch_git(self):
        for changes in (
            {"expected_branch": "--unsafe"}, {"expected_head": "short"},
            {"expected_origin_url": ""}, {"expected_origin_url": "url\nextra"},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.check(**changes).error_code, "INVALID_EXPECTED_STATE")
        for timeout in (0, -1, True, float("inf"), float("nan"), "3", 10 ** 1000):
            with self.subTest(timeout=timeout):
                self.assertEqual(self.check(timeout_seconds=timeout).error_code, "INVALID_TIMEOUT")
        self.assertEqual(self.calls, [])

    def test_invalid_repository_or_unsafe_temp_location_does_not_launch_git(self):
        self.assertEqual(self.check(repository_directory=self.parent / "missing").error_code,
                         "INVALID_REPOSITORY")
        with patch("auto_coding.git_publish._external_temporary_directory", return_value=None):
            self.assertEqual(self.check().error_code, "UNSAFE_TEMP_LOCATION")
        self.assertEqual(self.calls, [])

    def test_repository_environment_overrides_fail_closed(self):
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "private path"}):
                self.assertEqual(self.check().error_code, "UNSAFE_GIT_ENVIRONMENT")
        self.assertEqual(self.calls, [])

    @staticmethod
    def snapshot(repository):
        return {
            str(path.relative_to(repository)): path.read_bytes()
            for path in repository.rglob("*") if path.is_file()
        }

    def test_real_local_repository_and_remote_allow_dirty_worktree_without_writes(self):
        repository = self.parent / "actual"
        remote = self.parent / "remote.git"
        repository.mkdir()

        def git(*arguments, cwd=repository):
            with tempfile.TemporaryFile(mode="w+b") as output:
                subprocess.run(
                    ["git", *arguments], cwd=cwd, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.DEVNULL, shell=False, check=True, timeout=10,
                )
                output.seek(0)
                data = output.read(1025)
                self.assertLessEqual(len(data), 1024)
                return data

        git("init", "-q")
        git("config", "user.name", "Preflight Tests")
        git("config", "user.email", "preflight@example.invalid")
        (repository / "tracked.txt").write_bytes(b"original")
        git("add", "--", "tracked.txt")
        git("commit", "-qm", "initial")
        git("branch", "-M", self.branch)
        git("init", "--bare", "-q", str(remote))
        git("remote", "add", "origin", str(remote))
        git("push", "-q", "origin", self.branch)
        head = git("rev-parse", "HEAD").decode().strip()
        (repository / "tracked.txt").write_bytes(b"modified")
        (repository / "untracked.txt").write_bytes(b"untracked")
        before = self.snapshot(repository)
        result = check_publish_preflight(repository, self.branch, head, str(remote), 10)
        self.assertTrue(result.passed, result)
        self.assertEqual(self.snapshot(repository), before)


if __name__ == "__main__":
    unittest.main()
