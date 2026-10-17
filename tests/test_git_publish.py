"""Tests for read-only Git publishing preflight."""

import os
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.change_evidence import collect_change_evidence
import auto_coding.git_publish as git_publish
from auto_coding.git_publish import (
    check_publish_preflight, publish_reviewed_changes, stage_reviewed_changes,
)
from auto_coding.review import ReviewResult, ReviewVerdict, build_review_packet
from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity


class GitPreflightTests(unittest.TestCase):
    def test_preparation_can_verify_an_explicit_source_ref_without_relaxing_default_gate(self):
        new_branch = "feature/continued"
        self.responses[("symbolic-ref", "--quiet", "HEAD")] = (0, f"refs/heads/{new_branch}\n".encode())
        self.responses[("ls-remote", "--exit-code", "--refs", "origin", f"refs/heads/{new_branch}")] = (2, b"")
        with patch("auto_coding.git_publish.subprocess.run", side_effect=self.execute):
            prepared = git_publish._check_publish_state(self.repository, new_branch, self.head, self.origin,
                                                       3, remote_branch=self.branch)
        self.assertTrue(prepared.passed, prepared)
        self.assertEqual(self.calls[-1], self.remote_args)
        # Normal preflight still requires the active NEW remote branch itself.
        self.assertEqual(self.check(expected_branch=new_branch).error_code, "GIT_FAILED")

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


class GitStagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.repository = self.parent / "repo"
        self.remote = self.parent / "remote.git"
        self.repository.mkdir()
        self.branch = "feature/staging"
        self.git("init", "-q")
        self.git("config", "user.name", "Staging Tests")
        self.git("config", "user.email", "staging@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.filemode", "false")
        (self.repository / "tracked.txt").write_bytes(b"original\n")
        (self.repository / "deleted.txt").write_bytes(b"delete me\n")
        self.git("add", "--", "tracked.txt", "deleted.txt")
        self.git("commit", "-qm", "initial")
        self.git("branch", "-M", self.branch)
        self.git("init", "--bare", "-q", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", self.branch)
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        self.permitted = ["tracked.txt", "deleted.txt", "new.bin", "-option.txt"]
        self.criteria = ["Exact changes reviewed"]
        self.command = ["test-tool", "--exact"]
        self.max_bytes = 100_000
        self.add_calls = []

    def git(self, *arguments):
        with tempfile.TemporaryFile(mode="w+b") as output:
            subprocess.run(
                ["git", *arguments], cwd=self.repository, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.DEVNULL, shell=False, check=True, timeout=10,
            )
            output.seek(0)
            data = output.read(100_001)
            self.assertLessEqual(len(data), 100_000)
            return data

    def approved_snapshot(self):
        evidence = collect_change_evidence(self.repository, self.permitted, self.max_bytes)
        self.assertTrue(evidence.passed, evidence)
        results = TestRunResult(True, 1, (
            TestCommandResult(1, 0, 0.25, False, "not_needed", _command_identity(self.command)),
        ), None)
        packet = build_review_packet(
            evidence, self.criteria, [self.command], results,
            [("tracked.txt", b"selected code")], self.max_bytes,
        )
        review = ReviewResult(
            True, ReviewVerdict("APPROVED", (), (), tuple(self.criteria), "HIGH"),
            None, 0, False, "not_needed", "a26947d0-1111-4444-8888-849398fad344",
            hashlib.sha256(packet).hexdigest(), evidence.fingerprint,
        )
        return packet, review

    def modify(self):
        (self.repository / "tracked.txt").write_bytes(b"reviewed\n")
        return self.approved_snapshot()

    def stage(self, packet, review, **changes):
        inputs = {
            "repository_directory": self.repository, "expected_branch": self.branch,
            "expected_head": self.head, "expected_origin_url": str(self.remote),
            "timeout_seconds": 10, "permitted_files": self.permitted,
            "reviewed_packet": packet, "review_result": review, "max_bytes": self.max_bytes,
        }
        return stage_reviewed_changes(**(inputs | changes))

    def intercept_add(self, action):
        original_run = subprocess.run

        def execute(command, **kwargs):
            # Only the implementation's literal add is intercepted. Fixture
            # mutations inside action use ordinary, explicitly scoped commands.
            if command[:4] == ["git", "--no-optional-locks", "--literal-pathspecs", "add"]:
                self.add_calls.append(command)
                result = original_run(command, **kwargs)
                return action(command, kwargs, result)
            self.assertNotIn("commit", command)
            self.assertNotIn("push", command)
            self.assertNotIn("reset", command)
            self.assertNotIn("restore", command)
            return original_run(command, **kwargs)

        return patch("auto_coding.git_publish.subprocess.run", side_effect=execute)

    def staged_paths(self):
        return tuple(sorted(self.git("diff", "--cached", "--name-only", "-z", self.head, "--")
                            .decode().split("\0")[:-1]))

    def test_exact_tracked_additions_deletions_and_binary_content_are_staged(self):
        (self.repository / "tracked.txt").write_bytes(b"reviewed\n")
        (self.repository / "deleted.txt").unlink()
        binary = b"\0\xff\x80new binary"
        (self.repository / "new.bin").write_bytes(binary)
        packet, review = self.approved_snapshot()
        with self.intercept_add(lambda command, kwargs, result: result):
            result = self.stage(packet, review)
        self.assertTrue(result.passed, result)
        self.assertTrue(result.staging_attempted)
        self.assertEqual(result.staged_paths, ("deleted.txt", "new.bin", "tracked.txt"))
        self.assertEqual(result.staged_paths, self.staged_paths())
        self.assertEqual(len(result.staged_changes_sha256), 64)
        self.assertEqual(self.git("show", ":new.bin"), binary)
        self.assertEqual(self.git("show", ":tracked.txt"), b"reviewed\n")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("ls-remote", "--refs", "origin", f"refs/heads/{self.branch}")
                         .decode().split("\t")[0], self.head)
        self.assertEqual([call[4:] for call in self.add_calls], [
            ["--all", "--", "deleted.txt"], ["--all", "--", "new.bin"],
            ["--all", "--", "tracked.txt"],
        ])
        self.assertNotIn("reviewed\n", repr(result))
        self.assertNotIn("new binary", repr(result))

    def test_leading_dash_filename_is_staged_literally(self):
        (self.repository / "-option.txt").write_bytes(b"ordinary addition")
        packet, review = self.approved_snapshot()
        with self.intercept_add(lambda command, kwargs, result: result):
            result = self.stage(packet, review)
        self.assertTrue(result.passed, result)
        self.assertEqual(self.staged_paths(), ("-option.txt",))
        self.assertEqual(self.add_calls[0][-2:], ["--", "-option.txt"])

    def test_missing_mismatched_and_unapproved_bindings_stop_before_add(self):
        packet, review = self.modify()
        for invalid in (
            replace(review, reviewed_packet_sha256=None),
            replace(review, reviewed_packet_sha256="0" * 64),
            replace(review, reviewed_evidence_fingerprint=None),
            replace(review, reviewed_evidence_fingerprint="0" * 64),
            replace(review, approved=False),
            replace(review, verdict=replace(review.verdict, decision="REJECTED")),
            replace(review, verdict=replace(review.verdict, decision="UNCERTAIN")),
            replace(review, verdict=replace(review.verdict, checked_requirements=())),
            replace(review, verdict=replace(review.verdict, blocking_issues=("blocker",))),
            replace(review, timed_out=True),
            replace(review, session_id="invalid"),
        ):
            with self.subTest(review=invalid):
                with self.intercept_add(lambda command, kwargs, result: result):
                    result = self.stage(packet, invalid)
                self.assertFalse(result.passed)
                self.assertFalse(result.staging_attempted)
                self.assertEqual(self.staged_paths(), ())
        self.assertEqual(self.add_calls, [])

    def test_packet_content_mismatch_and_invalid_packet_fail_before_add(self):
        packet, review = self.modify()
        for invalid in (packet + b" ", b"{invalid", b"", b"x" * (self.max_bytes + 1)):
            with self.subTest(size=len(invalid)):
                result = self.stage(invalid, review)
                self.assertFalse(result.passed)
                self.assertFalse(result.staging_attempted)
        self.assertEqual(self.staged_paths(), ())

    def test_current_evidence_change_after_review_stops_before_add(self):
        packet, review = self.modify()
        (self.repository / "tracked.txt").write_bytes(b"not reviewed\n")
        result = self.stage(packet, review)
        self.assertEqual(result.error_code, "EVIDENCE_CHANGED")
        self.assertFalse(result.staging_attempted)
        self.assertEqual(self.staged_paths(), ())

    def test_out_of_scope_and_wildcard_like_paths_cannot_be_staged(self):
        packet, review = self.modify()
        for filename in ("outside.txt", "literal[1].txt"):
            with self.subTest(filename=filename):
                path = self.repository / filename
                path.write_bytes(b"outside")
                result = self.stage(packet, review)
                self.assertEqual(result.error_code, "OUT_OF_SCOPE")
                self.assertFalse(result.staging_attempted)
                path.unlink()
        result = self.stage(packet, review, permitted_files=["*.txt"])
        self.assertFalse(result.passed)
        self.assertFalse(result.staging_attempted)
        self.assertEqual(self.staged_paths(), ())

    def test_existing_index_changes_and_intent_to_add_are_preserved(self):
        packet, review = self.modify()
        self.git("add", "--", "tracked.txt")
        before = self.git("ls-files", "--stage", "-z")
        result = self.stage(packet, review)
        self.assertEqual(result.error_code, "STAGED_CHANGES")
        self.assertEqual(self.git("ls-files", "--stage", "-z"), before)
        # Reset is fixture setup only; the staging implementation never resets.
        self.git("reset", "-q", "--", "tracked.txt")
        (self.repository / "new.bin").write_bytes(b"intent")
        self.git("add", "-N", "--", "new.bin")
        before = self.git("ls-files", "--stage", "-z")
        result = self.stage(packet, review)
        self.assertIn(result.error_code, ("STAGED_CHANGES", "PREEXISTING_INDEX_CHANGES"))
        self.assertFalse(result.staging_attempted)
        self.assertEqual(self.git("ls-files", "--stage", "-z"), before)

    def test_unexpected_staged_path_is_reported_and_not_reset(self):
        packet, review = self.modify()
        original_run = subprocess.run

        def inject(command, kwargs, result):
            (self.repository / "outside.txt").write_bytes(b"unexpected")
            original_run(["git", "add", "--", "outside.txt"], cwd=self.repository,
                         check=True, timeout=10, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
            return result

        with self.intercept_add(inject):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "STAGED_PATHS_MISMATCH")
        self.assertTrue(result.staging_attempted)
        self.assertEqual(result.staged_paths, ("outside.txt", "tracked.txt"))
        self.assertEqual(self.staged_paths(), result.staged_paths)

    def test_unexpected_staged_tracked_and_binary_content_cannot_pass(self):
        for path in ("tracked.txt", "new.bin"):
            with self.subTest(path=path):
                if path == "tracked.txt":
                    packet, review = self.modify()
                else:
                    self.git("reset", "-q", "--", "tracked.txt")
                    (self.repository / "tracked.txt").write_bytes(b"original\n")
                    (self.repository / "new.bin").write_bytes(b"\0\xffreviewed")
                    packet, review = self.approved_snapshot()
                original = (self.repository / path).read_bytes()
                original_run = subprocess.run

                def inject(command, kwargs, result):
                    (self.repository / path).write_bytes(b"unreviewed content")
                    original_run(["git", "add", "--", path], cwd=self.repository,
                                 check=True, timeout=10, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                    (self.repository / path).write_bytes(original)
                    return result

                with self.intercept_add(inject):
                    result = self.stage(packet, review)
                self.assertEqual(result.error_code, "STAGED_CONTENT_MISMATCH")
                self.assertTrue(result.staging_attempted)
                self.assertEqual(self.git("show", f":{path}"), b"unreviewed content")

    def test_partial_add_failure_or_timeout_reports_index_without_retry(self):
        packet, review = self.modify()

        def timeout(command, kwargs, result):
            raise subprocess.TimeoutExpired("private", 10, output=b"private")

        with self.intercept_add(timeout):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertTrue(result.staging_attempted)
        self.assertEqual(result.staged_paths, ("tracked.txt",))
        self.assertEqual(result.timeout.operation, "git_staging")
        self.assertEqual(result.timeout.configured_seconds, 10)
        self.assertEqual(self.staged_paths(), result.staged_paths)
        self.assertEqual(len(self.add_calls), 1)
        self.assertNotIn("private", repr(result))

    def test_nonzero_add_and_ambiguous_add_output_preserve_staged_state(self):
        packet, review = self.modify()

        def fail(command, kwargs, result):
            return subprocess.CompletedProcess(command, 1)

        with self.intercept_add(fail):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "GIT_FAILED")
        self.assertEqual(result.staged_paths, ("tracked.txt",))
        self.git("reset", "-q", "--", "tracked.txt")

        def ambiguous(command, kwargs, result):
            kwargs["stdout"].write(b"private output")
            return result

        with self.intercept_add(ambiguous):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "AMBIGUOUS_STAGING_OUTPUT")
        self.assertEqual(result.staged_paths, ("tracked.txt",))
        self.assertNotIn("private output", repr(result))

    def test_remote_state_change_during_staging_fails_and_preserves_index(self):
        packet, review = self.modify()
        original_run = subprocess.run
        staged = False

        def execute(command, **kwargs):
            nonlocal staged
            if command[:4] == ["git", "--no-optional-locks", "--literal-pathspecs", "add"]:
                staged = True
            if staged and command[2:3] == ["ls-remote"]:
                kwargs["stdout"].write(b"b" * 40 + f"\trefs/heads/{self.branch}\n".encode())
                return subprocess.CompletedProcess(command, 0)
            return original_run(command, **kwargs)

        with patch("auto_coding.git_publish.subprocess.run", side_effect=execute):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "REMOTE_HEAD_MISMATCH")
        self.assertEqual(result.staged_paths, ("tracked.txt",))
        self.assertEqual(self.staged_paths(), result.staged_paths)

    def test_failed_preflight_is_reused_and_prevents_staging(self):
        packet, review = self.modify()
        failure = git_publish.GitPreflightResult(False, "HEAD_MISMATCH")
        with patch("auto_coding.git_publish.check_publish_preflight",
                   return_value=failure) as preflight:
            result = self.stage(packet, review)
        preflight.assert_called_once_with(
            self.repository, self.branch, self.head, str(self.remote), 10,
        )
        self.assertEqual(result.error_code, "HEAD_MISMATCH")
        self.assertFalse(result.staging_attempted)
        self.assertEqual(self.staged_paths(), ())

    def test_file_replaced_by_directory_before_add_cannot_expand_scope(self):
        (self.repository / "deleted.txt").unlink()
        packet, review = self.approved_snapshot()
        original_location = git_publish._safe_location
        checks = 0

        def location(repository, path):
            nonlocal checks
            if path == "deleted.txt":
                checks += 1
                if checks == 2:  # The check immediately before the add.
                    directory = self.repository / path
                    directory.mkdir()
                    (directory / "outside.txt").write_bytes(b"not authorized")
            return original_location(repository, path)

        with patch("auto_coding.git_publish._safe_location", side_effect=location):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "UNSAFE_PATH")
        self.assertFalse(result.staging_attempted)
        self.assertEqual(self.staged_paths(), ())

    def test_unverified_index_mode_and_post_add_intent_to_add_fail_closed(self):
        packet, review = self.modify()
        original_query = git_publish._git_query
        for failure in ("mode", "intent"):
            with self.subTest(failure=failure):
                staged = False

                def query(repository, arguments, *rest):
                    nonlocal staged
                    code, output, error = original_query(repository, arguments, *rest)
                    if arguments[:2] == ["--literal-pathspecs", "add"]:
                        staged = True
                    if staged and "--raw" in arguments and error is None:
                        if failure == "mode":
                            output = output.replace(b":100644 100644 ", b":100644 120000 ", 1)
                        elif "--ita-invisible-in-index" in arguments:
                            output = b""
                    return code, output, error

                with patch("auto_coding.git_publish._git_query", side_effect=query):
                    result = self.stage(packet, review)
                self.assertEqual(result.error_code,
                                 "UNVERIFIED_FILE_MODE" if failure == "mode" else "INTENT_TO_ADD")
                self.assertTrue(result.staging_attempted)
                self.assertEqual(self.staged_paths(), ("tracked.txt",))
                self.git("reset", "-q", "--", "tracked.txt")

    def test_ambiguous_partial_staging_with_unreadable_index_reports_unknown_state(self):
        packet, review = self.modify()
        original_query = git_publish._git_query
        staged = False
        add_count = 0

        def query(repository, arguments, *rest):
            nonlocal staged, add_count
            if staged and "--raw" in arguments:
                return None, None, "GIT_TIMEOUT"
            result = original_query(repository, arguments, *rest)
            if arguments[:2] == ["--literal-pathspecs", "add"]:
                staged = True
                add_count += 1
                return None, None, "GIT_TIMEOUT"
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.stage(packet, review)
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertTrue(result.staging_attempted)
        self.assertIsNone(result.staged_paths)
        self.assertIsNone(result.staged_changes_sha256)
        self.assertEqual(result.staged_state_error_code, "GIT_TIMEOUT")
        self.assertEqual(add_count, 1)
        self.assertEqual(self.staged_paths(), ("tracked.txt",))


class GitPublishingTests(unittest.TestCase):
    # Reuse the disposable local repository fixture without inheriting and
    # rerunning the staging test methods.
    git = GitStagingTests.git
    approved_snapshot = GitStagingTests.approved_snapshot
    modify = GitStagingTests.modify
    stage = GitStagingTests.stage

    def setUp(self):
        GitStagingTests.setUp(self)
        self.packet, self.review = self.modify()
        self.staged = self.stage(self.packet, self.review)
        self.assertTrue(self.staged.passed, self.staged)
        self.message = "feat: reviewed fixture change"
        self.commands = []

    def publish(self, **changes):
        inputs = {
            "repository_directory": self.repository, "expected_branch": self.branch,
            "expected_head": self.head, "expected_origin_url": str(self.remote),
            "timeout_seconds": 10, "staged_result": self.staged,
            "reviewed_packet": self.packet, "review_result": self.review,
            "max_bytes": self.max_bytes, "commit_message": self.message,
        }
        original_query = git_publish._git_query

        def observe(repository, arguments, *rest):
            self.commands.append(arguments)
            self.assertNotIn(arguments[0], ("add", "reset", "restore", "merge"))
            self.assertNotIn("--amend", arguments)
            self.assertNotIn("--force", arguments)
            self.assertNotIn("--force-with-lease", arguments)
            return original_query(repository, arguments, *rest)

        with patch("auto_coding.git_publish._git_query", side_effect=observe):
            return publish_reviewed_changes(**(inputs | changes))

    def remote_head(self):
        return self.git("ls-remote", "--refs", "origin", f"refs/heads/{self.branch}") \
            .decode().split("\t")[0]

    def assert_no_commit(self, result):
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "NOT_ATTEMPTED")
        self.assertFalse(result.commit_attempted)
        self.assertFalse(result.push_attempted)
        self.assertIsNone(result.commit_sha)
        self.assertIsNone(result.remote_sha)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertFalse(any(command[0] in ("commit", "push") for command in self.commands))

    def test_valid_staging_commits_and_pushes_exact_verified_content_once(self):
        tree = self.git("write-tree").decode().strip()
        # Explicit command flags must prevent follow-tags configuration from
        # pushing additional refs. Mirror configuration is rejected separately.
        self.git("config", "remote.origin.mirror", "false")
        self.git("config", "push.followTags", "true")
        self.git("tag", "-a", "not-approved-for-push", "-m", "fixture tag")
        result = self.publish()
        self.assertTrue(result.passed, result)
        self.assertEqual(result.outcome, "PUBLISHED")
        self.assertTrue(result.commit_attempted)
        self.assertTrue(result.push_attempted)
        self.assertEqual(result.commit_sha, self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(result.observed_head_sha, result.commit_sha)
        self.assertEqual(result.remote_sha, result.commit_sha)
        self.assertEqual(self.remote_head(), result.commit_sha)
        self.assertEqual(self.git("rev-parse", "HEAD^").decode().strip(), self.head)
        self.assertEqual(self.git("rev-parse", "HEAD^{tree}").decode().strip(), tree)
        self.assertEqual(self.git("rev-list", "--count", "HEAD").strip(), b"2")
        self.assertEqual(self.git("status", "--porcelain=v1", "-z", "--untracked-files=all"), b"")
        self.assertEqual(self.git("ls-remote", "--tags", "origin"), b"")
        self.assertEqual([command for command in self.commands if command[0] == "commit"],
                         [["commit", "--cleanup=verbatim", "-m", self.message]])
        self.assertEqual([command for command in self.commands if command[0] == "push"], [[
            "push", "--porcelain", "--no-force", "--no-mirror", "--no-follow-tags",
            "--recurse-submodules=no", "origin", f"{result.commit_sha}:refs/heads/{self.branch}",
        ]])
        self.assertNotIn("reviewed\n", repr(result))
        self.assertNotIn(self.message, repr(result))

    def test_additions_binary_and_deletions_publish_correctly(self):
        # Restore only the fixture's initial staged change, then make a new
        # reviewed fixture snapshot covering all supported change types.
        self.git("reset", "-q", "--", "tracked.txt")
        (self.repository / "deleted.txt").unlink()
        binary = b"\0\xff\x80reviewed binary"
        (self.repository / "new.bin").write_bytes(binary)
        self.packet, self.review = self.approved_snapshot()
        self.staged = self.stage(self.packet, self.review)
        self.assertTrue(self.staged.passed, self.staged)
        result = self.publish()
        self.assertTrue(result.passed, result)
        self.assertEqual(self.git("show", "HEAD:new.bin"), binary)
        self.assertEqual(self.git("show", "HEAD:tracked.txt"), b"reviewed\n")
        self.assertEqual(self.git("ls-tree", "--name-only", "HEAD", "--", "deleted.txt"), b"")
        self.assertEqual(self.remote_head(), result.commit_sha)

    def test_missing_approval_bindings_and_failed_stage_results_prevent_commit(self):
        for changes in (
            {"review_result": None},
            {"review_result": replace(self.review, approved=False)},
            {"review_result": replace(self.review, reviewed_packet_sha256=None)},
            {"review_result": replace(self.review, reviewed_packet_sha256="0" * 64)},
            {"review_result": replace(self.review, reviewed_evidence_fingerprint="0" * 64)},
            {"review_result": replace(self.review, verdict=replace(
                self.review.verdict, checked_requirements=(),
            ))},
            {"staged_result": replace(self.staged, passed=False)},
            {"staged_result": replace(self.staged, staged_paths=("outside.txt",))},
            {"staged_result": replace(self.staged, staged_changes_sha256="0" * 64)},
        ):
            with self.subTest(field=tuple(changes)):
                self.assert_no_commit(self.publish(**changes))

    def test_invalid_commit_messages_and_protected_main_prevent_mutation(self):
        for message in (None, "", " ", "bad\0message", "bad\rmessage", "x" * 4097):
            with self.subTest(message=message):
                self.assert_no_commit(self.publish(commit_message=message))
        self.assert_no_commit(self.publish(expected_branch="main"))

    def test_mirror_configuration_is_rejected_before_commit(self):
        self.git("config", "remote.origin.mirror", "true")
        result = self.publish()
        self.assert_no_commit(result)
        self.assertEqual(result.error_code, "UNSAFE_PUSH_CONFIGURATION")

    def test_modified_index_tracked_worktree_and_untracked_files_prevent_commit(self):
        (self.repository / "tracked.txt").write_bytes(b"changed index\n")
        self.git("add", "--", "tracked.txt")
        self.assert_no_commit(self.publish())
        (self.repository / "tracked.txt").write_bytes(b"reviewed\n")
        self.git("add", "--", "tracked.txt")
        (self.repository / "tracked.txt").write_bytes(b"unstaged modification\n")
        self.assert_no_commit(self.publish())
        (self.repository / "tracked.txt").write_bytes(b"reviewed\n")
        (self.repository / "outside.txt").write_bytes(b"untracked addition")
        self.assert_no_commit(self.publish())

    def test_wrong_branch_head_origin_or_remote_prevents_commit(self):
        for changes in (
            {"expected_branch": "other"}, {"expected_head": "b" * 40},
            {"expected_origin_url": "https://other.invalid/private"},
            {"timeout_seconds": 0},
        ):
            with self.subTest(changes=changes):
                self.assert_no_commit(self.publish(**changes))
        tree = self.git("rev-parse", "HEAD^{tree}").decode().strip()
        other = self.git("commit-tree", tree, "-p", self.head, "-m", "remote fixture advance") \
            .decode().strip()
        self.git("push", "-q", "origin", f"{other}:refs/heads/{self.branch}")
        result = self.publish()
        self.assert_no_commit(result)
        self.assertEqual(result.error_code, "REMOTE_HEAD_MISMATCH")

    def test_commit_failure_preserves_index_and_does_not_push(self):
        before = self.git("ls-files", "--stage", "-z")
        original = git_publish._git_query
        commits = []

        def query(repository, arguments, *rest):
            if arguments[0] == "commit":
                commits.append(arguments)
                return 1, b"private output", None
            return original(repository, arguments, *rest)

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_FAILED")
        self.assertTrue(result.commit_attempted)
        self.assertFalse(result.push_attempted)
        self.assertIsNone(result.commit_sha)
        self.assertEqual(len(commits), 1)
        self.assertEqual(self.git("ls-files", "--stage", "-z"), before)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertNotIn("private output", repr(result))

    def test_commit_timeout_after_creation_is_ambiguous_and_never_retried(self):
        original = git_publish._git_query
        commits = 0

        def query(repository, arguments, *rest):
            nonlocal commits
            result = original(repository, arguments, *rest)
            if arguments[0] == "commit":
                commits += 1
                return None, None, "GIT_TIMEOUT"
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(result.timeout.operation, "git_commit")
        self.assertEqual(result.timeout.configured_seconds, 10)
        self.assertIsNone(result.timeout.elapsed_seconds)
        self.assertIsNone(result.commit_sha)
        self.assertFalse(result.push_attempted)
        self.assertEqual(commits, 1)
        self.assertNotEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.remote_head(), self.head)

    def test_hook_changed_commit_tree_is_not_pushed(self):
        original = git_publish._git_query

        def query(repository, arguments, *rest):
            if arguments[0] == "commit":
                # Deterministically simulate a pre-commit hook staging
                # unreviewed content before Git creates its commit.
                (self.repository / "tracked.txt").write_bytes(b"hook changed private code")
                self.git("add", "--", "tracked.txt")
            return original(repository, arguments, *rest)

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertEqual(result.error_code, "COMMIT_TREE_MISMATCH")
        self.assertFalse(result.push_attempted)
        self.assertIsNone(result.commit_sha)
        self.assertEqual(result.observed_head_sha, self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(self.remote_head(), self.head)
        self.assertNotIn("private code", repr(result))

    def test_wrong_commit_parent_is_not_pushed(self):
        original = git_publish._git_query

        def query(repository, arguments, *rest):
            code, output, error = original(repository, arguments, *rest)
            if arguments[:3] == ["--no-replace-objects", "cat-file", "commit"] and error is None:
                output = output.replace(b"parent " + self.head.encode(), b"parent " + b"b" * 40)
            return code, output, error

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertEqual(result.error_code, "COMMIT_PARENT_MISMATCH")
        self.assertFalse(result.push_attempted)
        self.assertEqual(self.remote_head(), self.head)

    def test_pending_changes_created_by_hook_prevent_push(self):
        original = git_publish._git_query

        def query(repository, arguments, *rest):
            result = original(repository, arguments, *rest)
            if arguments[0] == "commit":
                (self.repository / "outside.txt").write_bytes(b"hook side effect")
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertEqual(result.error_code, "UNEXPECTED_PENDING_CHANGES")
        self.assertIsNotNone(result.commit_sha)
        self.assertFalse(result.push_attempted)
        self.assertEqual(self.remote_head(), self.head)

    def test_branch_changed_by_hook_prevents_push(self):
        original = git_publish._git_query

        def query(repository, arguments, *rest):
            result = original(repository, arguments, *rest)
            if arguments[0] == "commit":
                self.git("branch", "after-hook")
                self.git("symbolic-ref", "HEAD", "refs/heads/after-hook")
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertEqual(result.error_code, "BRANCH_MISMATCH")
        self.assertIsNotNone(result.commit_sha)
        self.assertFalse(result.push_attempted)
        self.assertEqual(self.remote_head(), self.head)

    def test_remote_change_before_push_preserves_commit_without_publishing(self):
        original = git_publish._git_query
        committed = False

        def query(repository, arguments, *rest):
            nonlocal committed
            if committed and arguments[0] == "ls-remote":
                return 0, f"{'b' * 40}\trefs/heads/{self.branch}\n".encode(), None
            result = original(repository, arguments, *rest)
            if arguments[0] == "commit":
                committed = True
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertEqual(result.error_code, "REMOTE_HEAD_MISMATCH")
        self.assertIsNotNone(result.commit_sha)
        self.assertFalse(result.push_attempted)
        self.assertFalse(any(command[0] == "push" for command in self.commands))

    def test_push_failure_preserves_verified_local_commit(self):
        original = git_publish._git_query
        pushes = 0

        def query(repository, arguments, *rest):
            nonlocal pushes
            if arguments[0] == "push":
                pushes += 1
                return 1, b"private output", None
            return original(repository, arguments, *rest)

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_FAILED")
        self.assertTrue(result.push_attempted)
        self.assertEqual(result.commit_sha, self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(self.remote_head(), self.head)
        self.assertIsNone(result.remote_sha)
        self.assertEqual(pushes, 1)
        self.assertNotIn("private output", repr(result))

    def test_push_timeout_after_remote_update_stops_without_retry(self):
        original = git_publish._git_query
        pushes = 0

        def query(repository, arguments, *rest):
            nonlocal pushes
            result = original(repository, arguments, *rest)
            if arguments[0] == "push":
                pushes += 1
                return None, None, "GIT_TIMEOUT"
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(result.timeout.operation, "git_push")
        self.assertIsNone(result.remote_sha)
        self.assertEqual(self.remote_head(), result.commit_sha)
        self.assertEqual(pushes, 1)

    def test_remote_verification_timeout_is_distinct_from_push_timeout(self):
        original = git_publish._git_query
        pushes = []
        def query(repository, arguments, *rest):
            if pushes and arguments[0] == "ls-remote":
                return None, None, "GIT_TIMEOUT"
            result = original(repository, arguments, *rest)
            if arguments[0] == "push":
                pushes.append(arguments)
            return result
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(result.timeout.operation, "git_verification")
        self.assertEqual(len(pushes), 1)
        self.assertIsNone(result.remote_sha)
        self.assertEqual(self.remote_head(), result.commit_sha)

    def test_successful_push_requires_independent_matching_remote_sha(self):
        original = git_publish._git_query
        pushed = False

        def query(repository, arguments, *rest):
            nonlocal pushed
            if pushed and arguments[0] == "ls-remote":
                return 0, f"{'b' * 40}\trefs/heads/{self.branch}\n".encode(), None
            result = original(repository, arguments, *rest)
            if arguments[0] == "push":
                pushed = True
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "REMOTE_VERIFICATION_MISMATCH")
        self.assertFalse(result.passed)
        self.assertEqual(result.remote_sha, "b" * 40)

    def test_unavailable_remote_verification_is_ambiguous_without_retry(self):
        original = git_publish._git_query
        pushed = False
        verifications = 0

        def query(repository, arguments, *rest):
            nonlocal pushed, verifications
            if pushed and arguments[0] == "ls-remote":
                verifications += 1
                return None, None, "GIT_TIMEOUT"
            result = original(repository, arguments, *rest)
            if arguments[0] == "push":
                pushed = True
            return result

        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertIsNone(result.remote_sha)
        self.assertEqual(verifications, 1)
        self.assertEqual(self.remote_head(), result.commit_sha)


if __name__ == "__main__":
    unittest.main()
