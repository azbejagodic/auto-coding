"""Approved v2 execution preflight against disposable repositories only."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import auto_coding.git_publish as git_publish
import auto_coding.plan_preflight as plan_preflight
from auto_coding.plan_freeze import freeze_approved_plan, load_frozen_plan, plan_fingerprint
from auto_coding.plan_preflight import preflight_frozen_plan
from test_plans import make_plan, make_v2_plan


_STATUS = ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"]


class PlanPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.repository = self.parent / "target"
        self.remote = self.parent / "remote.git"
        self.frozen_path = self.parent / "approved.json"
        self.repository.mkdir()
        clean_env = {k: v for k, v in os.environ.items() if k not in git_publish._REPOSITORY_ENVIRONMENT}
        environment = patch.dict(os.environ, clean_env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.git("init", "-q")
        self.git("config", "user.name", "Plan Preflight Tests")
        self.git("config", "user.email", "preflight@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "commit.gpgsign", "false")
        (self.repository / "tracked.txt").write_bytes(b"original\n")
        (self.repository / ".gitignore").write_bytes(b"ignored.txt\n")
        self.git("add", "--", "tracked.txt", ".gitignore")
        self.git("commit", "-qm", "initial")
        self.branch = "feature/preflight"
        self.git("branch", "-M", self.branch)
        self.git("init", "--bare", "-q", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", self.branch)
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        self.plan = make_v2_plan(str(self.remote))
        self.plan.update(expected_branch=self.branch, expected_head=self.head)
        # Successful preflight must not execute this deliberately absent command.
        self.plan["tasks"][0]["test_commands"] = [["must-never-be-executed"], ["also-not-executed", "arg"]]
        self.approval = self.freeze(self.plan)

    def git(self, *arguments, cwd=None):
        with tempfile.TemporaryFile(mode="w+b") as output:
            subprocess.run(
                ["git", *arguments], cwd=self.repository if cwd is None else cwd,
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                shell=False, check=True, timeout=10,
            )
            output.seek(0)
            result = output.read(4097)
            self.assertLessEqual(len(result), 4096)
            return result

    def freeze(self, plan):
        self.frozen_path.unlink(missing_ok=True)
        fingerprint = plan_fingerprint(plan)
        freeze_approved_plan(plan, fingerprint, self.frozen_path)
        return fingerprint

    def check(self, **changes):
        arguments = dict(frozen_plan_path=self.frozen_path, approved_sha256=self.approval,
                         repository_directory=self.repository, git_timeout_seconds=5)
        return preflight_frozen_plan(**(arguments | changes))

    @staticmethod
    def snapshot(path):
        return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}

    def test_valid_approved_v2_returns_only_verified_identity_and_timeouts(self):
        result = self.check()
        self.assertTrue(result.ready, result)
        self.assertIsNone(result.error_code)
        self.assertEqual(result.batch_id, self.plan["batch_id"])
        self.assertEqual(result.plan_sha256, self.approval)
        self.assertEqual(result.branch, self.branch)
        self.assertEqual(result.starting_head, self.head)
        self.assertEqual(result.repository_directory, self.repository)
        self.assertEqual(result.origin_url, str(self.remote))
        self.assertEqual(result.test_timeouts, (("task-1", 120),))
        self.assertNotIn("prompt", repr(result))

    def test_per_task_timeout_is_frozen_and_applies_to_each_ordered_command(self):
        second = deepcopy(self.plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        second["limits"]["test_seconds"] = 35
        self.plan["tasks"].append(second)
        self.approval = self.freeze(self.plan)
        result = self.check()
        self.assertTrue(result.ready, result)
        self.assertEqual(result.test_timeouts, (("task-1", 120), ("task-2", 35)))
        frozen = load_frozen_plan(self.frozen_path, self.approval)
        for task_id, seconds in result.test_timeouts:
            task = next(t for t in frozen["tasks"] if t["task_id"] == task_id)
            self.assertEqual([seconds for _ in task["test_commands"]], [task["limits"]["test_seconds"]] * 2)

    def test_v1_remains_loadable_but_execution_is_rejected_without_git(self):
        legacy = make_plan()
        self.approval = self.freeze(legacy)
        before = self.frozen_path.read_bytes()
        with patch("auto_coding.plan_preflight.check_publish_preflight") as git:
            result = self.check()
            git.assert_not_called()
        self.assertEqual(result.error_code, "PLAN_SCHEMA_NOT_RUNNABLE")
        self.assertEqual(self.frozen_path.read_bytes(), before)
        self.assertEqual(load_frozen_plan(self.frozen_path, self.approval), legacy)

    def test_missing_malformed_or_wrong_approval_fails_before_git(self):
        for approval in (None, "", "bad", "0" * 64, "A" * 64):
            with self.subTest(approval=approval), patch("auto_coding.plan_preflight.check_publish_preflight") as git:
                result = self.check(approved_sha256=approval)
                self.assertEqual(result.error_code, "PLAN_INVALID_OR_UNAPPROVED")
                git.assert_not_called()
                self.assertIsNone(result.batch_id)

    def test_changed_timeout_or_origin_cannot_reuse_old_approval(self):
        for kind in ("timeout", "origin"):
            plan = deepcopy(self.plan)
            if kind == "timeout":
                plan["tasks"][0]["limits"]["test_seconds"] += 1
            else:
                plan["expected_origin_url"] = str(self.parent / "different.git")
            self.frozen_path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            with patch("auto_coding.plan_preflight.check_publish_preflight") as git:
                self.assertEqual(self.check().error_code, "PLAN_INVALID_OR_UNAPPROVED")
                git.assert_not_called()

    def test_corrupted_unknown_and_noncanonical_frozen_data_fail_closed(self):
        unknown = dict(self.plan, unexpected=True)
        invalid_timeout = deepcopy(self.plan)
        invalid_timeout["tasks"][0]["limits"]["test_seconds"] = False
        for raw in (b"", b"{", b"\xff", json.dumps(unknown).encode(), json.dumps(invalid_timeout).encode(),
                    json.dumps(self.plan, indent=2).encode()):
            self.frozen_path.write_bytes(raw)
            raw_digest = hashlib.sha256(raw).hexdigest()
            with patch("auto_coding.plan_preflight.check_publish_preflight") as git:
                result = self.check(approved_sha256=raw_digest)
                self.assertFalse(result.ready)
                git.assert_not_called()
            self.assertEqual(self.frozen_path.read_bytes(), raw)

    def test_canonical_fingerprint_is_recomputed_after_loading(self):
        with patch("auto_coding.plan_preflight.plan_fingerprint", return_value="0" * 64):
            self.assertEqual(self.check().error_code, "PLAN_FINGERPRINT_MISMATCH")

    def test_missing_or_inaccessible_frozen_file_is_sanitized(self):
        result = self.check(frozen_plan_path=self.parent / "missing.json")
        self.assertFalse(result.ready)
        with patch("auto_coding.plan_preflight.load_frozen_plan", side_effect=PermissionError("private")):
            result = self.check()
            self.assertEqual(result.error_code, "PLAN_IO_FAILED")
            self.assertNotIn("private", repr(result))

    def test_wrong_repository_root_branch_and_head_are_rejected(self):
        child = self.repository / "child"
        child.mkdir()
        self.assertEqual(self.check(repository_directory=child).error_code, "REPOSITORY_ROOT_MISMATCH")
        self.git("checkout", "-qb", "feature/wrong")
        self.assertEqual(self.check().error_code, "BRANCH_MISMATCH")
        self.git("checkout", "-q", self.branch)
        self.git("commit", "--allow-empty", "-qm", "new local commit")
        self.assertEqual(self.check().error_code, "HEAD_MISMATCH")

    def test_wrong_fetch_or_push_origin_is_rejected(self):
        other = self.parent / "other.git"
        self.git("remote", "set-url", "origin", str(other))
        self.assertEqual(self.check().error_code, "ORIGIN_MISMATCH")
        self.git("remote", "set-url", "origin", str(self.remote))
        self.git("remote", "set-url", "--push", "origin", str(other))
        self.assertEqual(self.check().error_code, "ORIGIN_MISMATCH")

    def test_changed_remote_sha_is_rejected(self):
        self.git("commit", "--allow-empty", "-qm", "remote-only commit")
        self.git("push", "-q", "origin", self.branch)
        # Restore only this disposable fixture's branch ref to the approved base.
        self.git("update-ref", f"refs/heads/{self.branch}", self.head)
        self.assertEqual(self.check().error_code, "REMOTE_HEAD_MISMATCH")

    def test_staged_unstaged_untracked_and_intent_to_add_are_not_clean(self):
        tracked = self.repository / "tracked.txt"
        tracked.write_bytes(b"modified\n")
        self.assertEqual(self.check().error_code, "WORKTREE_NOT_CLEAN")
        self.git("add", "--", "tracked.txt")
        self.assertEqual(self.check().error_code, "STAGED_CHANGES")
        # Use a fresh tracked file state without resetting Git history.
        tracked.write_bytes(b"original\n")
        self.git("add", "--", "tracked.txt")
        untracked = self.repository / "visible.txt"
        untracked.write_bytes(b"visible untracked\n")
        self.assertEqual(self.check().error_code, "WORKTREE_NOT_CLEAN")
        self.git("add", "-N", "--", "visible.txt")
        self.assertFalse(self.check().ready)

    def test_ignored_files_remain_ignored(self):
        (self.repository / "ignored.txt").write_bytes(b"ignored content")
        self.assertTrue(self.check().ready)

    def test_operation_markers_and_index_lock_are_preserved(self):
        for marker, expected in (("MERGE_HEAD", "GIT_OPERATION_IN_PROGRESS"),
                                 ("rebase-merge", "GIT_OPERATION_IN_PROGRESS"),
                                 ("index.lock", "INDEX_LOCK_PRESENT")):
            path = self.repository / ".git" / marker
            path.write_bytes(b"existing")
            self.assertEqual(self.check().error_code, expected)
            self.assertEqual(path.read_bytes(), b"existing")
            path.unlink()

    def test_git_launch_failure_timeout_nonzero_and_overflow_fail_closed(self):
        cases = (
            (FileNotFoundError("private"), "GIT_IO_FAILED"),
            (subprocess.TimeoutExpired("private", 5), "GIT_TIMEOUT"),
            ((7, b"uncontrolled output"), "GIT_FAILED"),
            ((0, b"x" * (64 * 1024 + 1)), "GIT_OUTPUT_TOO_LARGE"),
        )
        for response, expected in cases:
            with self.subTest(error=expected):
                def execute(command, **kwargs):
                    self.assertEqual(command[:2], ["git", "--no-optional-locks"])
                    self.assertNotEqual(kwargs["stdout"], subprocess.PIPE)
                    self.assertEqual(kwargs["timeout"], 5)
                    self.assertFalse(kwargs["shell"])
                    if isinstance(response, Exception):
                        raise response
                    code, output = response
                    kwargs["stdout"].write(output)
                    return subprocess.CompletedProcess(command, code)
                with patch("auto_coding.git_publish.subprocess.run", side_effect=execute):
                    result = self.check()
                self.assertEqual(result.error_code, expected)
                self.assertNotIn("private", repr(result))

    def test_status_launch_failure_timeout_overflow_and_malformed_output_fail_closed(self):
        for response, expected in (
            ((None, None, "GIT_IO_FAILED"), "GIT_IO_FAILED"),
            ((None, None, "GIT_TIMEOUT"), "GIT_TIMEOUT"),
            ((None, None, "GIT_OUTPUT_TOO_LARGE"), "GIT_OUTPUT_TOO_LARGE"),
            ((7, b"private", None), "GIT_FAILED"),
            ((0, b"malformed without NUL", None), "WORKTREE_NOT_CLEAN"),
        ):
            with self.subTest(expected=expected), patch("auto_coding.plan_preflight._git_query", return_value=response) as query:
                result = self.check()
                self.assertEqual(result.error_code, expected)
                self.assertEqual(query.call_args.args[1], _STATUS)

    def test_invalid_git_timeout_prevents_all_loading_and_git(self):
        for timeout in (True, None, 0, -1, float("nan"), float("inf"), 10**1000, "5"):
            with self.subTest(timeout_type=type(timeout)), patch("auto_coding.plan_preflight.load_frozen_plan") as load:
                self.assertEqual(self.check(git_timeout_seconds=timeout).error_code, "INVALID_TIMEOUT")
                load.assert_not_called()

    def test_no_mutation_and_only_read_only_git_commands_execute(self):
        before = self.snapshot(self.parent)
        original = git_publish._git_query
        calls = []

        def observe(repository, arguments, *args, **kwargs):
            self.assertIn(arguments[0], {"rev-parse", "symbolic-ref", "remote", "diff", "ls-remote", "status"})
            if arguments[0] == "remote":
                self.assertEqual(arguments[1], "get-url")
            calls.append(arguments)
            return original(repository, arguments, *args, **kwargs)

        with patch("auto_coding.git_publish._git_query", side_effect=observe), patch("auto_coding.plan_preflight._git_query", side_effect=observe):
            result = self.check()
        self.assertTrue(result.ready, result)
        self.assertIn(_STATUS, calls)
        self.assertEqual(self.snapshot(self.parent), before)

    def test_unsafe_temporary_directory_and_repository_environment_fail_closed(self):
        with patch("auto_coding.plan_preflight._external_temporary_directory", return_value=None):
            self.assertEqual(self.check().error_code, "UNSAFE_TEMP_LOCATION")
        with patch.dict(os.environ, {"GIT_DIR": "private"}):
            result = self.check()
            self.assertEqual(result.error_code, "UNSAFE_GIT_ENVIRONMENT")
            self.assertNotIn("private", repr(result))


if __name__ == "__main__":
    unittest.main()
