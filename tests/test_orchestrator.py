"""First-task integration with fake Codex/test execution and local Git only."""

from copy import deepcopy
from dataclasses import replace
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
import auto_coding.orchestrator as orchestrator
from auto_coding.batch_state import load_checkpoint, update_checkpoint
from auto_coding.integrations.codex import ImplementationOutcome, ImplementationReport
from auto_coding.plan_freeze import freeze_approved_plan, plan_fingerprint
from auto_coding.review import ReviewResult, ReviewVerdict, build_review_packet
from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity
from test_plans import make_v2_plan


_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="First Task ")
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name).resolve()
        self.repository = self.parent / "target"
        self.remote = self.parent / "remote.git"
        self.frozen_path = self.parent / "frozen.json"
        self.checkpoint_path = self.parent / "checkpoint.json"
        self.repository.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE")}
        environment = patch.dict(os.environ, env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.git("init", "-q")
        self.git("config", "user.name", "Orchestration Tests")
        self.git("config", "user.email", "orchestration@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "commit.gpgsign", "false")
        (self.repository / "app.txt").write_bytes(b"original\n")
        (self.repository / "deleted.txt").write_bytes(b"original deleted context\n")
        self.git("add", "--", "app.txt", "deleted.txt")
        self.git("commit", "-qm", "initial")
        self.branch = "feature/first-task"
        self.git("branch", "-M", self.branch)
        self.git("init", "--bare", "-q", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", self.branch)
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        self.plan = make_v2_plan(str(self.remote))
        self.plan.update(expected_branch=self.branch, expected_head=self.head)
        task = self.plan["tasks"][0]
        task.update(permitted_files=["app.txt", "deleted.txt", "new.bin"],
                    test_commands=[["frozen-test", "--literal", "PRIVATE_ARG"], ["second-test", "arg"]],
                    acceptance_criteria=["Behavior works", "Binary fidelity"])
        task["implementation"] = {"prompt": "PRIVATE_LITERAL_IMPLEMENTATION\nDo the task.", "model": "gpt-6-sol", "reasoning_effort": "medium"}
        task["review"] = {"prompt": "PRIVATE_LITERAL_REVIEW\nInspect every criterion \u2603.", "model": "gpt-6.1-sol", "reasoning_effort": "high"}
        task["limits"] = {"implementation_seconds": 11, "review_seconds": 12, "test_seconds": 13}
        second = deepcopy(task)
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        second["implementation"]["prompt"] = "NEVER_RUN_SECOND_TASK"
        self.plan["tasks"].append(second)
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        self.a_calls, self.test_calls, self.b_calls = [], [], []
        self.a_result = ImplementationOutcome(0, 1.25, False, True, "not_needed", _A, None,
                                               ImplementationReport("PRIVATE_MODEL_SUMMARY", "Small change", ("wrong-self-report.txt",), "HIGH", ()))

    def git(self, *arguments):
        with tempfile.TemporaryFile(mode="w+b") as output:
            subprocess.run(["git", *arguments], cwd=self.repository, stdin=subprocess.DEVNULL,
                           stdout=output, stderr=subprocess.DEVNULL, shell=False, check=True, timeout=10)
            output.seek(0)
            result = output.read(4097)
            self.assertLessEqual(len(result), 4096)
            return result

    def implement(self, *args, **kwargs):
        self.a_calls.append((args, kwargs))
        (self.repository / "app.txt").write_bytes(b"implemented\n")
        (self.repository / "new.bin").write_bytes(b"\0\xffPRIVATE_BINARY\x80")
        return self.a_result

    def run_tests_fake(self, *args, **kwargs):
        self.test_calls.append((args, kwargs))
        commands = args[1]
        return TestRunResult(True, len(commands), tuple(
            TestCommandResult(i, 0, 0.25, False, "not_needed", _command_identity(command))
            for i, command in enumerate(commands, 1)
        ), None)

    def review(self, *args, **kwargs):
        self.b_calls.append((args, kwargs))
        packet = build_review_packet(*args)
        return ReviewResult(True, ReviewVerdict("APPROVED", (), (), tuple(args[1]), "HIGH"), None,
                            0, False, "not_needed", _B, hashlib.sha256(packet).hexdigest(), args[0].fingerprint)

    def execute(self, implement=None, tests=None, review=None, **changes):
        inputs = dict(frozen_plan_path=self.frozen_path, approved_sha256=self.approval,
                      repository_directory=self.repository, checkpoint_path=self.checkpoint_path,
                      git_timeout_seconds=5, evidence_max_bytes=100_000, packet_max_bytes=200_000)
        with patch("auto_coding.orchestrator.run_implementation", side_effect=implement or self.implement), \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=tests or self.run_tests_fake), \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=review or self.review):
            return orchestrator.execute_first_task_through_review(**(inputs | changes))

    def assert_stopped(self, result, code=None):
        self.assertFalse(result.ready_for_publication)
        self.assertNotEqual(result.outcome, "READY_FOR_PUBLICATION")
        if code is not None:
            self.assertEqual(result.error_code, code)
        if self.checkpoint_path.exists():
            self.assertNotEqual(load_checkpoint(self.checkpoint_path)["status"], "completed")

    def test_valid_first_task_has_exact_frozen_settings_and_publication_handoff(self):
        result = self.execute()
        self.assertTrue(result.ready_for_publication, result)
        self.assertEqual(result.task_id, "task-1")
        self.assertEqual(result.plan_sha256, self.approval)
        self.assertEqual(result.implementer_session_id, _A)
        self.assertEqual(result.reviewer_session_id, _B)
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(len(self.test_calls), 1)
        self.assertEqual(len(self.b_calls), 1)
        task = self.plan["tasks"][0]
        self.assertEqual(self.a_calls[0], ((task["implementation"]["prompt"], "gpt-6-sol", "medium", self.repository, 11), {"session_id": None}))
        self.assertEqual(self.test_calls[0], ((self.repository, task["test_commands"], 13), {}))
        args, kwargs = self.b_calls[0]
        self.assertEqual(kwargs, {"model": "gpt-6.1-sol", "reasoning_effort": "high", "repository_directory": self.repository,
                                  "timeout_seconds": 12, "session_id": None, "review_prompt": task["review"]["prompt"]})
        self.assertEqual(args[1], task["acceptance_criteria"])
        self.assertEqual(args[2], task["test_commands"])
        self.assertEqual(result.reviewed_packet, build_review_packet(*args))
        self.assertEqual(result.review.reviewed_packet_sha256, hashlib.sha256(result.reviewed_packet).hexdigest())
        self.assertEqual(result.review.reviewed_evidence_fingerprint, result.evidence.fingerprint)
        state = load_checkpoint(self.checkpoint_path)
        self.assertEqual((state["status"], state["stage"], state["current_task_id"]), ("running", "publishing", "task-1"))
        self.assertEqual((state["implementer_session_id"], state["reviewer_session_id"]), (_A, _B))
        self.assertIsNone(state["stop_reason_code"])
        self.assertEqual(result.implementation.report.files_reported, ("wrong-self-report.txt",))
        self.assertEqual(result.evidence.changed_paths, ("app.txt", "new.bin"))
        for private in ("PRIVATE_LITERAL", "PRIVATE_BINARY", "PRIVATE_MODEL_SUMMARY", "tracked_diff_base64", "PRIVATE_ARG"):
            self.assertNotIn(private, repr(result))

    def test_invalid_approval_or_dirty_start_stops_before_checkpoint_and_a(self):
        result = self.execute(approved_sha256="0" * 64)
        self.assert_stopped(result)
        self.assertFalse(self.checkpoint_path.exists())
        (self.repository / "outside.txt").write_bytes(b"dirty start")
        result = self.execute()
        self.assert_stopped(result, "WORKTREE_NOT_CLEAN")
        self.assertEqual(result.outcome, "STOPPED_BEFORE_IMPLEMENTATION")
        self.assertFalse(self.checkpoint_path.exists())
        self.assertEqual(self.a_calls, [])

    def test_external_checkpoint_and_existing_checkpoint_are_enforced(self):
        result = self.execute(checkpoint_path=self.repository / "checkpoint.json")
        self.assert_stopped(result, "UNSAFE_CHECKPOINT_LOCATION")
        self.checkpoint_path.write_bytes(b"existing checkpoint")
        result = self.execute()
        self.assertFalse(result.ready_for_publication)
        self.assertEqual(result.error_code, "CHECKPOINT_EXISTS")
        self.assertEqual(self.checkpoint_path.read_bytes(), b"existing checkpoint")
        self.assertEqual(self.a_calls, [])

    def test_invalid_limits_stop_before_any_adapter(self):
        for limits in ({"evidence_max_bytes": 0}, {"packet_max_bytes": True}, {"git_timeout_seconds": 0}):
            self.assert_stopped(self.execute(**limits))
        self.assertFalse(self.checkpoint_path.exists())
        self.assertEqual(self.a_calls, [])

    def test_invalid_a_results_stop_before_tests_and_b(self):
        variants = (None, replace(self.a_result, succeeded=False, error_code="PROCESS_TIMEOUT", timed_out=True),
                    replace(self.a_result, session_id=None), replace(self.a_result, session_id="bad"),
                    replace(self.a_result, report=None), replace(self.a_result, exit_status=1),
                    replace(self.a_result, cleanup_result="taskkill_failed"))
        for variant in variants:
            with self.subTest(result=type(variant)):
                # Explicitly remove metadata only; no target history is repaired.
                self.checkpoint_path.unlink(missing_ok=True)
                result = self.execute(implement=lambda *a, **k: variant)
                self.assert_stopped(result)
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])

    def test_out_of_scope_implementation_stops_before_tests(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            (self.repository / "outside.txt").write_bytes(b"out of scope")
            return result
        self.assert_stopped(self.execute(implement=implement), "OUT_OF_SCOPE")
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])

    def test_implementation_staged_changes_stop_before_tests(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            self.git("add", "--", "app.txt")  # Deliberately unsafe fake implementer.
            return result
        self.assert_stopped(self.execute(implement=implement), "STAGED_CHANGES")
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])

    def test_failed_and_timed_out_tests_stop_before_b(self):
        for timed_out in (False, True):
            self.checkpoint_path.unlink(missing_ok=True)
            def tests(*args, **kwargs):
                good = self.run_tests_fake(*args, **kwargs)
                first = replace(good.command_results[0], exit_status=None if timed_out else 1,
                                timed_out=timed_out, cleanup_result="taskkill_completed" if timed_out else "not_needed")
                return TestRunResult(False, 1, (first,), 1)
            # Restore only this disposable fixture's files between cases.
            if not self.a_calls:
                result = self.execute(tests=tests)
            else:
                (self.repository / "app.txt").write_bytes(b"original\n")
                (self.repository / "new.bin").unlink()
                result = self.execute(tests=tests)
            self.assert_stopped(result, "TEST_TIMEOUT" if timed_out else "TESTS_NOT_PASSED")
        self.assertEqual(self.b_calls, [])

    def test_test_generated_scope_changes_stop_before_review(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            (self.repository / "outside.txt").write_bytes(b"from tests")
            return result
        self.assert_stopped(self.execute(tests=tests), "OUT_OF_SCOPE")
        self.assertEqual(self.b_calls, [])

    def test_test_generated_head_change_stops_before_review(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            self.git("commit", "--allow-empty", "-qm", "unsafe test side effect")
            return result
        self.assert_stopped(self.execute(tests=tests), "HEAD_MISMATCH")
        self.assertEqual(self.b_calls, [])

    def test_permitted_test_generated_changes_are_in_the_fresh_packet(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            (self.repository / "app.txt").write_bytes(b"after tests\n")
            return result
        result = self.execute(tests=tests)
        self.assertTrue(result.ready_for_publication, result)
        self.assertIn(("app.txt", b"after tests\n"), self.b_calls[0][0][4])
        self.assertIn(b"after tests", result.evidence.tracked_diff)

    def test_evidence_and_packet_overflow_stop_without_review(self):
        result = self.execute(evidence_max_bytes=10)
        self.assert_stopped(result, "EVIDENCE_TOO_LARGE")
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])

    def test_packet_overflow_prevents_b_launch(self):
        self.assert_stopped(self.execute(packet_max_bytes=10), "REVIEW_PACKET_INVALID")
        self.assertEqual(len(self.test_calls), 1)
        self.assertEqual(self.b_calls, [])

    def test_missing_review_instructions_stop_before_a(self):
        loaded = deepcopy(self.plan)
        loaded["tasks"][0]["review"]["prompt"] = ""
        # The loader normally rejects this; exercise the orchestration guard as
        # well, without replacing any execution setting through the public API.
        with patch("auto_coding.orchestrator.load_frozen_plan", return_value=loaded), \
             patch("auto_coding.orchestrator.plan_fingerprint", return_value=self.approval):
            result = self.execute()
        self.assert_stopped(result, "REVIEW_INSTRUCTIONS_INVALID")
        self.assertEqual(self.a_calls, [])

    def test_rejected_uncertain_missing_binding_and_missing_sessions_are_rejected(self):
        variations = (
            lambda r: replace(r, approved=False, verdict=replace(r.verdict, decision="REJECTED")),
            lambda r: replace(r, approved=False, verdict=replace(r.verdict, decision="UNCERTAIN")),
            lambda r: replace(r, verdict=None), lambda r: replace(r, session_id=None),
            lambda r: replace(r, reviewed_packet_sha256=None), lambda r: replace(r, reviewed_evidence_fingerprint=None),
            lambda r: replace(r, reviewed_packet_sha256="0" * 64),
            lambda r: replace(r, reviewed_evidence_fingerprint="0" * 64),
            lambda r: replace(r, timed_out=True), lambda r: replace(r, error_code="STDIN_INCOMPLETE"),
            lambda r: replace(r, verdict=replace(r.verdict, checked_requirements=("Behavior works",))),
        )
        for change in variations:
            self.checkpoint_path.unlink(missing_ok=True)
            (self.repository / "app.txt").write_bytes(b"original\n")
            (self.repository / "new.bin").unlink(missing_ok=True)
            result = self.execute(review=lambda *a, **k: change(self.review(*a, **k)))
            self.assert_stopped(result)

    def test_same_session_is_rejected_without_substituting_uuid(self):
        result = self.execute(review=lambda *a, **k: replace(self.review(*a, **k), session_id=_A))
        self.assert_stopped(result, "REVIEW_SESSION_NOT_INDEPENDENT")
        self.assertEqual(result.reviewer_session_id, _A)
        state = load_checkpoint(self.checkpoint_path)
        self.assertEqual(state["implementer_session_id"], _A)
        self.assertIsNone(state["reviewer_session_id"])

    def test_b_mutation_invalidates_otherwise_bound_approval(self):
        def review(*args, **kwargs):
            result = self.review(*args, **kwargs)
            (self.repository / "app.txt").write_bytes(b"reviewer changed file\n")
            return result
        self.assert_stopped(self.execute(review=review), "EVIDENCE_CHANGED_AFTER_REVIEW")

    def test_checkpoint_update_failure_at_each_stage_stops_without_retry(self):
        for failure in range(1, 7):
            self.checkpoint_path.unlink(missing_ok=True)
            (self.repository / "app.txt").write_bytes(b"original\n")
            (self.repository / "new.bin").unlink(missing_ok=True)
            self.a_calls.clear()
            self.test_calls.clear()
            self.b_calls.clear()
            calls = 0
            def update(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == failure:
                    raise OSError("private checkpoint failure")
                return update_checkpoint(*args, **kwargs)
            with patch("auto_coding.orchestrator.update_checkpoint", side_effect=update):
                result = self.execute()
            self.assert_stopped(result, "CHECKPOINT_WRITE_FAILED")
            self.assertEqual(calls, failure)
            self.assertEqual(result.checkpoint_error_code, "CHECKPOINT_WRITE_FAILED")
            if failure == 1:
                self.assertEqual(self.a_calls, [])
            if failure <= 3:
                self.assertEqual(self.test_calls, [])
            if failure <= 4:
                self.assertEqual(self.b_calls, [])

    def test_ordinary_exception_records_sanitized_stop_without_later_execution(self):
        def tests(*args, **kwargs):
            raise OSError("PRIVATE_LITERAL_IMPLEMENTATION private execution output")
        result = self.execute(tests=tests)
        self.assert_stopped(result, "EXECUTION_FAILED")
        self.assertNotIn("PRIVATE", repr(result))
        self.assertEqual(load_checkpoint(self.checkpoint_path)["stop_reason_code"], "EXECUTION_FAILED")
        self.assertEqual(self.b_calls, [])

    def test_deleted_files_have_verified_base_context(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            (self.repository / "deleted.txt").unlink()
            return result
        result = self.execute(implement=implement)
        self.assertTrue(result.ready_for_publication, result)
        self.assertIn(("deleted.txt", b"original deleted context\n"), self.b_calls[0][0][4])

    def test_no_runtime_staging_committing_or_pushing_occurs(self):
        run = subprocess.run
        commands = []
        def observe(command, **kwargs):
            commands.append(command)
            self.assertEqual(command[0], "git")
            self.assertNotIn("commit", command)
            self.assertNotIn("push", command)
            self.assertNotIn("add", command)
            return run(command, **kwargs)
        with patch("subprocess.run", side_effect=observe):
            result = self.execute()
        self.assertTrue(result.ready_for_publication, result)
        self.assertTrue(commands)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), b"")

    def test_inconsistent_successful_terminal_identity_prevents_review(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            first = replace(result.command_results[0], command_identity="0" * 64)
            return replace(result, command_results=(first, result.command_results[1]))
        self.assert_stopped(self.execute(tests=tests), "REVIEW_PACKET_INVALID")
        self.assertEqual(self.b_calls, [])

    def test_checkpoint_creation_failure_prevents_implementation(self):
        with patch("auto_coding.orchestrator.create_checkpoint", side_effect=OSError("private storage failure")):
            result = self.execute()
        self.assert_stopped(result, "CHECKPOINT_CREATE_FAILED")
        self.assertEqual(self.a_calls, [])
        self.assertFalse(self.checkpoint_path.exists())

    def test_starting_state_rechecked_after_checkpoint_creation(self):
        create = orchestrator.create_checkpoint
        def changed(*args, **kwargs):
            result = create(*args, **kwargs)
            (self.repository / "outside.txt").write_bytes(b"concurrent change")
            return result
        with patch("auto_coding.orchestrator.create_checkpoint", side_effect=changed):
            result = self.execute()
        self.assert_stopped(result, "WORKTREE_NOT_CLEAN")
        self.assertEqual(self.a_calls, [])
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "stopped")

    def test_context_budget_rejects_complete_large_file_without_truncation(self):
        baseline = b"original line\n" * 256
        (self.repository / "app.txt").write_bytes(baseline)
        self.git("add", "--", "app.txt")
        self.git("commit", "-qm", "context fixture")
        self.git("push", "-q", "origin", self.branch)
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        self.plan["expected_head"] = self.head
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        def implement(*args, **kwargs):
            self.a_calls.append((args, kwargs))
            (self.repository / "app.txt").write_bytes(baseline + b"added line\n")
            return self.a_result
        self.assert_stopped(self.execute(implement=implement, evidence_max_bytes=1000), "CONTEXT_TOO_LARGE")
        self.assertEqual(len(self.test_calls), 1)
        self.assertEqual(self.b_calls, [])


if __name__ == "__main__":
    unittest.main()
