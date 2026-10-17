"""First-task integration with fake Codex/test execution and local Git only."""

from copy import deepcopy
from contextlib import redirect_stderr
from dataclasses import replace
import hashlib
import io
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
from auto_coding.batch_state import create_checkpoint, load_checkpoint, update_checkpoint
from auto_coding.batch_report import ReportExportError
from auto_coding.change_evidence import collect_change_evidence
import auto_coding.git_publish as git_publish
from auto_coding.git_publish import GitStageResult, publish_reviewed_changes, stage_reviewed_changes
from auto_coding.integrations.codex import ImplementationOutcome, ImplementationReport
from auto_coding.plan_freeze import freeze_approved_plan, plan_fingerprint
from auto_coding.review import ReviewResult, ReviewVerdict, build_review_packet
from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity
from auto_coding.operations import TimeoutDiagnostic
from auto_coding.integrations.windows_process import WindowsProcessResult
import auto_coding.test_runner as approved_runner
from test_plans import make_v2_plan


_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"


class OrchestratorTests(unittest.TestCase):
    def freeze_test_contract(self, *, inline=False, single=False):
        task = self.plan["tasks"][0]
        task["permitted_files"].remove("deleted.txt")
        contents = (self.repository / "deleted.txt").read_bytes()
        task["test_contract"] = {"inputs_complete": True, "failure_protocols": ["exit-status-v1"] * 2,
            "protected_files": [] if inline else [{"path": "deleted.txt", "size_bytes": len(contents),
                                                    "sha256": hashlib.sha256(contents).hexdigest()}]}
        if single:
            self.plan["tasks"] = [task]
        self.frozen_path = self.parent / "contract-frozen.json"
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)

    def test_protected_input_mismatch_stops_before_codex_a(self):
        self.freeze_test_contract()
        # A wrong manifest at a clean Git base exercises the controller hash
        # check independently of the ordinary dirty-state preflight.
        task = self.plan["tasks"][0]
        task["test_contract"]["protected_files"][0]["sha256"] = "0" * 64
        self.frozen_path = self.parent / "wrong-contract-frozen.json"
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        result = self.execute()
        self.assert_stopped(result, "PROTECTED_TEST_INPUT_CHANGED")
        self.assertEqual((self.a_calls, self.test_calls, self.b_calls), ([], [], []))
        self.assertFalse(self.checkpoint_path.exists())

    def test_codex_a_modification_of_protected_input_stops_before_tests(self):
        self.freeze_test_contract()
        def mutate(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            (self.repository / "deleted.txt").write_bytes(b"tampered\n")
            return result
        result = self.execute(implement=mutate)
        self.assert_stopped(result, "PROTECTED_TEST_INPUT_CHANGED")
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (1, 0, 0))
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)

    def test_protected_input_mutated_during_tests_prevents_review(self):
        self.freeze_test_contract()
        def mutate(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            (self.repository / "deleted.txt").write_bytes(b"tampered\n")
            return result
        result = self.execute(tests=mutate)
        self.assert_stopped(result, "PROTECTED_TEST_INPUT_CHANGED")
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (1, 1, 0))
        self.assertEqual(self.test_calls[0][1], {"test_contract": self.plan["tasks"][0]["test_contract"]})

    def test_protected_input_change_after_review_stops_publication_handoff(self):
        self.freeze_test_contract()
        def mutate(*args, **kwargs):
            result = self.review(*args, **kwargs)
            (self.repository / "deleted.txt").write_bytes(b"tampered\n")
            return result
        result = self.execute(review=mutate)
        self.assert_stopped(result, "PROTECTED_TEST_INPUT_CHANGED")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], self.head)

    def test_approved_inline_contract_stops_once_and_reports_functional_failure(self):
        self.freeze_test_contract(inline=True, single=True)
        def tests(*args, **kwargs):
            self.test_calls.append((args, kwargs))
            with patch("auto_coding.test_runner.run_windows_process", return_value=WindowsProcessResult(10, False, "not_needed")):
                return approved_runner.run_approved_tests(*args, **kwargs)
        result = self.execute(tests=tests)
        self.assert_stopped(result, "TESTS_NOT_PASSED")
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (1, 1, 0))
        self.assertEqual(result.terminal_results.diagnostic.classification, "APPLICATION_FAILURE")
        metadata = self.parent / "stopped-contract-report"
        metadata.mkdir()
        exported = orchestrator.export_stopped_batch_report(result, self.frozen_path, self.approval,
                                                            metadata, "example/project", 200000)
        self.assertTrue(exported.reports_exported, exported)
        record = json.loads(exported.report_paths[0].read_bytes())["tasks"][0]
        self.assertEqual(record["terminal_results"]["diagnostic"]["reason_code"], "APPROVED_FUNCTIONAL_FAILURE")
        self.assertIsNone(record["review"])
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)

    def test_implementation_timeout_has_durable_diagnostic_and_no_later_execution(self):
        def timeout(*args, **kwargs):
            return replace(self.implement(*args, **kwargs), succeeded=False, exit_status=None,
                           timed_out=True, elapsed_seconds=11.5, cleanup_result="parent_not_reaped",
                           error_code="PROCESS_TIMEOUT", report=None, session_id=None)
        output = io.StringIO()
        with redirect_stderr(output):
            result = self.execute(implement=timeout)
        self.assert_stopped(result, "PROCESS_TIMEOUT")
        self.assertEqual(result.timeout.operation, "implementation")
        self.assertEqual(result.timeout.configured_seconds, 11)
        self.assertEqual(result.timeout.elapsed_seconds, 11.5)
        self.assertEqual(load_checkpoint(self.checkpoint_path)["timeout"], result.timeout.record())
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (1, 0, 0))
        self.assertIn("AUTO-CODING STOPPED - TIMEOUT", output.getvalue())
        self.assertIn("Codex A implementation", output.getvalue())
        self.assertIn("Configured limit: 11 seconds", output.getvalue())

    def test_review_timeout_has_durable_diagnostic_without_publication(self):
        def timeout(*args, **kwargs):
            self.b_calls.append((args, kwargs))
            return ReviewResult(False, None, "PROCESS_TIMEOUT", None, True, "taskkill_completed")
        output = io.StringIO()
        with redirect_stderr(output):
            result = self.execute(review=timeout)
        self.assert_stopped(result, "PROCESS_TIMEOUT")
        self.assertEqual(result.timeout.operation, "review")
        self.assertEqual(result.timeout.configured_seconds, 12)
        self.assertGreaterEqual(result.timeout.elapsed_seconds, 0)
        self.assertEqual(load_checkpoint(self.checkpoint_path)["timeout"], result.timeout.record())
        self.assertIn("Operation: Codex B review", output.getvalue())
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)

    def test_usage_or_invocation_failure_is_not_a_timeout(self):
        # Preserve either a generic failed exit or an explicit adapter reason;
        # the controller does not infer a quota diagnosis from elapsed time.
        for code in ("PROCESS_FAILED", "CODEX_USAGE_LIMIT"):
            with self.subTest(code=code):
                def failure(*args, **kwargs):
                    self.a_calls.append((args, kwargs))
                    return replace(self.a_result, succeeded=False, exit_status=1, timed_out=False,
                                   error_code=code, report=None, session_id=None)
                output = io.StringIO()
                path = self.parent / f"{code}.json"
                with redirect_stderr(output):
                    result = self.execute(implement=failure, checkpoint_path=path)
                self.assert_stopped(result, code)
                self.assertIsNone(result.timeout)
                self.assertNotIn("timeout", load_checkpoint(path))
                self.assertNotIn("TIMEOUT", output.getvalue())
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (2, 0, 0))

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


class SingleTaskPublicationTests(unittest.TestCase):
    git = OrchestratorTests.git
    implement = OrchestratorTests.implement
    run_tests_fake = OrchestratorTests.run_tests_fake
    review = OrchestratorTests.review
    execute = OrchestratorTests.execute

    def setUp(self):
        OrchestratorTests.setUp(self)
        self.plan["tasks"] = self.plan["tasks"][:1]
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        implementation = self.implement()
        tests = self.run_tests_fake(self.repository, self.plan["tasks"][0]["test_commands"], 13)
        evidence = collect_change_evidence(self.repository, self.plan["tasks"][0]["permitted_files"], 100_000)
        context = orchestrator._context(self.repository, self.plan, evidence, 5, 100_000)
        packet = build_review_packet(evidence, self.plan["tasks"][0]["acceptance_criteria"],
                                     self.plan["tasks"][0]["test_commands"], tests, context, 200_000)
        review = self.review(evidence, self.plan["tasks"][0]["acceptance_criteria"],
                             self.plan["tasks"][0]["test_commands"], tests, context, 200_000)
        state = {"schema_version": 1, "batch_id": self.plan["batch_id"], "frozen_plan_sha256": self.approval,
                 "expected_branch": self.branch, "base_head": self.head, "status": "pending", "stage": "pending",
                 "current_task_id": None, "implementer_session_id": None, "reviewer_session_id": None, "stop_reason_code": None}
        create_checkpoint(self.checkpoint_path, state)
        state.update(status="running", stage="publishing", current_task_id="task-1",
                     implementer_session_id=_A, reviewer_session_id=_B)
        update_checkpoint(self.checkpoint_path, state)
        self.handoff = orchestrator.FirstTaskReviewResult(
            "READY_FOR_PUBLICATION", "publishing", batch_id=self.plan["batch_id"], task_id="task-1", plan_sha256=self.approval,
            implementer_session_id=_A, reviewer_session_id=_B, implementation_started=True, review_started=True,
            repository_directory=self.repository, checkpoint_path=self.checkpoint_path, expected_branch=self.branch,
            expected_head=self.head, expected_origin_url=str(self.remote),
            permitted_files=tuple(self.plan["tasks"][0]["permitted_files"]), implementation=implementation,
            terminal_results=tests, evidence=evidence, review=review, reviewed_packet=packet,
        )
        self.metadata = self.parent / "Final Reports"
        self.metadata.mkdir()
        self.stage_calls, self.publish_calls = [], []

    def publish(self, handoff=None, **changes):
        inputs = dict(handoff=self.handoff if handoff is None else handoff,
                      frozen_plan_path=self.frozen_path, approved_sha256=self.approval,
                      git_timeout_seconds=5, evidence_max_bytes=100_000, packet_max_bytes=200_000,
                      report_max_bytes=200_000, metadata_directory=self.metadata, repository_identifier="example/project")
        def stage(*args, **kwargs):
            self.stage_calls.append((args, kwargs))
            return stage_reviewed_changes(*args, **kwargs)
        def publish(*args, **kwargs):
            self.publish_calls.append((args, kwargs))
            return publish_reviewed_changes(*args, **kwargs)
        with patch("auto_coding.orchestrator.stage_reviewed_changes", side_effect=stage), \
             patch("auto_coding.orchestrator.publish_reviewed_changes", side_effect=publish), \
             patch("auto_coding.orchestrator.run_implementation", side_effect=AssertionError("extra Codex invocation")), \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=AssertionError("extra reviewer invocation")), \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=AssertionError("extra terminal execution")), \
             patch("auto_coding.orchestrator.preflight_frozen_plan", side_effect=AssertionError("initial clean preflight repeated")):
            return orchestrator.publish_single_task_batch(**(inputs | changes))

    def remote_head(self):
        return self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split("\t")[0]

    def report(self):
        return json.loads((self.metadata / "batch-report.json").read_bytes())

    def assert_no_mutations(self):
        self.assertEqual(self.stage_calls, [])
        self.assertEqual(self.publish_calls, [])
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.remote_head(), self.head)

    def test_actual_d16a_handoff_publishes_exact_binary_changes_once_and_finalizes(self):
        self.checkpoint_path.unlink()
        (self.repository / "app.txt").write_bytes(b"original\n")
        (self.repository / "new.bin").unlink()
        self.a_calls.clear()
        self.test_calls.clear()
        self.b_calls.clear()
        self.handoff = self.execute()
        self.assertTrue(self.handoff.ready_for_publication, self.handoff)
        result = self.publish()
        self.assertEqual(result.outcome, "PUBLISHED_FINALIZED", result)
        self.assertTrue(result.published)
        self.assertTrue(result.checkpoint_finalized)
        self.assertTrue(result.reports_exported)
        self.assertEqual(len(self.stage_calls), 1)
        self.assertEqual(len(self.publish_calls), 1)
        self.assertIs(self.stage_calls[0][0][6], self.handoff.reviewed_packet)
        self.assertIs(self.stage_calls[0][0][7], self.handoff.review)
        self.assertIs(self.publish_calls[0][0][6], self.handoff.reviewed_packet)
        self.assertEqual(self.publish_calls[0][0][-1], "auto-coding: task-1")
        self.assertEqual(result.commit_sha, self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(result.remote_sha, result.commit_sha)
        self.assertEqual(self.remote_head(), result.commit_sha)
        self.assertEqual(self.git("rev-list", "--count", f"{self.head}..HEAD"), b"1\n")
        self.assertEqual(self.git("show", "--format=%s", "--no-patch", "HEAD"), b"auto-coding: task-1\n")
        self.assertEqual(self.git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines(), [b"app.txt", b"new.bin"])
        self.assertEqual(self.git("show", "HEAD:new.bin"), b"\0\xffPRIVATE_BINARY\x80")
        state = load_checkpoint(self.checkpoint_path)
        self.assertEqual((state["status"], state["stage"], state["current_task_id"]), ("completed", "finished", None))
        self.assertEqual((state["implementer_session_id"], state["reviewer_session_id"]), (_A, _B))
        data = self.report()
        self.assertEqual(data["batch"]["completed_tasks"], 1)
        self.assertIsNone(data["batch"]["started_at"])
        self.assertIsNone(data["batch"]["ended_at"])
        self.assertEqual(data["events"], [])
        task = data["tasks"][0]
        self.assertEqual(task["publishing"]["remote_verification"], "verified")
        self.assertEqual(task["implementation"]["model_claims"]["model_reported_confidence"], "HIGH")
        self.assertEqual(task["review"]["reviewed_packet_sha256"], self.handoff.review.reviewed_packet_sha256)
        self.assertEqual(task["approved_command_identities"], [r.command_identity for r in self.handoff.terminal_results.command_results])
        self.assertIn("Not separately recorded", task["objective"])
        report_text = (self.metadata / "batch-report.json").read_text() + (self.metadata / "batch-report.md").read_text(encoding="utf-8")
        for private in ("PRIVATE_LITERAL_IMPLEMENTATION", "PRIVATE_LITERAL_REVIEW", "PRIVATE_ARG", "PRIVATE_BINARY", "tracked_diff_base64"):
            self.assertNotIn(private, report_text)
            self.assertNotIn(private, repr(result))
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (1, 1, 1))

    def test_invalid_or_inconsistent_handoffs_never_reach_staging(self):
        variants = (object(), replace(self.handoff, outcome="STOPPED"), replace(self.handoff, error_code="FAILED"),
                    replace(self.handoff, checkpoint_error_code="FAILED"), replace(self.handoff, batch_id="other"),
                    replace(self.handoff, task_id="task-2"), replace(self.handoff, plan_sha256="0" * 64),
                    replace(self.handoff, expected_branch="other"), replace(self.handoff, expected_head="0" * 40),
                    replace(self.handoff, expected_origin_url="https://wrong.invalid/repo"),
                    replace(self.handoff, permitted_files=("app.txt",)),
                    replace(self.handoff, timeout=TimeoutDiagnostic("implementation", 11)),
                    replace(self.handoff, implementer_session_id=_B), replace(self.handoff, reviewer_session_id=None),
                    replace(self.handoff, reviewed_packet=None), replace(self.handoff, evidence=None),
                    replace(self.handoff, implementation=replace(self.handoff.implementation, succeeded=False)),
                    replace(self.handoff, terminal_results=replace(self.handoff.terminal_results, passed=False)),
                    replace(self.handoff, review=replace(self.handoff.review, approved=False)),
                    replace(self.handoff, review=replace(self.handoff.review, reviewed_packet_sha256="0" * 64)))
        before = self.checkpoint_path.read_bytes()
        for handoff in variants:
            result = self.publish(handoff=handoff)
            self.assertFalse(result.published)
            self.assertEqual(result.outcome, "STOPPED_BEFORE_STAGING")
            self.assertEqual(self.checkpoint_path.read_bytes(), before)
        self.assert_no_mutations()

    def test_multi_task_plan_rejected_before_git_mutation(self):
        second = deepcopy(self.plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        self.plan["tasks"].append(second)
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        result = self.publish()
        self.assertEqual(result.error_code, "SINGLE_TASK_BATCH_REQUIRED")
        self.assert_no_mutations()

    def test_modified_frozen_plan_or_wrong_approval_prevents_staging(self):
        self.assertFalse(self.publish(approved_sha256="0" * 64).published)
        self.frozen_path.write_bytes(b"{corrupt}")
        self.assertFalse(self.publish().published)
        self.assert_no_mutations()

    def test_wrong_checkpoint_identity_or_sessions_prevent_staging(self):
        original = load_checkpoint(self.checkpoint_path)
        for change in ({"batch_id": "other"}, {"frozen_plan_sha256": "0" * 64},
                       {"reviewer_session_id": "33333333-3333-4333-8333-333333333333"}, {"stage": "review"}):
            self.checkpoint_path.write_text(json.dumps(dict(original, **change)), encoding="utf-8")
            result = self.publish()
            self.assertFalse(result.published)
            self.assertEqual(result.error_code, "CHECKPOINT_HANDOFF_MISMATCH")
        self.assert_no_mutations()

    def test_packet_cannot_substitute_frozen_criteria_or_commands(self):
        raw = json.loads(self.handoff.reviewed_packet)
        raw["caller_supplied_review_inputs"]["acceptance_criteria"]["items"] = ["replacement criterion"]
        packet = json.dumps(raw, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        review = replace(self.handoff.review, reviewed_packet_sha256=hashlib.sha256(packet).hexdigest(),
                         verdict=replace(self.handoff.review.verdict, checked_requirements=("replacement criterion",)))
        result = self.publish(handoff=replace(self.handoff, reviewed_packet=packet, review=review))
        self.assertFalse(result.published)
        self.assert_no_mutations()

    def test_unreviewed_worktree_changes_fail_real_staging_gate(self):
        (self.repository / "app.txt").write_bytes(b"unreviewed\n")
        result = self.publish()
        self.assertEqual(result.outcome, "STAGING_FAILED_OR_UNCERTAIN")
        self.assertEqual(result.stage, "staging")
        self.assertEqual(len(self.stage_calls), 1)
        self.assertEqual(self.publish_calls, [])
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "stopped")
        self.assertFalse(self.report()["tasks"][0]["staging"]["passed"])
        publishing = self.report()["tasks"][0]["publishing"]
        self.assertEqual(publishing["outcome"], "NOT_ATTEMPTED")
        self.assertFalse(publishing["commit_attempted"])
        self.assertFalse(publishing["push_attempted"])

    def test_preexisting_index_changes_are_preserved_without_commit(self):
        self.git("add", "--", "app.txt")
        before = self.git("diff", "--cached", "--binary")
        result = self.publish()
        self.assertFalse(result.published)
        self.assertEqual(self.publish_calls, [])
        self.assertEqual(self.git("diff", "--cached", "--binary"), before)

    def test_failed_stage_does_not_call_commit_gate(self):
        with patch("auto_coding.orchestrator.stage_reviewed_changes", return_value=GitStageResult(False, "GIT_TIMEOUT")):
            result = orchestrator.publish_single_task_batch(self.handoff, self.frozen_path, self.approval, 5,
                                                            100_000, 200_000, 200_000, self.metadata, "example/project")
        self.assertEqual(result.outcome, "STAGING_FAILED_OR_UNCERTAIN")
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertFalse(result.publishing_invoked)

    def failure_query(self, action, *, after_actual=False):
        original = git_publish._git_query
        calls = []
        def query(repository, arguments, *args, **kwargs):
            calls.append(arguments)
            if arguments[0] == action:
                if after_actual:
                    code, output, error = original(repository, arguments, *args, **kwargs)
                    self.assertEqual((code, error), (0, None))
                    return None, None, "GIT_TIMEOUT"
                return 1, b"", None
            return original(repository, arguments, *args, **kwargs)
        return query, calls

    def test_commit_failure_preserves_index_and_stops_without_push(self):
        query, calls = self.failure_query("commit")
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_FAILED_OR_AMBIGUOUS")
        self.assertFalse(result.published)
        self.assertIsNone(result.commit_sha)
        self.assertFalse(any(args[0] == "push" for args in calls))
        self.assertEqual(self.git("diff", "--cached", "--name-only").splitlines(), [b"app.txt", b"new.bin"])
        self.assertEqual(self.report()["tasks"][0]["publishing"]["outcome"], "COMMIT_FAILED_OR_AMBIGUOUS")

    def test_ambiguous_commit_is_not_pushed_or_retried(self):
        query, calls = self.failure_query("commit", after_actual=True)
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_FAILED_OR_AMBIGUOUS")
        self.assertIsNone(result.commit_sha)
        self.assertNotEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(sum(args[0] == "commit" for args in calls), 1)
        self.assertFalse(any(args[0] == "push" for args in calls))

    def test_created_but_unverified_commit_is_not_pushed(self):
        original = git_publish._git_query
        calls = []
        def query(repository, arguments, *args, **kwargs):
            calls.append(arguments)
            code, output, error = original(repository, arguments, *args, **kwargs)
            if arguments[0] == "--no-replace-objects":
                output = b"tree " + b"0" * 40 + output[45:]
            return code, output, error
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "COMMIT_CREATED_NOT_PUSHED")
        self.assertIsNone(result.commit_sha)
        self.assertIsNotNone(result.publishing.observed_head_sha)
        self.assertFalse(any(args[0] == "push" for args in calls))

    def test_push_failure_preserves_verified_local_commit_and_stopped_report(self):
        query, calls = self.failure_query("push")
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertFalse(result.published)
        self.assertEqual(result.commit_sha, self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(self.remote_head(), self.head)
        self.assertEqual(sum(args[0] == "push" for args in calls), 1)
        self.assertEqual(self.report()["checkpoint"]["status"], "stopped")
        self.assertEqual(self.report()["tasks"][0]["publishing"]["remote_verification"], "not_confirmed")

    def test_push_timeout_after_actual_remote_update_remains_ambiguous(self):
        query, calls = self.failure_query("push", after_actual=True)
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertFalse(result.published)
        self.assertIsNone(result.remote_sha)
        self.assertEqual(self.remote_head(), result.commit_sha)
        self.assertEqual(sum(args[0] == "push" for args in calls), 1)
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "stopped")
        self.assertEqual(result.timeout.operation, "git_push")
        self.assertEqual(load_checkpoint(self.checkpoint_path)["timeout"], result.timeout.record())
        self.assertEqual(self.report()["checkpoint"]["timeout"], result.timeout.record())

    def test_checkpoint_failure_before_staging_prevents_all_git_mutations(self):
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=OSError("private")) as update:
            result = self.publish()
        self.assertEqual(result.outcome, "STOPPED_BEFORE_STAGING")
        self.assertEqual(result.checkpoint_error_code, "CHECKPOINT_WRITE_FAILED")
        self.assertEqual(update.call_count, 1)
        self.assert_no_mutations()

    def test_checkpoint_failure_after_staging_prevents_commit(self):
        def update(path, state, **kwargs):
            if state["status"] == "running" and state["stage"] == "publishing":
                raise OSError("private")
            return update_checkpoint(path, state, **kwargs)
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=update):
            result = self.publish()
        self.assertEqual(result.outcome, "STOPPED_AFTER_STAGING")
        self.assertFalse(result.published)
        self.assertEqual(self.publish_calls, [])
        self.assertTrue(self.git("diff", "--cached", "--name-only"))

    def test_verified_publication_survives_checkpoint_finalization_failure(self):
        completions = []
        def update(path, state, **kwargs):
            if state["status"] == "completed":
                completions.append(kwargs)
                raise OSError("private finalization failure")
            self.assertFalse(kwargs["completion_verified"])
            return update_checkpoint(path, state, **kwargs)
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=update):
            result = self.publish()
        self.assertTrue(result.published)
        self.assertEqual(result.outcome, "PUBLISHED_CHECKPOINT_FAILED")
        self.assertEqual(completions, [{"completion_verified": True}])
        self.assertEqual(result.commit_sha, self.remote_head())
        self.assertEqual(len(self.publish_calls), 1)
        self.assertFalse(result.checkpoint_finalized)
        self.assertEqual(result.report_error_code, "REPORT_CHECKPOINT_NOT_TERMINAL")

    def test_actual_report_export_failure_preserves_verified_push_and_completed_checkpoint(self):
        link = os.link
        calls = 0
        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("private export failure")
            link(source, destination)
        with patch("auto_coding.batch_report.os.link", side_effect=fail_second):
            result = self.publish()
        self.assertTrue(result.published)
        self.assertEqual(result.outcome, "PUBLISHED_REPORT_FAILED")
        self.assertEqual(result.report_error_code, "REPORT_EXPORT_FAILED")
        self.assertEqual(result.commit_sha, self.remote_head())
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "completed")
        self.assertEqual(list(self.metadata.iterdir()), [])
        self.assertEqual(len(self.publish_calls), 1)

    def test_partial_report_paths_are_preserved_without_hiding_publication(self):
        partial = self.metadata / "batch-report.json"
        def export(*args, **kwargs):
            partial.write_bytes(b"partial controlled export")
            raise ReportExportError((partial,))
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=export):
            result = self.publish()
        self.assertTrue(result.published)
        self.assertEqual(result.partial_report_paths, (partial,))
        self.assertEqual(result.outcome, "PUBLISHED_REPORT_FAILED")

    def test_interrupted_report_export_preserves_verified_publication(self):
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=KeyboardInterrupt):
            result = self.publish()
        self.assertTrue(result.published)
        self.assertEqual(result.outcome, "PUBLISHED_REPORT_FAILED")
        self.assertEqual(result.report_error_code, "REPORT_EXPORT_INTERRUPTED")
        self.assertEqual(result.commit_sha, self.remote_head())
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "completed")
        self.assertEqual(len(self.publish_calls), 1)

    def existing_report(self, name):
        path = self.metadata / name
        path.write_bytes(b"existing report")
        result = self.publish()
        self.assertEqual(result.error_code, "REPORTS_ALREADY_EXIST")
        self.assertEqual(path.read_bytes(), b"existing report")
        self.assert_no_mutations()

    def test_existing_json_report_blocks_staging(self):
        self.existing_report("batch-report.json")

    def test_existing_markdown_report_blocks_staging(self):
        self.existing_report("batch-report.md")

    def test_report_directory_inside_target_blocks_staging(self):
        inside = self.repository / "metadata"
        inside.mkdir()
        result = self.publish(metadata_directory=inside)
        self.assertEqual(result.error_code, "UNSAFE_REPORT_LOCATION")
        self.assertEqual(list(inside.iterdir()), [])
        self.assert_no_mutations()

    def test_secondary_reporting_failure_does_not_hide_original_publish_failure(self):
        query, calls = self.failure_query("push")
        with patch("auto_coding.git_publish._git_query", side_effect=query), \
             patch("auto_coding.orchestrator.export_batch_report", side_effect=ReportExportError()):
            result = self.publish()
        self.assertEqual(result.outcome, "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(result.error_code, "GIT_FAILED")
        self.assertEqual(result.report_error_code, "REPORT_EXPORT_FAILED")
        self.assertEqual(result.stage, "publishing")

    def test_no_force_reset_merge_or_extra_gate_attempts(self):
        original = git_publish._git_query
        arguments_seen = []
        def query(repository, arguments, *args, **kwargs):
            arguments_seen.append(arguments)
            self.assertFalse(any(value in arguments for value in ("--force", "-f", "reset", "merge", "checkout", "--amend")))
            return original(repository, arguments, *args, **kwargs)
        with patch("auto_coding.git_publish._git_query", side_effect=query):
            result = self.publish()
        self.assertTrue(result.published, result)
        self.assertEqual(sum(args[0] == "commit" for args in arguments_seen), 1)
        self.assertEqual(sum(args[0] == "push" for args in arguments_seen), 1)
        self.assertEqual(self.publish_calls[0][0][-1], "auto-coding: task-1")

    def test_invalid_timeout_and_limits_prevent_staging(self):
        for changes in ({"git_timeout_seconds": 0}, {"git_timeout_seconds": 10**1000},
                        {"git_timeout_seconds": True}, {"evidence_max_bytes": 0},
                        {"packet_max_bytes": False}, {"report_max_bytes": -1}):
            result = self.publish(**changes)
            self.assertFalse(result.published)
            self.assertIn(result.error_code, ("INVALID_TIMEOUT", "INVALID_MAX_BYTES"))
        self.assert_no_mutations()

    def test_malformed_checkpoint_is_rejected_without_overwrite(self):
        self.checkpoint_path.write_bytes(b"{incomplete checkpoint")
        result = self.publish()
        self.assertEqual(result.error_code, "CHECKPOINT_INVALID")
        self.assertEqual(self.checkpoint_path.read_bytes(), b"{incomplete checkpoint")
        self.assert_no_mutations()

    def test_fabricated_successful_stage_flag_cannot_bypass_d13_index_verification(self):
        fabricated = GitStageResult(True, None, True, self.handoff.evidence.changed_paths, "0" * 64)
        with patch("auto_coding.orchestrator.stage_reviewed_changes", return_value=fabricated), \
             patch("auto_coding.orchestrator.publish_reviewed_changes", wraps=publish_reviewed_changes) as publish:
            result = orchestrator.publish_single_task_batch(self.handoff, self.frozen_path, self.approval, 5,
                                                            100_000, 200_000, 200_000, self.metadata, "example/project")
        self.assertFalse(result.published)
        self.assertEqual(publish.call_count, 1)
        self.assertFalse(result.publishing.commit_attempted)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), b"")


class StoppedBatchReportTests(unittest.TestCase):
    git = OrchestratorTests.git
    implement = OrchestratorTests.implement
    run_tests_fake = OrchestratorTests.run_tests_fake
    review = OrchestratorTests.review
    execute = OrchestratorTests.execute

    def setUp(self):
        OrchestratorTests.setUp(self)
        self.plan["tasks"] = self.plan["tasks"][:1]
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        self.metadata = self.parent / "Stopped Reports"
        self.metadata.mkdir()

    def stopped(self):
        failure = replace(self.a_result, succeeded=False, exit_status=1, error_code="CODEX_FAILED", report=None)
        return self.execute(implement=lambda *a, **k: failure)

    def export(self, handoff, **changes):
        values = dict(handoff=handoff, frozen_plan_path=self.frozen_path, approved_sha256=self.approval,
                      metadata_directory=self.metadata, repository_identifier="example/disposable", report_max_bytes=200000)
        return orchestrator.export_stopped_batch_report(**(values | changes))

    def test_terminal_stop_exports_observations_without_mutating_checkpoint_or_git(self):
        handoff = self.stopped()
        checkpoint = self.checkpoint_path.read_bytes()
        with patch("auto_coding.orchestrator.stage_reviewed_changes", side_effect=AssertionError("must not stage")), \
             patch("auto_coding.orchestrator.publish_reviewed_changes", side_effect=AssertionError("must not publish")):
            result = self.export(handoff)
        self.assertTrue(result.reports_exported, result)
        self.assertIsNone(result.error_code)
        self.assertEqual(result.report_paths, (self.metadata / "batch-report.json", self.metadata / "batch-report.md"))
        record = json.loads(result.report_paths[0].read_bytes())["tasks"][0]
        self.assertEqual(record["status"], "stopped")
        self.assertEqual(record["stop_reason_code"], "CODEX_FAILED")
        self.assertEqual(record["implementation"]["exit_status"], 1)
        for field in ("evidence", "terminal_results", "review", "staging", "publishing"):
            self.assertIsNone(record[field])
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("status", "--porcelain"), b"")

    def test_uncertain_review_with_three_descriptive_checks_exports_without_publication(self):
        # Reproduce the live failure pattern entirely in this disposable fixture.
        # These descriptions model the reported shape, not a saved live transcript.
        task = self.plan["tasks"][0]
        task.update(permitted_files=["proof.txt"], acceptance_criteria=[
            "proof.txt contains the approved bytes.", "The approved test succeeds.", "Only proof.txt changes."])
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        proof = b"Stopped review reporting regression\n"
        verdict = ReviewVerdict(
            "UNCERTAIN", ("Independent confirmation of the proof bytes remains unavailable",),
            ("The terminal result reports success",),
            ("Compared the proof file with the requested content",
             "Inspected the reported terminal test success", "Examined the changed file scope"), "MEDIUM")

        def implement(*args, **kwargs):
            (self.repository / "proof.txt").write_bytes(proof)
            return replace(self.a_result, report=replace(self.a_result.report, files_reported=("proof.txt",)))

        def review(*args, **kwargs):
            return ReviewResult(False, verdict, "REVIEW_NOT_APPROVED", 0, False, "not_needed", _B)

        with patch("auto_coding.orchestrator.stage_reviewed_changes") as stage, \
             patch("auto_coding.orchestrator.publish_reviewed_changes") as publish:
            handoff = self.execute(implement=implement, review=review)
            self.assertEqual(handoff.outcome, "STOPPED")
            self.assertEqual(handoff.stage, "review")
            self.assertEqual(handoff.review.verdict, verdict)
            checkpoint = self.checkpoint_path.read_bytes()
            state = load_checkpoint(self.checkpoint_path)
            self.assertEqual((state["status"], state["stage"]), ("stopped", "review"))
            self.assertEqual((state["implementer_session_id"], state["reviewer_session_id"]), (_A, _B))
            result = self.export(handoff)
            stage.assert_not_called()
            publish.assert_not_called()
        self.assertTrue(result.reports_exported, result)
        self.assertIsNone(result.error_code)
        data = json.loads(result.report_paths[0].read_bytes())
        recorded = data["tasks"][0]
        self.assertTrue(recorded["implementation"]["succeeded"])
        self.assertTrue(recorded["terminal_results"]["passed"])
        self.assertEqual(recorded["review"]["model_verdict"], {
            "decision": verdict.decision, "blocking_issues": list(verdict.blocking_issues),
            "nonblocking_notes": list(verdict.nonblocking_notes),
            "checked_requirements": list(verdict.checked_requirements), "model_reported_confidence": verdict.confidence})
        self.assertFalse(recorded["review"]["approved"])
        self.assertIsNone(recorded["staging"])
        self.assertIsNone(recorded["publishing"])
        self.assertEqual(data["batch"]["completed_tasks"], 0)
        markdown = result.report_paths[1].read_text(encoding="utf-8")
        for text in (verdict.decision, verdict.confidence, *verdict.blocking_issues,
                     *verdict.nonblocking_notes, *verdict.checked_requirements):
            self.assertIn(text, markdown)
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint)
        self.assertEqual((self.repository / "proof.txt").read_bytes(), proof)
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], self.head)
        self.assertEqual(self.git("diff", "--cached"), b"")
        self.assertEqual(self.git("status", "--porcelain"), b"?? proof.txt\n")

    def test_stop_before_implementation_keeps_pending_task_unavailable(self):
        self.plan["tasks"][0]["review"]["prompt"] = "\ud800"
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        handoff = self.execute()
        self.assertEqual(handoff.error_code, "REVIEW_INSTRUCTIONS_INVALID")
        result = self.export(handoff)
        self.assertTrue(result.reports_exported, result)
        data = json.loads(result.report_paths[0].read_bytes())
        self.assertEqual(data["checkpoint"]["status"], "stopped")
        self.assertEqual(data["tasks"][0]["status"], "pending")
        self.assertFalse(data["tasks"][0]["implementation_started"])
        self.assertIsNone(data["tasks"][0]["implementation"])

    def test_missing_or_nonterminal_checkpoint_does_not_invent_stop(self):
        handoff = self.execute(approved_sha256="0" * 64)
        result = self.export(handoff)
        self.assertEqual(result.error_code, "REPORT_CHECKPOINT_UNAVAILABLE")
        self.assertEqual(list(self.metadata.iterdir()), [])
        handoff = self.execute()
        result = self.export(replace(handoff, outcome="STOPPED", error_code="EXECUTION_FAILED"))
        self.assertEqual(result.error_code, "REPORT_CHECKPOINT_NOT_TERMINAL")
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "running")

    def test_mismatched_terminal_checkpoint_prevents_export(self):
        handoff = self.stopped()
        result = self.export(replace(handoff, implementer_session_id=_B))
        self.assertEqual(result.error_code, "REPORT_CHECKPOINT_MISMATCH")
        self.assertFalse(result.reports_exported)
        self.assertEqual(list(self.metadata.iterdir()), [])

    def test_invalid_observations_are_not_repaired_to_pass_report_validation(self):
        handoff = self.stopped()
        malformed = replace(handoff, implementation=replace(handoff.implementation, succeeded=True))
        result = self.export(malformed)
        self.assertEqual(result.error_code, "REPORT_INPUT_INVALID_OR_TOO_LARGE")
        self.assertEqual(list(self.metadata.iterdir()), [])

    def test_report_size_and_no_overwrite_are_preserved(self):
        handoff = self.stopped()
        self.assertEqual(self.export(handoff, report_max_bytes=1).error_code, "REPORT_INPUT_INVALID_OR_TOO_LARGE")
        existing = self.metadata / "batch-report.json"
        existing.write_bytes(b"preserved")
        self.assertFalse(self.export(handoff).reports_exported)
        self.assertEqual(existing.read_bytes(), b"preserved")
        self.assertFalse((self.metadata / "batch-report.md").exists())

    def test_report_export_failure_and_partial_paths_are_separate_observations(self):
        handoff = self.stopped()
        checkpoint = self.checkpoint_path.read_bytes()
        partial = self.metadata / "batch-report.json"
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=ReportExportError((partial,))):
            result = self.export(handoff)
        self.assertEqual(result.error_code, "REPORT_EXPORT_FAILED")
        self.assertEqual(result.partial_report_paths, (partial,))
        self.assertEqual(handoff.error_code, "CODEX_FAILED")
        self.assertEqual(self.checkpoint_path.read_bytes(), checkpoint)


class SequentialBatchTests(unittest.TestCase):
    def test_opted_in_contracts_preserve_real_sequential_publication_and_diagnostics(self):
        for task in self.plan["tasks"]:
            task["test_contract"] = {"inputs_complete": True, "protected_files": [], "failure_protocols": ["exit-status-v1"]}
        self.configure(3)
        def tests(*args, **kwargs):
            self.test_calls.append((args, kwargs))
            with patch("auto_coding.test_runner.run_windows_process", return_value=WindowsProcessResult(0, False, "not_needed")):
                return approved_runner.run_approved_tests(*args, **kwargs)
        result = self.run_batch(tests=tests)
        self.assert_success(result, 3)
        for index, record in enumerate(self.data()["tasks"]):
            self.assertEqual(self.test_calls[index][1], {"test_contract": self.plan["tasks"][index]["test_contract"]})
            self.assertEqual(record["terminal_results"]["diagnostic"]["classification"], "PASSED")

    git = OrchestratorTests.git
    run_tests_fake = OrchestratorTests.run_tests_fake

    def setUp(self):
        OrchestratorTests.setUp(self)
        template = deepcopy(self.plan["tasks"][0])
        self.plan["tasks"] = []
        for index in range(1, 4):
            task = deepcopy(template)
            task.update(task_id=f"task-{index}", order=index,
                        depends_on=[] if index == 1 else [f"task-{index-1}"],
                        permitted_files=[f"file-{index}.txt"], acceptance_criteria=[f"Criterion {index}"],
                        test_commands=[[f"frozen-test-{index}", f"PRIVATE_ARG_{index}"]])
            task["implementation"].update(prompt=f"PRIVATE_IMPLEMENT_{index}",
                                           model=("gpt-6-sol", "gpt-6.1-sol", "gpt-6-sol")[index-1],
                                           reasoning_effort=("low", "medium", "high")[index-1])
            task["review"].update(prompt=f"PRIVATE_REVIEW_{index}", reasoning_effort=("high", "medium", "low")[index-1])
            task["limits"].update(implementation_seconds=10 + index, test_seconds=20 + index, review_seconds=30 + index)
            self.plan["tasks"].append(task)
        self.metadata = self.parent / "Sequential Metadata"
        self.metadata.mkdir()
        self.active = 0
        self.observed_heads, self.stage_calls, self.publish_calls = [], [], []
        self.configure(3)

    def configure(self, count):
        self.plan["tasks"] = self.plan["tasks"][:count]
        self.frozen_path.unlink()
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)
        self.frozen_bytes = self.frozen_path.read_bytes()

    @staticmethod
    def session(index, role):
        return f"{index * 2 + role:08x}-1111-4111-8111-111111111111"

    def implement(self, *args, **kwargs):
        self.active += 1
        self.a_calls.append((args, kwargs))
        self.observed_heads.append(self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(self.git("status", "--porcelain"), b"")
        (self.repository / f"file-{self.active}.txt").write_bytes(f"content {self.active}\n".encode())
        return replace(self.a_result, session_id=self.session(self.active, 0))

    def review(self, *args, **kwargs):
        self.b_calls.append((args, kwargs))
        packet = build_review_packet(*args)
        return ReviewResult(True, ReviewVerdict("APPROVED", (), (), tuple(args[1]), "HIGH"), None,
                            0, False, "not_needed", self.session(self.active, 1),
                            hashlib.sha256(packet).hexdigest(), args[0].fingerprint)

    def run_batch(self, implement=None, tests=None, review=None, **changes):
        values = dict(frozen_plan_path=self.frozen_path, approved_sha256=self.approval,
                      repository_directory=self.repository, metadata_directory=self.metadata,
                      repository_identifier="example/sequential", git_timeout_seconds=5,
                      evidence_max_bytes=100000, packet_max_bytes=200000, report_max_bytes=200000)
        def stage(*args, **kwargs):
            self.stage_calls.append(args)
            return stage_reviewed_changes(*args, **kwargs)
        def publish(*args, **kwargs):
            self.publish_calls.append(args)
            return publish_reviewed_changes(*args, **kwargs)
        with patch("auto_coding.orchestrator.run_implementation", side_effect=implement or self.implement), \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=tests or self.run_tests_fake), \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=review or self.review), \
             patch("auto_coding.orchestrator.stage_reviewed_changes", side_effect=stage), \
             patch("auto_coding.orchestrator.publish_reviewed_changes", side_effect=publish):
            return orchestrator.execute_sequential_batch(**(values | changes))

    def data(self):
        return json.loads((self.metadata / "batch-report.json").read_bytes())

    def checkpoint(self):
        return load_checkpoint(self.metadata / "batch-checkpoint.json")

    def assert_success(self, result, count):
        self.assertEqual(result.outcome, "COMPLETED", result)
        self.assertIsNone(result.error_code)
        self.assertEqual(len(result.completed_task_ids), count)
        self.assertEqual(len(self.a_calls), count)
        self.assertEqual(len(self.test_calls), count)
        self.assertEqual(len(self.b_calls), count)
        self.assertEqual(len(self.stage_calls), count)
        self.assertEqual(len(self.publish_calls), count)
        commits = [entry[1] for entry in result.publications]
        self.assertEqual(self.observed_heads, [self.head, *commits[:-1]])
        self.assertEqual([args[2] for args in self.stage_calls], self.observed_heads)
        self.assertEqual([args[2] for args in self.publish_calls], self.observed_heads)
        self.assertEqual(self.checkpoint()["base_head"], self.head)
        self.assertEqual(self.checkpoint()["current_head"], commits[-1])
        self.assertEqual(self.checkpoint()["status"], "completed")
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], commits[-1])
        self.assertEqual(self.git("status", "--porcelain"), b"")
        self.assertEqual(self.git("rev-list", "--count", f"{self.head}..HEAD").decode().strip(), str(count))
        self.assertEqual(self.frozen_path.read_bytes(), self.frozen_bytes)
        report = self.data()
        self.assertEqual(report["batch"]["completed_tasks"], count)
        self.assertTrue((self.metadata / "batch-report.md").is_file())
        for index, task in enumerate(self.plan["tasks"]):
            self.assertEqual(self.a_calls[index], ((task["implementation"]["prompt"], task["implementation"]["model"],
                             task["implementation"]["reasoning_effort"], self.repository, task["limits"]["implementation_seconds"]), {"session_id": None}))
            expected_options = {"test_contract": task["test_contract"]} if "test_contract" in task else {}
            self.assertEqual(self.test_calls[index], ((self.repository, task["test_commands"], task["limits"]["test_seconds"]), expected_options))
            args, options = self.b_calls[index]
            self.assertEqual(args[1:3], (task["acceptance_criteria"], task["test_commands"]))
            self.assertEqual(options["review_prompt"], task["review"]["prompt"])
            self.assertEqual(options["model"], task["review"]["model"])
            self.assertEqual(options["reasoning_effort"], task["review"]["reasoning_effort"])
            self.assertEqual(options["timeout_seconds"], task["limits"]["review_seconds"])
            self.assertIsNone(options["session_id"])
            self.assertEqual(report["tasks"][index]["evidence"]["changed_paths"], task["permitted_files"])
            child = load_checkpoint(self.metadata / f"task-{index+1:04d}" / "batch-checkpoint.json")
            self.assertEqual(child["base_head"], self.observed_heads[index])
            self.assertEqual(child["status"], "completed")
            self.assertEqual(report["tasks"][index]["publishing"]["commit_sha"], commits[index])
        text = (self.metadata / "batch-report.json").read_text() + (self.metadata / "batch-report.md").read_text(encoding="utf-8")
        for private in ("PRIVATE_IMPLEMENT", "PRIVATE_REVIEW", "PRIVATE_ARG", "tracked_diff_base64"):
            self.assertNotIn(private, text)

    def test_two_tasks_publish_in_order_at_verified_successive_heads(self):
        self.configure(2)
        self.assert_success(self.run_batch(), 2)

    def test_three_tasks_publish_with_independent_frozen_settings(self):
        self.assert_success(self.run_batch(), 3)

    def test_later_deleted_context_uses_previous_task_commit_not_batch_base(self):
        self.configure(2)
        self.plan["tasks"][1]["permitted_files"] = ["file-1.txt"]
        self.configure(2)
        def implement(*args, **kwargs):
            if self.active == 0:
                return self.implement(*args, **kwargs)
            self.active += 1
            self.a_calls.append((args, kwargs))
            self.observed_heads.append(self.git("rev-parse", "HEAD").decode().strip())
            (self.repository / "file-1.txt").unlink()
            return replace(self.a_result, session_id=self.session(2, 0))
        result = self.run_batch(implement=implement)
        self.assert_success(result, 2)
        self.assertEqual(self.b_calls[1][0][4], [("file-1.txt", b"content 1\n")])
        self.assertIn(b"deleted file mode", self.b_calls[1][0][0].tracked_diff)

    def test_one_task_also_supported_by_sequential_entry_point(self):
        self.configure(1)
        self.assert_success(self.run_batch(), 1)

    def test_failure_at_first_implementation_stops_all_later_execution(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            return replace(result, succeeded=False, exit_status=1, report=None, error_code="CODEX_FAILED")
        result = self.run_batch(implement=implement)
        self.assertEqual(result.outcome, "STOPPED", result)
        self.assertEqual(result.error_code, "CODEX_FAILED")
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])
        self.assertEqual(self.stage_calls, [])
        self.assertEqual(self.publish_calls, [])
        self.assertEqual([task["status"] for task in self.data()["tasks"]], ["stopped", "pending", "pending"])
        self.assertEqual(self.checkpoint()["publications"], [])

    def test_later_failed_tests_preserve_first_publication_and_partial_report(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            return result if self.active == 1 else replace(result, passed=False, first_failed_index=1,
                  command_results=(replace(result.command_results[0], exit_status=1),))
        result = self.run_batch(tests=tests)
        self.assertEqual(result.outcome, "PARTIALLY_COMPLETED", result)
        self.assertEqual(result.error_code, "TESTS_NOT_PASSED")
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (2, 2, 1))
        self.assertEqual((len(self.stage_calls), len(self.publish_calls)), (1, 1))
        self.assertEqual(self.checkpoint()["current_task_id"], "task-2")
        self.assertEqual(self.checkpoint()["base_head"], self.head)
        self.assertEqual(self.checkpoint()["current_head"], result.publications[0][1])
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), result.publications[0][1])
        self.assertEqual([task["status"] for task in self.data()["tasks"]], ["completed", "stopped", "pending"])
        self.assertIsNone(self.data()["tasks"][1]["review"])
        self.assertTrue((self.repository / "file-2.txt").exists())

    def test_later_implementation_failure_preserves_previous_commit(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            return result if self.active == 1 else replace(result, succeeded=False, exit_status=1,
                                                          report=None, error_code="CODEX_FAILED")
        result = self.run_batch(implement=implement)
        self.assertEqual(result.completed_task_ids, ("task-1",))
        self.assertEqual(result.error_code, "CODEX_FAILED", result)
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls)), (2, 1, 1))
        self.assertIsNone(self.data()["tasks"][1]["terminal_results"])
        self.assertEqual(self.data()["tasks"][2]["status"], "pending")

    def test_later_timeout_stops_before_review_and_next_task(self):
        def tests(*args, **kwargs):
            result = self.run_tests_fake(*args, **kwargs)
            return result if self.active == 1 else replace(result, passed=False, first_failed_index=1,
                  command_results=(replace(result.command_results[0], exit_status=None, timed_out=True,
                                           cleanup_result="taskkill_completed"),))
        output = io.StringIO()
        with redirect_stderr(output):
            result = self.run_batch(tests=tests)
        self.assertEqual(result.error_code, "TEST_TIMEOUT", result)
        self.assertEqual((len(self.a_calls), len(self.b_calls)), (2, 1))
        self.assertTrue(self.data()["tasks"][1]["terminal_results"]["command_results"][0]["timed_out"])
        self.assertEqual(result.completed_task_ids, ("task-1",))
        self.assertEqual((len(self.test_calls), len(self.stage_calls), len(self.publish_calls)), (2, 1, 1))
        self.assertEqual(result.timeout.operation, "tests")
        self.assertEqual(result.timeout.configured_seconds, 22)
        self.assertEqual(self.checkpoint()["timeout"], result.timeout.record())
        self.assertEqual(self.data()["checkpoint"]["timeout"], result.timeout.record())
        self.assertIn("STOPPED - TIMEOUT", (self.metadata / "batch-report.md").read_text(encoding="utf-8"))
        self.assertIn("Published task: task-1", output.getvalue())
        self.assertIn("Task: task-2", output.getvalue())
        self.assertEqual(self.data()["tasks"][2]["status"], "pending")
        commit = result.publications[0][1]
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), commit)
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], commit)

    def _fail_later_publication(self, action, *, ambiguous=False):
        query = git_publish._git_query
        failures = []
        def fail(repository, arguments, *args, **kwargs):
            if self.active == 2 and arguments[0] == action:
                failures.append(arguments)
                if ambiguous:
                    result = query(repository, arguments, *args, **kwargs)
                    self.assertEqual((result[0], result[2]), (0, None))
                    return None, None, "GIT_TIMEOUT"
                return 1, b"", None
            return query(repository, arguments, *args, **kwargs)
        with patch("auto_coding.git_publish._git_query", side_effect=fail):
            result = self.run_batch()
        self.assertEqual(result.outcome, "PARTIALLY_COMPLETED", result)
        self.assertEqual(result.completed_task_ids, ("task-1",))
        self.assertEqual(len(self.a_calls), 2)
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.data()["tasks"][0]["status"], "completed")
        self.assertEqual(self.data()["tasks"][1]["status"], "stopped")
        self.assertEqual(self.data()["tasks"][2]["status"], "pending")
        self.assertEqual(self.checkpoint()["current_head"], result.publications[0][1])
        return self.data()["tasks"][1]["publishing"]

    def test_later_commit_failure_stops_without_push_or_retry(self):
        result = self._fail_later_publication("commit")
        self.assertEqual(result["outcome"], "COMMIT_FAILED_OR_AMBIGUOUS")
        self.assertFalse(result["push_attempted"])
        self.assertNotEqual(self.git("diff", "--cached"), b"")

    def test_later_push_failure_preserves_actual_local_commit(self):
        result = self._fail_later_publication("push")
        self.assertEqual(result["outcome"], "PUSH_FAILED_OR_AMBIGUOUS")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), result["commit_sha"])
        self.assertNotEqual(result["commit_sha"], self.checkpoint()["current_head"])

    def test_later_ambiguous_push_is_not_retried_even_if_remote_updated(self):
        result = self._fail_later_publication("push", ambiguous=True)
        self.assertEqual(result["outcome"], "PUSH_FAILED_OR_AMBIGUOUS")
        remote = self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0]
        self.assertEqual(remote, result["commit_sha"])
        self.assertNotEqual(remote, self.checkpoint()["current_head"])

    def test_later_uncertain_review_never_stages_or_starts_next(self):
        def review(*args, **kwargs):
            result = self.review(*args, **kwargs)
            return result if self.active == 1 else replace(result, approved=False,
                verdict=replace(result.verdict, decision="UNCERTAIN", blocking_issues=("Missing evidence",)),
                reviewed_packet_sha256=None, reviewed_evidence_fingerprint=None)
        result = self.run_batch(review=review)
        self.assertEqual(result.error_code, "REVIEW_NOT_APPROVED", result)
        self.assertEqual((len(self.a_calls), len(self.b_calls), len(self.stage_calls)), (2, 2, 1))
        self.assertEqual(self.data()["tasks"][1]["review"]["model_verdict"]["decision"], "UNCERTAIN")

    def test_later_scope_cannot_include_files_approved_only_for_earlier_task(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            if self.active == 2:
                (self.repository / "file-1.txt").write_bytes(b"unauthorized later change")
            return result
        result = self.run_batch(implement=implement)
        self.assertEqual(result.error_code, "OUT_OF_SCOPE", result)
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.publish_calls)), (2, 1, 1))
        self.assertEqual((self.repository / "file-1.txt").read_bytes(), b"unauthorized later change")

    def test_dirty_state_created_between_tasks_stops_before_next_implementation(self):
        update = orchestrator.update_checkpoint
        def mutate(path, state, **kwargs):
            result = update(path, state, **kwargs)
            if state["schema_version"] == 2 and state["current_task_id"] == "task-2" and state["status"] == "running":
                (self.repository / "foreign.txt").write_bytes(b"external mutation")
            return result
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=mutate):
            result = self.run_batch()
        self.assertEqual(result.error_code, "WORKTREE_NOT_CLEAN", result)
        self.assertEqual(result.outcome, "PARTIALLY_COMPLETED")
        self.assertEqual(result.stage, "preflight")
        self.assertIsNone(result.report_error_code)
        self.assertEqual(result.completed_task_ids, ("task-1",))
        self.assertEqual((len(self.a_calls), len(self.test_calls), len(self.b_calls),
                          len(self.stage_calls), len(self.publish_calls)), (1, 1, 1, 1, 1))
        report = self.data()
        current = report["tasks"][1]
        self.assertIs(current["implementation_started"], False)
        self.assertIs(current["review_started"], False)
        self.assertEqual(current["status"], "stopped")
        for field in ("implementation", "evidence", "terminal_results", "review", "staging", "publishing"):
            self.assertIsNone(current[field], field)
        self.assertEqual(report["tasks"][0]["status"], "completed")
        self.assertEqual(report["tasks"][2]["status"], "pending")
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["status"], "stopped")
        self.assertEqual(checkpoint["current_task_id"], "task-2")
        self.assertEqual(len(checkpoint["publications"]), 1)
        self.assertIs(checkpoint["publications"][0]["finalized"], True)
        commit = result.publications[0][1]
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), commit)
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], commit)
        self.assertEqual(self.git("rev-list", "--count", f"{self.head}..HEAD").decode().strip(), "1")
        self.assertFalse((self.repository / "file-2.txt").exists())
        self.assertEqual((self.repository / "foreign.txt").read_bytes(), b"external mutation")

    def test_modified_frozen_plan_between_tasks_cannot_be_reapproved_implicitly(self):
        update = orchestrator.update_checkpoint
        def mutate(path, state, **kwargs):
            result = update(path, state, **kwargs)
            if state["schema_version"] == 2 and state["current_task_id"] == "task-2" and state["status"] == "running":
                self.frozen_path.write_bytes(self.frozen_bytes + b" ")
            return result
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=mutate):
            result = self.run_batch()
        self.assertNotEqual(result.outcome, "COMPLETED")
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(len(self.publish_calls), 1)

    def test_batch_checkpoint_write_failure_stops_without_retry(self):
        update = orchestrator.update_checkpoint
        failures = []
        def fail(path, state, **kwargs):
            if state["schema_version"] == 2 and state["publications"]:
                failures.append(state)
                raise OSError("PRIVATE_ERROR")
            return update(path, state, **kwargs)
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=fail):
            result = self.run_batch()
        self.assertEqual(result.error_code, "BATCH_CHECKPOINT_WRITE_FAILED")
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(len(result.publications), 1)
        self.assertEqual(load_checkpoint(self.metadata / "task-0001" / "batch-checkpoint.json")["status"], "completed")
        self.assertTrue((self.metadata / "task-0001" / "batch-report.json").exists())

    def test_interrupted_task_call_reports_unknown_execution_without_starting_next(self):
        def interrupt(*args, **kwargs):
            self.implement(*args, **kwargs)
            raise KeyboardInterrupt()
        result = self.run_batch(implement=interrupt)
        self.assertEqual(result.error_code, "BATCH_INTERRUPTED_STATE_UNCERTAIN", result)
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(self.test_calls, [])
        self.assertEqual(self.b_calls, [])
        self.assertEqual(self.publish_calls, [])
        current = self.data()["tasks"][0]
        self.assertIsNone(current["implementation_started"])
        self.assertIsNone(current["implementation"])
        self.assertIsNone(current["publishing"])
        self.assertEqual(self.data()["tasks"][1]["status"], "pending")
        self.assertTrue((self.repository / "file-1.txt").exists())

    def test_interrupted_batch_write_after_actual_update_is_never_retried(self):
        update = orchestrator.update_checkpoint
        interruptions = []
        def interrupt(path, state, **kwargs):
            value = update(path, state, **kwargs)
            if state["schema_version"] == 2 and state["publications"]:
                interruptions.append(state)
                raise KeyboardInterrupt()
            return value
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=interrupt):
            result = self.run_batch()
        self.assertEqual(result.error_code, "BATCH_CHECKPOINT_WRITE_FAILED", result)
        self.assertEqual(len(interruptions), 1)
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(len(self.checkpoint()["publications"]), 1)
        self.assertEqual(self.checkpoint()["current_task_id"], "task-2")
        self.assertTrue((self.metadata / "task-0001" / "batch-report.json").exists())

    def test_per_task_checkpoint_finalization_failure_retains_verified_publication(self):
        update = orchestrator.update_checkpoint
        def fail(path, state, **kwargs):
            if state["schema_version"] == 1 and state["status"] == "completed":
                raise OSError("PRIVATE_ERROR")
            return update(path, state, **kwargs)
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=fail):
            result = self.run_batch()
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(result.error_code, "CHECKPOINT_FINALIZATION_FAILED", result)
        self.assertEqual(len(result.publications), 1)
        self.assertEqual(self.checkpoint()["publications"][0]["finalized"], False)
        self.assertEqual(self.data()["tasks"][0]["publishing"]["outcome"], "PUBLISHED")
        self.assertEqual(self.data()["tasks"][0]["status"], "stopped")

    def test_per_task_report_failure_stops_with_verified_commit_in_batch_report(self):
        export = orchestrator.export_batch_report
        def fail(*args, **kwargs):
            if Path(kwargs["destination"]).name == "task-0001":
                raise OSError("PRIVATE_ERROR")
            return export(*args, **kwargs)
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=fail):
            result = self.run_batch()
        self.assertEqual(len(self.a_calls), 1)
        self.assertEqual(result.error_code, "REPORT_UNAVAILABLE", result)
        self.assertEqual(self.checkpoint()["publications"][0]["finalized"], False)
        self.assertEqual(self.data()["tasks"][0]["publishing"]["commit_sha"], result.publications[0][1])
        self.assertEqual(self.data()["tasks"][1]["status"], "pending")

    def test_final_batch_report_failure_keeps_all_published_tasks_and_completion(self):
        export = orchestrator.export_batch_report
        def fail(*args, **kwargs):
            if Path(kwargs["destination"]) == self.metadata:
                raise OSError("PRIVATE_ERROR")
            return export(*args, **kwargs)
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=fail):
            result = self.run_batch()
        self.assertNotEqual(result.outcome, "COMPLETED")
        self.assertEqual(len(result.completed_task_ids), 3)
        self.assertEqual(len(result.publications), 3)
        self.assertEqual(self.checkpoint()["status"], "completed")
        self.assertEqual(result.report_error_code, "REPORT_UNAVAILABLE")
        self.assertTrue((self.metadata / "task-0003" / "batch-report.json").exists())

    def test_invalid_approval_or_small_report_budget_prevents_all_codex_calls(self):
        for changes in ({"approved_sha256": "0" * 64}, {"report_max_bytes": 1}, {"packet_max_bytes": 0}):
            with self.subTest(changes=changes):
                result = self.run_batch(**changes)
                self.assertNotEqual(result.outcome, "COMPLETED")
                self.assertEqual(self.a_calls, [])
                self.assertFalse((self.metadata / "batch-checkpoint.json").exists())

    def test_checkpoint_capacity_is_reserved_before_any_task_runs(self):
        template = self.plan["tasks"][0]
        self.plan["tasks"] = [dict(template, task_id=f"task-{index}", order=index, depends_on=[])
                              for index in range(1, 101)]
        self.configure(100)
        result = self.run_batch()
        self.assertEqual(result.error_code, "BATCH_METADATA_TOO_LARGE", result)
        self.assertEqual(self.a_calls, [])
        self.assertFalse((self.metadata / "batch-checkpoint.json").exists())

    def test_reused_session_stops_before_later_tests(self):
        def implement(*args, **kwargs):
            result = self.implement(*args, **kwargs)
            return replace(result, session_id=self.session(1, 0))
        result = self.run_batch(implement=implement)
        self.assertEqual(result.error_code, "BATCH_SESSION_REUSED", result)
        self.assertEqual((len(self.a_calls), len(self.test_calls)), (2, 1))

    def test_existing_task_metadata_prevents_all_execution(self):
        (self.metadata / "task-0002").mkdir()
        result = self.run_batch()
        self.assertEqual(result.error_code, "METADATA_DESTINATION_EXISTS")
        self.assertEqual(self.a_calls, [])
        self.assertFalse((self.metadata / "batch-checkpoint.json").exists())

    def test_cli_runs_real_three_task_pipeline_and_returns_completed(self):
        from contextlib import redirect_stdout, redirect_stderr
        import io
        from auto_coding.cli import main
        output, errors = io.StringIO(), io.StringIO()
        with patch("auto_coding.orchestrator.run_implementation", side_effect=self.implement), \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=self.run_tests_fake), \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=self.review), \
             redirect_stdout(output), redirect_stderr(errors):
            status = main(["run", str(self.frozen_path), "--approved-sha256", self.approval,
                           "--repository", str(self.repository), "--metadata-directory", str(self.metadata),
                           "--repository-id", "example/sequential", "--git-timeout-seconds", "5",
                           "--evidence-max-bytes", "100000", "--packet-max-bytes", "200000", "--report-max-bytes", "200000"])
        self.assertEqual(status, 0, output.getvalue() + errors.getvalue())
        self.assertIn("COMPLETED", output.getvalue())
        self.assertIn("Completed tasks: 3/3", output.getvalue())
        self.assertEqual(self.checkpoint()["status"], "completed")


if __name__ == "__main__":
    unittest.main()
