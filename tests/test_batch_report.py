"""Deterministic batch reports and safe external export."""

from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.batch_report import (
    BatchReportError, ReportExportError, build_batch_report, export_batch_report,
)
from auto_coding.test_runner import _command_identity


_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"
_FINGERPRINT = "a" * 64
_COMMIT = "c" * 40
_COMMAND = _command_identity(["py", "-m", "unittest"])
_LIMIT = 128 * 1024


def checkpoint(status="completed"):
    return {
        "schema_version": 1, "batch_id": "batch-1", "frozen_plan_sha256": "f" * 64,
        "expected_branch": "feature/example", "base_head": "b" * 40,
        "status": status, "stage": "finished" if status == "completed" else "review",
        "current_task_id": None if status == "completed" else "task-1",
        "implementer_session_id": _A, "reviewer_session_id": _B,
        "stop_reason_code": None if status == "completed" else "REVIEW_REJECTED",
    }


def batch():
    return {"repository_id": "example/project", "started_at": "2026-10-15T10:00:00Z",
            "ended_at": "2026-10-15T10:10:00+00:00"}


def task(task_id="task-1"):
    return {
        "task_id": task_id, "title": "Implement a module", "objective": "Add the approved behavior.",
        "acceptance_criteria": ["Behavior is correct.", "Tests pass."],
        "permitted_files": ["src/example.py", "tests/test_example.py"],
        "implementer": {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "session_id": _A, "session_mode": "fresh"},
        "reviewer": {"model": "gpt-6-sol", "reasoning_effort": "high", "session_id": _B, "session_mode": "resumed"},
        "status": "completed", "failure_stage": None, "stop_reason_code": None,
        "implementation_started": True, "review_started": True,
        "implementation": {
            "succeeded": True, "exit_status": 0, "elapsed_seconds": 2.5, "timed_out": False,
            "cleanup_result": "not_needed", "error_code": None, "session_id": _A,
            "model_claims": {"summary": "Added behavior.", "approach": "Used a small function.",
                             "files_reported": ["src/example.py"], "model_reported_confidence": "HIGH", "concerns": []},
        },
        "evidence": {"passed": True, "changed_paths": ["src/example.py", "tests/test_example.py"],
                     "fingerprint": _FINGERPRINT, "error_code": None},
        "approved_command_identities": [_COMMAND],
        "terminal_results": {
            "passed": True, "commands_executed": 1, "first_failed_index": None,
            "command_results": [{"index": 1, "command_identity": _COMMAND, "exit_status": 0,
                                 "elapsed_seconds": 1.25, "timed_out": False, "cleanup_result": "not_needed"}],
        },
        "review": {
            "approved": True, "exit_status": 0, "timed_out": False, "cleanup_result": "not_needed",
            "error_code": None, "session_id": _B, "reviewed_packet_sha256": "d" * 64,
            "reviewed_evidence_fingerprint": _FINGERPRINT,
            "model_verdict": {"decision": "APPROVED", "blocking_issues": [], "nonblocking_notes": ["Consider a later simplification."],
                              "checked_requirements": ["Behavior is correct.", "Tests pass."], "model_reported_confidence": "MEDIUM"},
        },
        "staging": {"passed": True, "error_code": None, "staging_attempted": True,
                    "staged_paths": ["src/example.py", "tests/test_example.py"],
                    "staged_changes_sha256": "e" * 64, "staged_state_error_code": None},
        "publishing": {"outcome": "PUBLISHED", "error_code": None, "commit_attempted": True,
                       "push_attempted": True, "commit_sha": _COMMIT, "remote_sha": _COMMIT, "observed_head_sha": _COMMIT},
        "preserved_evidence_location": None,
    }


def stopped_task():
    result = task()
    result.update(status="stopped", failure_stage="review", stop_reason_code="REVIEW_REJECTED",
                  staging=None, publishing=None, preserved_evidence_location="metadata/evidence-1")
    result["review"].update(approved=False, error_code="REVIEW_REJECTED")
    result["review"]["model_verdict"].update(decision="REJECTED", blocking_issues=["A requirement is missing."])
    return result


def descriptive_stopped_review(decision="UNCERTAIN"):
    """Reported live failure pattern; descriptions are not verbatim criteria."""
    record = stopped_task()
    record["acceptance_criteria"] = ["Approved file bytes are exact.", "Approved tests pass.", "Only approved files change."]
    record["review"]["model_verdict"] = {
        "decision": decision,
        "blocking_issues": ["Independent confirmation of the file bytes remains unavailable"],
        "nonblocking_notes": ["Inspect <source> before any later approval"],
        "checked_requirements": ["Compared the proof file with the requested content",
                                 "Inspected the reported terminal test success",
                                 "Examined the changed file scope"],
        "model_reported_confidence": "MEDIUM",
    }
    record["review"].update(reviewed_packet_sha256=None, reviewed_evidence_fingerprint=None)
    return record


def pending_task(task_id="task-1"):
    result = task(task_id)
    result.update(status="pending", implementation_started=None, review_started=False,
                  implementation=None, evidence=None, terminal_results=None, review=None, staging=None, publishing=None)
    for role in ("implementer", "reviewer"):
        result[role].update(session_id=None, session_mode=None)
    return result


class BatchReportTests(unittest.TestCase):
    def build(self, tasks=None, state=None, metadata=None, events=None, limit=_LIMIT):
        return build_batch_report(checkpoint() if state is None else state, batch() if metadata is None else metadata,
                                  [task()] if tasks is None else tasks, [] if events is None else events, limit)

    def test_complete_batch_preserves_machine_results_and_models(self):
        report = self.build()
        data = json.loads(report.json_bytes)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["batch"]["planned_tasks"], 1)
        self.assertEqual(data["batch"]["completed_tasks"], 1)
        self.assertEqual(data["batch"]["stopped_tasks"], 0)
        recorded = data["tasks"][0]
        expected = task()
        expected["publishing"]["remote_verification"] = "verified"
        self.assertEqual(recorded, expected)
        self.assertEqual(recorded["implementer"]["model"], "gpt-6.1-sol")
        self.assertEqual(recorded["reviewer"]["reasoning_effort"], "high")
        self.assertEqual(recorded["implementer"]["session_id"], _A)
        self.assertEqual(recorded["reviewer"]["session_id"], _B)
        self.assertEqual(recorded["terminal_results"]["command_results"][0]["command_identity"], _COMMAND)
        self.assertIn(b"# Auto-Coding batch report", report.markdown_bytes)
        self.assertIn(_COMMIT.encode(), report.markdown_bytes)

    def test_sequential_partial_report_binds_completed_commits_to_durable_ledger(self):
        first, second = task(), stopped_task()
        second["task_id"] = "task-2"
        state = dict(checkpoint("stopped"), schema_version=2, current_task_id="task-2",
                     implementer_session_id=None, reviewer_session_id=None, task_ids=["task-1", "task-2"],
                     current_head=_COMMIT, publications=[dict(task_id="task-1", starting_head="b" * 40,
                     commit_sha=_COMMIT, remote_sha=_COMMIT, finalized=True)])
        data = json.loads(self.build([first, second], state).json_bytes)
        self.assertEqual(data["batch"]["completed_tasks"], 1)
        self.assertEqual(data["batch"]["stopped_tasks"], 1)
        self.assertEqual(data["checkpoint"]["base_head"], "b" * 40)
        self.assertEqual(data["checkpoint"]["publications"][0]["commit_sha"], _COMMIT)
        wrong = deepcopy(first)
        wrong["publishing"].update(commit_sha="d" * 40, remote_sha="d" * 40, observed_head_sha="d" * 40)
        with self.assertRaises(BatchReportError):
            self.build([wrong, second], state)
        with self.assertRaises(BatchReportError):
            self.build([second, first], state)

    def test_stopped_batch_records_failure_and_partial_task_without_success(self):
        report = self.build([stopped_task()], checkpoint("stopped"))
        data = json.loads(report.json_bytes)
        self.assertEqual(data["checkpoint"]["status"], "stopped")
        self.assertEqual(data["batch"]["completed_tasks"], 0)
        self.assertEqual(data["batch"]["stopped_tasks"], 1)
        self.assertEqual(data["tasks"][0]["failure_stage"], "review")
        self.assertEqual(data["tasks"][0]["preserved_evidence_location"], "metadata/evidence-1")
        self.assertIsNone(data["tasks"][0]["publishing"])
        self.assertIn(b"REVIEW\\_REJECTED", report.markdown_bytes)

    def test_nonapproved_review_descriptions_preserve_all_findings(self):
        for decision in ("UNCERTAIN", "REJECTED"):
            with self.subTest(decision=decision):
                record = descriptive_stopped_review(decision)
                original = deepcopy(record)
                data = json.loads(self.build([record], checkpoint("stopped")).json_bytes)
                self.assertEqual(data["tasks"][0], original)
                self.assertEqual(record, original)
                self.assertFalse(data["tasks"][0]["review"]["approved"])
                self.assertEqual(data["batch"]["completed_tasks"], 0)

    def test_approved_verdict_still_requires_exact_complete_unique_criteria(self):
        for accepted in (False, True):
            for checks in (["Tests pass."], ["Behavior works", "Tests pass."],
                           ["Behavior is correct.", "Tests pass.", "Tests pass."],
                           ["Tests pass.", "Tests pass."], []):
                with self.subTest(accepted=accepted, checks=checks):
                    record = task() if accepted else stopped_task()
                    record["review"]["model_verdict"].update(
                        decision="APPROVED", blocking_issues=[], checked_requirements=checks)
                    with self.assertRaises(BatchReportError):
                        self.build([record], checkpoint() if accepted else checkpoint("stopped"))

    def test_nonapproved_descriptions_retain_text_schema_and_size_bounds(self):
        for decision in ("UNCERTAIN", "REJECTED"):
            for field in ("checked_requirements", "blocking_issues", "nonblocking_notes"):
                for invalid in (["x" * 2001], [""], ["bad\x00text"], ["\ud800"], [7], ["x"] * 101):
                    with self.subTest(decision=decision, field=field, invalid_type=type(invalid[0])):
                        record = descriptive_stopped_review(decision)
                        record["review"]["model_verdict"][field] = invalid
                        with self.assertRaises(BatchReportError):
                            self.build([record], checkpoint("stopped"))
            record = descriptive_stopped_review(decision)
            report = self.build([record], checkpoint("stopped"))
            size = max(len(report.json_bytes), len(report.markdown_bytes))
            self.assertEqual(self.build([record], checkpoint("stopped"), limit=size), report)
            with self.assertRaises(BatchReportError):
                self.build([record], checkpoint("stopped"), limit=size - 1)

    def test_nonapproved_decision_cannot_be_marked_approved_or_published(self):
        for decision in ("UNCERTAIN", "REJECTED"):
            for claim in ("approval", "publication"):
                record = descriptive_stopped_review(decision)
                if claim == "approval":
                    record["review"]["approved"] = True
                else:
                    completed = task()
                    record.update(staging=completed["staging"], publishing=completed["publishing"])
                with self.subTest(decision=decision, claim=claim), self.assertRaises(BatchReportError):
                    self.build([record], checkpoint("stopped"))

    def test_multiple_tasks_preserve_supplied_approved_order(self):
        report = self.build([task("task-z"), task("task-a")])
        self.assertEqual([t["task_id"] for t in json.loads(report.json_bytes)["tasks"]], ["task-z", "task-a"])
        self.assertLess(report.markdown_bytes.index(b"## Task 1:"), report.markdown_bytes.index(b"## Task 2:"))

    def test_confidence_and_findings_are_explicit_model_claims(self):
        data = json.loads(self.build().json_bytes)
        self.assertIn("unverified model reports", data["provenance"])
        record = data["tasks"][0]
        self.assertEqual(record["implementation"]["model_claims"]["model_reported_confidence"], "HIGH")
        self.assertEqual(record["review"]["model_verdict"]["model_reported_confidence"], "MEDIUM")
        self.assertEqual(record["review"]["model_verdict"]["nonblocking_notes"], ["Consider a later simplification."])
        self.assertIn("not semantically checked for secrets", data["provenance"])

    def test_missing_information_remains_unavailable(self):
        state = checkpoint("stopped")
        state.update(stage="pending", current_task_id=None, implementer_session_id=None, reviewer_session_id=None,
                     stop_reason_code="USER_STOP")
        report = self.build([pending_task()], state, metadata={"repository_id": "example/project", "started_at": None, "ended_at": None})
        data = json.loads(report.json_bytes)
        self.assertIsNone(data["batch"]["started_at"])
        self.assertIsNone(data["tasks"][0]["terminal_results"])
        self.assertIsNone(data["tasks"][0]["implementation_started"])
        self.assertEqual(data["batch"]["completed_tasks"], 0)
        self.assertIn(b"Unavailable", report.markdown_bytes)

    def test_operational_events_accept_only_relevant_structured_fields(self):
        event = {"timestamp": "2026-10-15T10:01:00Z", "run_id": "run-1", "stage": "implementation",
                 "status": "succeeded", "batch_id": "batch-1", "task_id": "task-1", "reason_code": None}
        self.assertEqual(json.loads(self.build(events=[event]).json_bytes)["events"], [event])
        for field, value in (("batch_id", "other"), ("task_id", "other"), ("status", "approved"), ("raw_log", "SECRET")):
            with self.subTest(field=field):
                with self.assertRaises(BatchReportError):
                    self.build(events=[dict(event, **{field: value})])

    def test_publishing_attempts_failures_and_ambiguity_remain_distinct(self):
        variants = (
            {"outcome": "NOT_ATTEMPTED", "commit_attempted": False, "push_attempted": False,
             "commit_sha": None, "remote_sha": None, "observed_head_sha": None},
            {"outcome": "COMMIT_FAILED_OR_AMBIGUOUS", "commit_attempted": True, "push_attempted": False,
             "commit_sha": None, "remote_sha": None, "observed_head_sha": None},
            {"outcome": "COMMIT_CREATED_NOT_PUSHED", "commit_attempted": True, "push_attempted": False,
             "commit_sha": None, "remote_sha": None, "observed_head_sha": _COMMIT},
            {"outcome": "PUSH_FAILED_OR_AMBIGUOUS", "commit_attempted": True, "push_attempted": True,
             "commit_sha": _COMMIT, "remote_sha": None, "observed_head_sha": _COMMIT},
            {"outcome": "PUSH_FAILED_OR_AMBIGUOUS", "commit_attempted": True, "push_attempted": True,
             "commit_sha": _COMMIT, "remote_sha": "d" * 40, "observed_head_sha": _COMMIT},
            {"outcome": "PUSH_FAILED_OR_AMBIGUOUS", "commit_attempted": True, "push_attempted": True,
             "commit_sha": _COMMIT, "remote_sha": _COMMIT, "observed_head_sha": _COMMIT},
        )
        for variant in variants:
            with self.subTest(outcome=variant["outcome"], remote=variant["remote_sha"]):
                record = task()
                record.update(status="stopped", failure_stage="publishing", stop_reason_code="GIT_FAILED")
                record["publishing"] = dict(variant, error_code="GIT_FAILED")
                state = checkpoint("stopped")
                state.update(stage="publishing", stop_reason_code="GIT_FAILED")
                result = json.loads(self.build([record], state).json_bytes)["tasks"][0]["publishing"]
                self.assertEqual(result["outcome"], variant["outcome"])
                self.assertEqual(result["commit_sha"], variant["commit_sha"])
                self.assertNotEqual(result["remote_verification"], "verified")

    def test_failed_terminal_command_and_timeout_preserved(self):
        for timed_out, exit_status, cleanup in ((False, 1, "not_needed"), (True, None, "taskkill_failed")):
            with self.subTest(timeout=timed_out):
                record = stopped_task()
                record.update(failure_stage="testing", stop_reason_code="TEST_FAILED", review=None, review_started=False)
                record["reviewer"].update(session_id=None, session_mode=None)
                terminal = record["terminal_results"]
                terminal.update(passed=False, first_failed_index=1)
                terminal["command_results"][0].update(timed_out=timed_out, exit_status=exit_status, cleanup_result=cleanup)
                state = checkpoint("stopped")
                state.update(stage="testing", stop_reason_code="TEST_FAILED", reviewer_session_id=None)
                report = self.build([record], state)
                self.assertEqual(json.loads(report.json_bytes)["tasks"][0]["terminal_results"], terminal)

    def test_deterministic_output_and_input_not_modified(self):
        tasks, state, metadata = [task()], checkpoint(), batch()
        before = deepcopy((tasks, state, metadata))
        first = self.build(tasks, state, metadata)
        second = self.build(tasks, dict(reversed(list(state.items()))), dict(reversed(list(metadata.items()))))
        self.assertEqual(first.json_bytes, second.json_bytes)
        self.assertEqual(first.markdown_bytes, second.markdown_bytes)
        self.assertEqual((tasks, state, metadata), before)

    def test_exact_size_boundary_and_one_byte_below(self):
        report = self.build()
        boundary = max(len(report.json_bytes), len(report.markdown_bytes))
        self.assertEqual(self.build(limit=boundary), report)
        with self.assertRaises(BatchReportError):
            self.build(limit=boundary - 1)

    def test_small_budget_rejects_before_full_serialization(self):
        import auto_coding.batch_report as module
        original = module.json.dumps

        def observe(value, *args, **kwargs):
            if isinstance(value, dict) and "tasks" in value:
                self.fail("full report serialization happened before budget rejection")
            return original(value, *args, **kwargs)

        with patch("auto_coding.batch_report.json.dumps", side_effect=observe):
            with self.assertRaises(BatchReportError):
                self.build([task(f"task-{i}") for i in range(30)], limit=2000)

    def test_invalid_positive_limits(self):
        for limit in (True, 0, -1, 1.5, "1000"):
            with self.subTest(limit=limit):
                with self.assertRaises(BatchReportError):
                    self.build(limit=limit)

    def test_markdown_escapes_html_headings_links_code_and_newlines(self):
        record = task()
        record["title"] = '<script>alert(1)</script>\n# forged [link](https://bad) `code` | table'
        record["implementation"]["model_claims"]["summary"] = "Unicode café 日本語 **bold**"
        report = self.build([record])
        markdown = report.markdown_bytes.decode("utf-8")
        self.assertNotIn("<script>", markdown)
        self.assertNotIn("\n# forged", markdown)
        self.assertNotIn("[link](https://bad)", markdown)
        self.assertIn("&lt;script&gt;", markdown)
        self.assertIn("café 日本語", markdown)
        self.assertIn("\\*\\*bold\\*\\*", markdown)
        self.assertEqual(json.loads(report.json_bytes)["tasks"][0]["title"], record["title"])

    def test_raw_private_fields_rejected_at_each_record_level(self):
        for location in ((), ("implementation",), ("implementation", "model_claims"), ("review",), ("publishing",)):
            for field in ("raw_prompt", "transcript", "credentials", "stdout", "diff", "environment"):
                with self.subTest(location=location, field=field):
                    record = task()
                    nested = record
                    for name in location:
                        nested = nested[name]
                    nested[field] = "SECRET"
                    with self.assertRaises(BatchReportError) as error:
                        self.build([record])
                    self.assertNotIn("SECRET", str(error.exception))

    def test_malformed_and_inconsistent_inputs_fail_closed(self):
        changes = (
            (("status",), "succeeded"), (("task_id",), 5), (("title",), "x" * 2001),
            (("implementation", "elapsed_seconds"), float("nan")),
            (("implementation", "exit_status"), True),
            (("implementation", "succeeded"), "yes"),
            (("implementer", "session_id"), "invalid"),
            (("reviewer", "session_id"), _A), (("reviewer", "session_mode"), None),
            (("implementation_started",), False), (("review_started",), False),
            (("terminal_results", "passed"), False),
            (("terminal_results", "commands_executed"), 2),
            (("terminal_results", "first_failed_index"), 1),
            (("review", "approved"), False),
            (("review", "model_verdict", "blocking_issues"), ["Blocker"]),
            (("review", "model_verdict", "checked_requirements"), ["Tests pass."]),
            (("review", "reviewed_evidence_fingerprint"), "0" * 64),
            (("publishing", "remote_sha"), "d" * 40),
            (("publishing", "push_attempted"), False),
            (("publishing", "observed_head_sha"), None),
            (("staging", "staged_paths"), []),
            (("evidence", "changed_paths"), ["outside.txt"]),
            (("permitted_files",), ["../unsafe"]),
        )
        for location, value in changes:
            with self.subTest(location=location):
                record = task()
                nested = record
                for name in location[:-1]:
                    nested = nested[name]
                nested[location[-1]] = value
                with self.assertRaises(BatchReportError):
                    self.build([record])

    def test_missing_success_evidence_does_not_become_completed(self):
        for field in ("implementation", "evidence", "terminal_results", "review", "staging", "publishing"):
            with self.subTest(field=field):
                record = task()
                record[field] = None
                with self.assertRaises(BatchReportError):
                    self.build([record])

    def test_unbound_direct_approval_is_not_publish_authorization(self):
        for field in ("reviewed_packet_sha256", "reviewed_evidence_fingerprint"):
            record = task()
            record["review"][field] = None
            with self.assertRaises(BatchReportError):
                self.build([record])

    def test_checkpoint_and_task_consistency(self):
        for changes in ({"schema_version": 2}, {"status": "running", "stage": "review", "current_task_id": "task-1"},
                        {"current_task_id": "missing"}, {"stage": "testing"}, {"stop_reason_code": "OTHER"}):
            with self.subTest(changes=changes):
                state = dict(checkpoint("stopped"), **changes)
                with self.assertRaises(BatchReportError):
                    self.build([stopped_task()], state)
        with self.assertRaises(BatchReportError):
            self.build([stopped_task()])
        with self.assertRaises(BatchReportError):
            self.build([task(), task()])

    def test_timestamps_and_repository_identifiers_are_validated(self):
        for metadata in (
            dict(batch(), repository_id="https://user:password@example.com/repo"),
            dict(batch(), started_at="2026-10-15T10:00:00"),
            dict(batch(), started_at="2026-10-15T10:00:00+02:00"),
            dict(batch(), ended_at="2026-10-14T10:00:00Z"),
        ):
            with self.assertRaises(BatchReportError):
                self.build(metadata=metadata)

    def test_remote_observation_without_a_push_is_inconsistent(self):
        record = stopped_task()
        record.update(failure_stage="publishing", stop_reason_code="GIT_FAILED")
        record["publishing"] = {
            "outcome": "NOT_ATTEMPTED", "error_code": "GIT_FAILED", "commit_attempted": False,
            "push_attempted": False, "commit_sha": None, "remote_sha": _COMMIT, "observed_head_sha": None,
        }
        state = checkpoint("stopped")
        state.update(stage="publishing", stop_reason_code="GIT_FAILED")
        with self.assertRaises(BatchReportError):
            self.build([record], state)

    def test_unicode_exact_boundaries_and_bounded_array_counts(self):
        record = task()
        record["objective"] = "日本語 café " * 10
        report = self.build([record])
        size = max(len(report.json_bytes), len(report.markdown_bytes))
        self.assertEqual(self.build([record], limit=size), report)
        with self.assertRaises(BatchReportError):
            self.build([record], limit=size - 1)
        with self.assertRaises(BatchReportError):
            self.build([task(f"task-{i}") for i in range(101)])


class BatchReportExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / "target"
        self.repository.mkdir()
        (self.repository / ".git").mkdir()
        (self.repository / ".git" / "HEAD").write_bytes(b"ref: refs/heads/feature/example\n")
        (self.repository / "source.txt").write_bytes(b"unchanged source")
        self.destination = self.root / "metadata"
        self.destination.mkdir()

    def export(self, destination=None):
        return export_batch_report(checkpoint(), batch(), [task()], [], _LIMIT,
                                   destination=self.destination if destination is None else destination,
                                   repository_directory=self.repository)

    def test_export_writes_exact_expected_files_and_does_not_modify_target(self):
        before = {p.relative_to(self.repository): p.read_bytes() for p in self.repository.rglob("*") if p.is_file()}
        expected = build_batch_report(checkpoint(), batch(), [task()], [], _LIMIT)
        with patch("auto_coding.batch_report.os.fsync", wraps=os.fsync) as sync:
            paths = self.export()
            self.assertEqual(sync.call_count, 2)
        self.assertEqual(paths, (self.destination / "batch-report.json", self.destination / "batch-report.md"))
        self.assertEqual(paths[0].read_bytes(), expected.json_bytes)
        self.assertEqual(paths[1].read_bytes(), expected.markdown_bytes)
        self.assertEqual(set(self.destination.iterdir()), set(paths))
        after = {p.relative_to(self.repository): p.read_bytes() for p in self.repository.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_uncertain_stopped_review_exports_complete_json_and_escaped_markdown(self):
        record, state = descriptive_stopped_review(), checkpoint("stopped")
        expected = build_batch_report(state, batch(), [record], [], _LIMIT)
        paths = export_batch_report(state, batch(), [record], [], _LIMIT,
                                    destination=self.destination, repository_directory=self.repository)
        self.assertEqual(paths[0].read_bytes(), expected.json_bytes)
        self.assertEqual(paths[1].read_bytes(), expected.markdown_bytes)
        data = json.loads(paths[0].read_bytes())
        self.assertEqual(data["checkpoint"], state)
        self.assertEqual(data["tasks"][0], record)
        self.assertTrue(data["tasks"][0]["terminal_results"]["passed"])
        self.assertEqual(data["tasks"][0]["review"]["session_id"], _B)
        self.assertIsNone(data["tasks"][0]["publishing"])
        markdown = paths[1].read_text(encoding="utf-8")
        for text in ("UNCERTAIN", "MEDIUM", *record["review"]["model_verdict"]["blocking_issues"],
                     *record["review"]["model_verdict"]["checked_requirements"]):
            self.assertIn(text, markdown)
        self.assertIn("&lt;source&gt;", markdown)
        self.assertNotIn("<source>", markdown)
        with self.assertRaises(ReportExportError):
            export_batch_report(state, batch(), [record], [], _LIMIT,
                                destination=self.destination, repository_directory=self.repository)
        self.assertEqual(paths[0].read_bytes(), expected.json_bytes)
        self.assertEqual(paths[1].read_bytes(), expected.markdown_bytes)

    def test_existing_either_report_is_never_overwritten(self):
        for name in ("batch-report.json", "batch-report.md"):
            existing = self.destination / name
            existing.write_bytes(b"original")
            with self.assertRaises(ReportExportError):
                self.export()
            self.assertEqual(existing.read_bytes(), b"original")
            self.assertEqual(list(self.destination.iterdir()), [existing])
            existing.unlink()

    def test_repository_and_nested_destinations_are_rejected(self):
        child = self.repository / "metadata"
        child.mkdir()
        for destination in (self.repository, child):
            with self.assertRaises(BatchReportError):
                self.export(destination)
        self.assertEqual(list(child.iterdir()), [])

    def test_resolved_repository_alias_cannot_export_inside_target(self):
        # A '..' alias is deterministic on Windows and exercises resolution.
        with self.assertRaises(BatchReportError):
            self.export(self.destination / ".." / "target")

    def test_second_publication_failure_rolls_back_first_report(self):
        link = os.link
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            self.assertEqual(Path(source).parent, self.destination)
            self.assertTrue(Path(source).read_bytes())
            if calls == 2:
                raise OSError(errno.EACCES, "uncontrolled error")
            link(source, destination)

        with patch("auto_coding.batch_report.os.link", side_effect=fail_second):
            with self.assertRaises(ReportExportError) as error:
                self.export()
        self.assertEqual(error.exception.partial_paths, ())
        self.assertNotIn("uncontrolled", str(error.exception))
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_sync_failure_cleans_temporary_files(self):
        with patch("auto_coding.batch_report.os.fsync", side_effect=OSError(errno.EIO, "failed")):
            with self.assertRaises(ReportExportError):
                self.export()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_rollback_failure_explicitly_reports_partial_export(self):
        link, unlink = os.link, Path.unlink
        calls = 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError(errno.EACCES, "failed")
            link(source, destination)

        def refuse_report(path, *args, **kwargs):
            if path.name == "batch-report.json":
                raise OSError(errno.EACCES, "failed rollback")
            return unlink(path, *args, **kwargs)

        with patch("auto_coding.batch_report.os.link", side_effect=fail_second), patch.object(Path, "unlink", refuse_report):
            with self.assertRaises(ReportExportError) as error:
                self.export()
        self.assertEqual(error.exception.partial_paths, (self.destination / "batch-report.json",))
        self.assertEqual(list(self.destination.iterdir()), [self.destination / "batch-report.json"])

    def test_invalid_input_stops_before_creating_files(self):
        invalid = task()
        invalid["terminal_results"] = None
        with self.assertRaises(BatchReportError):
            export_batch_report(checkpoint(), batch(), [invalid], [], _LIMIT,
                                destination=self.destination, repository_directory=self.repository)
        self.assertEqual(list(self.destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
