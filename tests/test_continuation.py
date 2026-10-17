"""Manual continuation using disposable repositories; never live Codex/GitHub."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import stat
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding import continuation, git_publish
from auto_coding.batch_report import export_batch_report
from auto_coding.batch_state import create_checkpoint, update_checkpoint
from auto_coding.change_evidence import collect_change_evidence
from auto_coding.cli import main
from auto_coding.plan_freeze import PlanIntegrityError, freeze_approved_plan, plan_fingerprint
from auto_coding.review import ReviewResult, ReviewVerdict, build_review_packet
from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity
import test_batch_report as reports
import test_orchestrator as fixtures


def new_plan(source, base):
    plan = deepcopy(source)
    plan.update(batch_id="continued-batch", expected_branch="feature/continued", expected_head=base)
    plan["tasks"] = plan["tasks"][1:]
    for index, task in enumerate(plan["tasks"]):
        task.update(order=index + 1, depends_on=[] if index == 0 else [plan["tasks"][index - 1]["task_id"]])
    return plan


def snapshot(repository, metadata, frozen):
    return ({str(path.relative_to(repository)): path.read_bytes() for path in repository.rglob("*") if path.is_file()},
            {str(path.relative_to(metadata)): path.read_bytes() for path in metadata.rglob("*") if path.is_file()},
            frozen.read_bytes())


def invoke(arguments):
    output, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        status = main(list(map(str, arguments)))
    return status, output.getvalue(), errors.getvalue()


class ContinuationTests(unittest.TestCase):
    def test_optional_diagnostics_and_new_approved_test_contract_remain_compatible(self):
        for record in self.records[:2]:
            record["terminal_results"]["diagnostic"] = {"classification": "PASSED", "reason_code": "APPROVED_TESTS_PASSED", "protocol": None}
            for command in record["terminal_results"]["command_results"]:
                command["diagnostic"] = {"classification": "PASSED", "reason_code": "TEST_PASSED", "protocol": None}
        self.export()
        for task in self.continued["tasks"]:
            task["test_contract"] = {"inputs_complete": True, "protected_files": [], "failure_protocols": ["exit-status-v1"]}
        self.write_plan()
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        self.assertTrue(self.inspect().safe)
        prepared = self.prepare(dry_run=True)
        self.assertIsNone(prepared.error_code, prepared)
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    git = fixtures.SequentialBatchTests.git
    configure = fixtures.SequentialBatchTests.configure

    def setUp(self):
        fixtures.SequentialBatchTests.setUp(self)
        self.source_metadata = self.metadata
        self.state = dict(schema_version=2, batch_id=self.plan["batch_id"], frozen_plan_sha256=self.approval,
                          expected_branch=self.branch, base_head=self.head, status="pending", current_task_id=None,
                          stage="pending", implementer_session_id=None, reviewer_session_id=None, stop_reason_code=None,
                          task_ids=[task["task_id"] for task in self.plan["tasks"]], publications=[], current_head=self.head)
        (self.repository / "file-1.txt").write_bytes(b"published task 1\n")
        first_evidence = collect_change_evidence(self.repository, ["file-1.txt"], 100000)
        self.git("add", "--", "file-1.txt")
        raw = self.git("diff", "--cached", "--no-ext-diff", "--no-textconv", "--no-renames", "--raw", "--abbrev=40",
                       "-z", "--ita-visible-in-index", self.head, "--")
        self.git("commit", "-qm", "auto-coding: task-1")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()
        self.git("push", "-q", "origin", self.branch)
        (self.repository / "file-2.txt").write_bytes(b"unapproved failed task changes\n")
        second_evidence = collect_change_evidence(self.repository, ["file-2.txt"], 100000)
        first, second = self.record(0, first_evidence), self.record(1, second_evidence)
        first["staging"].update(staged_paths=["file-1.txt"], staged_changes_sha256=hashlib.sha256(raw).hexdigest())
        first["publishing"].update(commit_sha=self.commit, remote_sha=self.commit, observed_head_sha=self.commit)
        second.update(status="stopped", failure_stage="review", stop_reason_code="REVIEW_NOT_APPROVED", staging=None, publishing=None)
        second["review"].update(approved=False, error_code=None)
        second["review"]["model_verdict"].update(decision="UNCERTAIN", blocking_issues=["Navigation needs correction."],
                                                  checked_requirements=["Inspected navigation behavior"])
        self.records = [first, second, fixtures.orchestrator._pending_task_record(self.plan["tasks"][2])]
        self.child(1, self.head, "completed", first)
        self.child(2, self.commit, "stopped", second)
        create_checkpoint(self.source_metadata / "batch-checkpoint.json", self.state)
        self.state.update(status="stopped", current_task_id="task-2", stage="review", stop_reason_code="REVIEW_NOT_APPROVED",
                          current_head=self.commit, publications=[dict(task_id="task-1", starting_head=self.head,
                          commit_sha=self.commit, remote_sha=self.commit, finalized=True)])
        update_checkpoint(self.source_metadata / "batch-checkpoint.json", self.state)
        self.export()
        self.proposed = self.parent / "New Proposed Plan.json"
        self.continued = new_plan(self.plan, self.commit)
        self.write_plan()
        self.destination = self.parent / "Fresh Clone With Spaces"
        self.new_metadata = self.parent / "Fresh Metadata With Spaces"

    def record(self, index, evidence):
        task = self.plan["tasks"][index]
        record = reports.task(task["task_id"])
        for field in ("title", "acceptance_criteria", "permitted_files"):
            record[field] = task[field]
        for role, setting, number in (("implementer", "implementation", 0), ("reviewer", "review", 1)):
            record[role].update(model=task[setting]["model"], reasoning_effort=task[setting]["reasoning_effort"],
                                session_id=fixtures.SequentialBatchTests.session(index + 1, number), session_mode="fresh")
        record["implementation"]["session_id"] = record["implementer"]["session_id"]
        record["review"].update(session_id=record["reviewer"]["session_id"], reviewed_evidence_fingerprint=evidence.fingerprint)
        record["review"]["model_verdict"]["checked_requirements"] = task["acceptance_criteria"]
        record["evidence"].update(changed_paths=list(evidence.changed_paths), fingerprint=evidence.fingerprint)
        identities = [_command_identity(command) for command in task["test_commands"]]
        record["approved_command_identities"] = identities
        record["terminal_results"].update(commands_executed=len(identities), command_results=[
            dict(index=i, command_identity=identity, exit_status=0, elapsed_seconds=0.1, timed_out=False, cleanup_result="not_needed")
            for i, identity in enumerate(identities, 1)])
        return record

    def child(self, index, base, status, record):
        directory = self.source_metadata / f"task-{index:04d}"
        directory.mkdir()
        state = {key: value for key, value in self.state.items() if key not in {"task_ids", "publications", "current_head"}}
        state.update(schema_version=1, base_head=base)
        path = directory / "batch-checkpoint.json"
        create_checkpoint(path, state)
        state.update(status=status, stage="finished" if status == "completed" else "review",
                     current_task_id=None if status == "completed" else record["task_id"],
                     implementer_session_id=record["implementer"]["session_id"], reviewer_session_id=record["reviewer"]["session_id"],
                     stop_reason_code=record["stop_reason_code"])
        update_checkpoint(path, state, completion_verified=status == "completed")

    def export(self):
        for name in ("batch-report.json", "batch-report.md"):
            (self.source_metadata / name).unlink(missing_ok=True)
        export_batch_report(self.state, {"repository_id": "example/continuation", "started_at": None, "ended_at": None},
                            self.records, [], 200000, destination=self.source_metadata, repository_directory=self.repository)

    def write_plan(self):
        self.proposed.write_text(json.dumps(self.continued), encoding="utf-8-sig")

    def inspect(self):
        return continuation.inspect_continuation(self.frozen_path, self.approval, self.source_metadata,
                                                 self.repository, 5, 200000)

    def prepare(self, **changes):
        return continuation.prepare_continuation(self.frozen_path, self.approval, self.source_metadata, self.repository,
                    self.proposed, changes.pop("destination", self.destination), changes.pop("metadata_directory", self.new_metadata),
                    git_timeout_seconds=5, report_max_bytes=200000, **changes)

    def test_inspection_verifies_published_base_and_preserves_dirty_source(self):
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        result = self.inspect()
        self.assertTrue(result.safe, result)
        self.assertEqual(result.published_task_ids, ("task-1",))
        self.assertEqual(result.base_sha, self.commit)
        self.assertEqual(result.stopped_task_id, "task-2")
        self.assertEqual(result.reported_review_decision, "UNCERTAIN")
        self.assertEqual(result.never_started_task_ids, ("task-3",))
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    def test_dry_run_creates_nothing(self):
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        result = self.prepare(dry_run=True)
        self.assertIsNone(result.error_code, result)
        self.assertTrue(result.dry_run)
        self.assertFalse(result.prepared)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.new_metadata.exists())
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    def test_preparation_uses_remote_objects_and_fresh_paths_without_approval(self):
        (self.repository / "file-1.txt").write_bytes(b"unapproved tracked changes after stopping\n")
        self.git("add", "--", "file-2.txt")
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        proposed = self.proposed.read_bytes()
        result = self.prepare()
        self.assertTrue(result.prepared, result)
        self.assertEqual(self.at(self.destination, "rev-parse", "HEAD").decode().strip(), self.commit)
        self.assertEqual(self.at(self.destination, "branch", "--show-current").decode().strip(), self.continued["expected_branch"])
        self.assertFalse((self.destination / "file-2.txt").exists())
        self.assertFalse((self.destination / "file-3.txt").exists())
        # Git may legitimately check text out as CRLF on Windows. Verify the
        # exact committed blob as well as the checked-out text content.
        self.assertEqual(self.at(self.destination, "show", f"{self.commit}:file-1.txt"), b"published task 1\n")
        self.assertEqual((self.destination / "file-1.txt").read_text(encoding="utf-8"), "published task 1\n")
        self.assertEqual(self.at(self.destination, "status", "--porcelain"), b"")
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)
        self.assertEqual(self.proposed.read_bytes(), proposed)
        self.assertEqual(sorted(path.name for path in self.new_metadata.iterdir()), ["continuation.json"])
        continuation.check_continuation_binding(self.continued, result.plan_sha256, self.destination, self.new_metadata)
        with self.assertRaises(PlanIntegrityError):
            freeze_approved_plan(self.continued, self.approval, self.new_metadata / "frozen.json")
        self.assertFalse((self.new_metadata / "frozen.json").exists())

    def test_new_plan_cannot_reuse_batch_branch_origin_head_or_published_task(self):
        original = deepcopy(self.continued)
        for changes in ({"batch_id": self.plan["batch_id"]}, {"expected_branch": self.branch},
                        {"expected_head": self.head}, {"expected_origin_url": str(self.parent / "other.git")},
                        {"expected_branch": "main"}):
            self.continued = original | changes
            self.write_plan()
            self.assertFalse(self.prepare().prepared)
        self.continued = deepcopy(original)
        self.continued["tasks"][0]["task_id"] = "task-1"
        self.continued["tasks"][1]["depends_on"] = ["task-1"]
        self.write_plan()
        self.assertEqual(self.prepare().error_code, "PUBLISHED_TASK_REPEATED")
        self.continued = deepcopy(original)
        self.continued["tasks"][0]["permitted_files"] = ["../escape.txt"]
        self.write_plan()
        self.assertFalse(self.prepare().prepared)
        self.assertFalse(self.destination.exists())

    def test_existing_or_internal_destinations_are_never_overwritten(self):
        self.destination.mkdir()
        marker = self.destination / "existing.txt"
        marker.write_bytes(b"preserve")
        self.assertEqual(self.prepare().error_code, "NEW_DESTINATION_REQUIRED")
        self.assertEqual(marker.read_bytes(), b"preserve")
        self.assertEqual(self.prepare(destination=self.repository / "inside").error_code, "UNSAFE_CONTINUATION_LOCATION")
        self.assertEqual(self.prepare(destination=self.parent / "other", metadata_directory=self.source_metadata / "inside").error_code,
                         "UNSAFE_CONTINUATION_LOCATION")

    def test_remote_mismatch_and_missing_ref_fail_closed(self):
        query = git_publish._git_query
        def moved(repository, arguments, *args):
            if arguments[0] == "ls-remote":
                return 0, f"{'f' * 40}\trefs/heads/{self.branch}\n".encode(), None
            return query(repository, arguments, *args)
        with patch("auto_coding.git_publish._git_query", side_effect=moved):
            self.assertEqual(self.inspect().error_code, "REMOTE_HEAD_MISMATCH")
        self.at(self.remote, "update-ref", "-d", f"refs/heads/{self.branch}")
        self.assertFalse(self.inspect().safe)

    def test_tampered_checkpoint_and_commit_evidence_are_rejected(self):
        query = continuation._git_query
        def missing_delta(repository, arguments, *args):
            if arguments[:2] == ["--no-replace-objects", "diff"]:
                return 0, b"", None
            return query(repository, arguments, *args)
        with patch("auto_coding.continuation._git_query", side_effect=missing_delta):
            self.assertEqual(self.inspect().error_code, "SOURCE_COMMIT_EVIDENCE_MISMATCH")
        self.records[0]["staging"]["staged_changes_sha256"] = "0" * 64
        self.export()
        self.assertEqual(self.inspect().error_code, "SOURCE_COMMIT_EVIDENCE_MISMATCH")
        path = self.source_metadata / "batch-checkpoint.json"
        state = json.loads(path.read_bytes())
        state["stop_reason_code"] = "TAMPERED"
        path.write_text(json.dumps(state), encoding="utf-8")
        self.assertEqual(self.inspect().error_code, "SOURCE_REPORT_CHECKPOINT_MISMATCH")

    def test_ambiguous_publication_is_rejected_even_with_unchanged_remote(self):
        self.records[1]["publishing"] = dict(outcome="COMMIT_FAILED_OR_AMBIGUOUS", error_code="GIT_TIMEOUT",
            commit_attempted=True, push_attempted=False, commit_sha=None, remote_sha=None, observed_head_sha=None)
        self.export()
        self.assertEqual(self.inspect().error_code, "SOURCE_PUBLICATION_AMBIGUOUS")

    def test_inaccessible_git_and_timeout_are_sanitized(self):
        for code in ("GIT_IO_FAILED", "GIT_TIMEOUT"):
            with patch("auto_coding.git_publish._git_query", return_value=(None, None, code)):
                self.assertEqual(self.inspect().error_code, code)

    def test_clone_failure_preserves_partial_paths_without_retry(self):
        query = continuation._git_query
        calls = []
        def fail(repository, arguments, *args):
            if "clone" in arguments:
                calls.append(arguments)
                return None, None, "GIT_TIMEOUT"
            return query(repository, arguments, *args)
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        with patch("auto_coding.continuation._git_query", side_effect=fail):
            result = self.prepare()
        self.assertEqual(result.error_code, "GIT_TIMEOUT")
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.destination.is_dir())
        self.assertTrue(self.new_metadata.is_dir())
        self.assertFalse((self.new_metadata / "continuation.json").exists())
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    def test_metadata_alias_is_rechecked_after_cloning_before_reference_write(self):
        query, plain = continuation._git_query, continuation._plain_path
        cloned = False
        def observe(repository, arguments, *args):
            nonlocal cloned
            result = query(repository, arguments, *args)
            if "clone" in arguments:
                cloned = True
            return result
        def changed_alias(value):
            if cloned and Path(value) == self.new_metadata:
                raise continuation.ContinuationError("UNSAFE_CONTINUATION_PATH")
            return plain(value)
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        with patch("auto_coding.continuation._git_query", side_effect=observe), \
             patch("auto_coding.continuation._plain_path", side_effect=changed_alias):
            result = self.prepare()
        self.assertEqual(result.error_code, "UNSAFE_CONTINUATION_PATH")
        self.assertFalse((self.new_metadata / "continuation.json").exists())
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    def test_reference_rejects_changed_plan_and_unapproved_execution(self):
        result = self.prepare()
        self.assertTrue(result.prepared, result)
        changed = deepcopy(self.continued)
        changed["tasks"][0]["implementation"]["prompt"] += " changed"
        with self.assertRaisesRegex(continuation.ContinuationError, "CONTINUATION_PLAN_MISMATCH"):
            continuation.check_continuation_binding(changed, plan_fingerprint(changed), self.destination, self.new_metadata)

    def test_reports_must_match_the_frozen_plan_and_child_checkpoints(self):
        self.records[0]["acceptance_criteria"] = ["Different criterion"]
        self.records[0]["review"]["model_verdict"]["checked_requirements"] = ["Different criterion"]
        self.export()
        self.assertEqual(self.inspect().error_code, "SOURCE_REPORT_PLAN_MISMATCH")
        self.records[0]["acceptance_criteria"] = self.plan["tasks"][0]["acceptance_criteria"]
        self.records[0]["review"]["model_verdict"]["checked_requirements"] = self.plan["tasks"][0]["acceptance_criteria"]
        self.export()
        path = self.source_metadata / "task-0001" / "batch-checkpoint.json"
        child = json.loads(path.read_bytes())
        child["base_head"] = "a" * 40
        path.write_text(json.dumps(child), encoding="utf-8")
        self.assertEqual(self.inspect().error_code, "SOURCE_TASK_CHECKPOINT_MISMATCH")

    def test_unknown_execution_and_uncertain_cleanup_are_not_continuation_authority(self):
        current = self.records[1]
        current.update(implementation_started=None, review_started=False, implementation=None, terminal_results=None,
                       review=None, evidence=None)
        for role in ("implementer", "reviewer"):
            current[role].update(session_id=None, session_mode=None)
        self.export()
        self.assertEqual(self.inspect().error_code, "SOURCE_EXECUTION_UNCERTAIN")

    def test_uncertain_cleanup_and_unsafe_aliases_fail_closed(self):
        record = self.records[1]
        record.update(review=None, review_started=False, terminal_results=None,
                      failure_stage="implementation", stop_reason_code="PROCESS_TIMEOUT")
        record["reviewer"].update(session_id=None, session_mode=None)
        record["implementation"].update(succeeded=False, exit_status=None, timed_out=True,
                                        cleanup_result="taskkill_failed", error_code="PROCESS_TIMEOUT")
        self.state.update(stage="implementation", stop_reason_code="PROCESS_TIMEOUT")
        (self.source_metadata / "batch-checkpoint.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.export()
        self.assertEqual(self.inspect().error_code, "SOURCE_PROCESS_CLEANUP_UNCERTAIN")
        original = Path.lstat
        def reparse(path, *args, **kwargs):
            if path == self.new_metadata:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(continuation.ContinuationError, "UNSAFE_CONTINUATION_PATH"):
            continuation._plain_path(self.new_metadata)

    def test_existing_remote_branch_and_missing_reports_are_rejected(self):
        self.git("push", "-q", "origin", f"{self.commit}:refs/heads/{self.continued['expected_branch']}")
        self.assertEqual(self.prepare().error_code, "NEW_REMOTE_BRANCH_REQUIRED")
        (self.source_metadata / "batch-report.md").unlink()
        self.assertFalse(self.inspect().safe)

    def test_cli_binding_rejects_a_changed_approved_plan_before_codex(self):
        result = self.prepare()
        self.assertTrue(result.prepared, result)
        self.at(self.destination, "push", "-q", "origin", self.continued["expected_branch"])
        changed = deepcopy(self.continued)
        changed["tasks"][0]["implementation"]["prompt"] += " revised after preparation"
        approval = plan_fingerprint(changed)
        frozen = self.new_metadata / "frozen.json"
        freeze_approved_plan(changed, approval, frozen)
        with patch("auto_coding.orchestrator.run_implementation") as implement:
            response = invoke(["run", frozen, "--approved-sha256", approval, "--repository", self.destination,
                "--metadata-directory", self.new_metadata, "--repository-id", "example/continued", "--git-timeout-seconds", "5",
                "--evidence-max-bytes", "100000", "--packet-max-bytes", "200000", "--report-max-bytes", "200000"])
        self.assertEqual(response[0], 1, response)
        self.assertIn("CONTINUATION_PLAN_MISMATCH", response[2])
        implement.assert_not_called()
        self.assertFalse((self.new_metadata / "batch-checkpoint.json").exists())

    def test_read_only_cli_and_module_help_with_paths_containing_spaces(self):
        before = snapshot(self.repository, self.source_metadata, self.frozen_path)
        arguments = ["continuation", "inspect", "--source-plan", self.frozen_path, "--source-approved-sha256", self.approval,
                     "--source-metadata", self.source_metadata, "--repository", self.repository, "--git-timeout-seconds", "5"]
        result = invoke(arguments)
        self.assertEqual(result[0], 0, result)
        for text in (self.commit, "task-1", "task-2", "task-3", "SAFE TO PREPARE"):
            self.assertIn(text, result[1])
        self.assertIn('Reported reviewer decision: "UNCERTAIN"', result[1])
        self.assertIn("batch-report.md", result[1])
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        with tempfile.TemporaryFile(mode="w+b") as output:
            process = subprocess.run([sys.executable, "-m", "auto_coding", "continuation", "prepare", "--help"],
                stdout=output, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=environment, timeout=10, check=False)
            output.seek(0)
            self.assertEqual(process.returncode, 0)
            self.assertIn(b"--dry-run", output.read(65537))
        self.assertEqual(snapshot(self.repository, self.source_metadata, self.frozen_path), before)

    @staticmethod
    def at(repository, *arguments):
        with tempfile.TemporaryFile(mode="w+b") as output:
            subprocess.run(["git", *arguments], cwd=repository, stdout=output, stderr=subprocess.DEVNULL,
                           stdin=subprocess.DEVNULL, check=True, timeout=10)
            output.seek(0)
            return output.read(65537)


class ContinuationIntegrationTests(unittest.TestCase):
    def harness(self):
        harness = fixtures.SequentialBatchTests()
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        return harness

    def test_real_stopped_three_task_batch_continues_only_remaining_tasks(self):
        source = self.harness()
        def uncertain(*args, **kwargs):
            if source.active == 1:
                return source.review(*args, **kwargs)
            source.b_calls.append((args, kwargs))
            return ReviewResult(False, ReviewVerdict("UNCERTAIN", ("Navigation needs correction.",), (),
                ("Inspected navigation",), "MEDIUM"), None, 0, False, "not_needed", source.session(2, 1))
        stopped = source.run_batch(review=uncertain)
        self.assertEqual(stopped.completed_task_ids, ("task-1",))
        self.assertEqual((len(source.a_calls), len(source.b_calls)), (2, 2))
        before = snapshot(source.repository, source.metadata, source.frozen_path)
        base = stopped.publications[0][1]
        plan = new_plan(source.plan, base)
        proposed = source.parent / "Continued Proposed.json"
        proposed.write_text(json.dumps(plan), encoding="utf-8")
        destination, metadata = source.parent / "Clean Continuation", source.parent / "New Metadata"
        prepared = invoke(["continuation", "prepare", "--source-plan", source.frozen_path,
            "--source-approved-sha256", source.approval, "--source-metadata", source.metadata, "--repository", source.repository,
            "--plan", proposed, "--destination", destination, "--metadata-directory", metadata, "--git-timeout-seconds", "5"])
        self.assertEqual(prepared[0], 0, prepared)
        self.assertIn("PREPARED - NOT APPROVED OR EXECUTED", prepared[1])
        self.assertFalse((destination / "file-2.txt").exists())
        self.assertEqual(ContinuationTests.at(destination, "rev-parse", "HEAD").decode().strip(), base)
        self.assertEqual(snapshot(source.repository, source.metadata, source.frozen_path), before)
        # These explicit local-remote/user identity steps are not done by prepare.
        ContinuationTests.at(destination, "config", "user.name", "Continuation Tests")
        ContinuationTests.at(destination, "config", "user.email", "continued@example.invalid")
        ContinuationTests.at(destination, "push", "-q", "origin", plan["expected_branch"])
        approval = plan_fingerprint(plan)
        frozen = metadata / "frozen.json"
        with self.assertRaises(PlanIntegrityError):
            freeze_approved_plan(plan, source.approval, frozen)
        freeze_approved_plan(plan, approval, frozen)
        implementations, commands, reviews = [], [], []
        def implement(*args, **kwargs):
            task = plan["tasks"][len(implementations)]
            implementations.append((args, kwargs))
            self.assertEqual(args[0], task["implementation"]["prompt"])
            self.assertEqual(args[1:3], (task["implementation"]["model"], task["implementation"]["reasoning_effort"]))
            self.assertEqual(args[4], task["limits"]["implementation_seconds"])
            self.assertIsNone(kwargs["session_id"])
            (destination / task["permitted_files"][0]).write_bytes(f"corrected {task['task_id']}\n".encode())
            return replace(source.a_result, session_id=source.session(len(implementations) + 10, 0))
        def tests(*args):
            commands.append(args)
            return TestRunResult(True, len(args[1]), tuple(TestCommandResult(i, 0, 0.1, False, "not_needed", _command_identity(command))
                for i, command in enumerate(args[1], 1)), None)
        def review(*args, **kwargs):
            reviews.append((args, kwargs))
            packet = build_review_packet(*args)
            return ReviewResult(True, ReviewVerdict("APPROVED", (), (), tuple(args[1]), "HIGH"), None, 0, False,
                "not_needed", source.session(len(reviews) + 10, 1), hashlib.sha256(packet).hexdigest(), args[0].fingerprint)
        with patch("auto_coding.orchestrator.run_implementation", side_effect=implement), \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=tests), \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=review):
            completed = invoke(["run", frozen, "--approved-sha256", approval, "--repository", destination,
                "--metadata-directory", metadata, "--repository-id", "example/continued", "--git-timeout-seconds", "5",
                "--evidence-max-bytes", "100000", "--packet-max-bytes", "200000", "--report-max-bytes", "200000"])
        self.assertEqual(completed[0], 0, completed)
        self.assertIn("MANUAL CONTINUATION", completed[1])
        self.assertEqual((len(implementations), len(commands), len(reviews)), (2, 2, 2))
        self.assertEqual([call[0][0] for call in implementations], [task["implementation"]["prompt"] for task in plan["tasks"]])
        result = json.loads((metadata / "batch-report.json").read_bytes())
        self.assertEqual([task["task_id"] for task in result["tasks"]], ["task-2", "task-3"])
        self.assertEqual(result["checkpoint"]["base_head"], base)
        self.assertEqual(result["checkpoint"]["status"], "completed")
        self.assertEqual(ContinuationTests.at(destination, "rev-list", "--count", f"{base}..HEAD").strip(), b"2")
        self.assertEqual(snapshot(source.repository, source.metadata, source.frozen_path), before)

    def test_no_publication_can_use_only_a_verified_original_baseline(self):
        source = self.harness()
        def fail(*args, **kwargs):
            return replace(source.implement(*args, **kwargs), succeeded=False, exit_status=1, report=None,
                           error_code="PROCESS_FAILED")
        result = source.run_batch(implement=fail)
        self.assertEqual(result.publications, ())
        before = snapshot(source.repository, source.metadata, source.frozen_path)
        response = invoke(["continuation", "inspect", "--source-plan", source.frozen_path, "--source-approved-sha256", source.approval,
            "--source-metadata", source.metadata, "--repository", source.repository, "--git-timeout-seconds", "5"])
        self.assertEqual(response[0], 0, response)
        self.assertIn("No tasks were published", response[1])
        self.assertIn(source.head, response[1])
        self.assertEqual(snapshot(source.repository, source.metadata, source.frozen_path), before)


if __name__ == "__main__":
    unittest.main()
