"""Execute approved tasks sequentially through independent review and publication.

The supplied approval fingerprint is a trusted caller assertion. Checks observe
snapshots, not locks or containment; live Codex sandbox enforcement is a separate
verification. Evidence substeps use their enclosing implementation/testing/review
checkpoint stage, preserving D14a's schema. No resume or recovery is attempted.
"""

from dataclasses import dataclass, field
import base64
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
import time

from auto_coding.batch_state import CheckpointError, create_checkpoint, load_checkpoint, update_checkpoint, validate_checkpoint_capacity
from auto_coding.batch_report import BatchReportError, ReportExportError, build_batch_report, export_batch_report
from auto_coding.change_evidence import (
    ChangeEvidence, _external_temporary_directory, _safe_location, collect_change_evidence,
)
from auto_coding.git_publish import (
    _REPOSITORY_ENVIRONMENT, _StageFailure, _git_query, _review_snapshot, check_publish_preflight,
    GitPublishResult, GitStageResult, publish_reviewed_changes, stage_reviewed_changes,
)
from auto_coding.integrations.codex import ImplementationOutcome, ImplementationReport, run_implementation
from auto_coding.plan_freeze import PlanIntegrityError, load_frozen_plan, plan_fingerprint
from auto_coding.plan_preflight import preflight_frozen_plan, _preflight_verified_plan
from auto_coding.review import ReviewResult, _evidence_fingerprint, _packet_json_size, build_review_packet, run_review_packet
from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity, run_approved_tests
from auto_coding.operations import TimeoutDiagnostic, show_timeout


@dataclass(frozen=True)
class FirstTaskReviewResult:
    outcome: str
    stage: str
    error_code: str | None = None
    checkpoint_error_code: str | None = None
    batch_id: str | None = None
    task_id: str | None = None
    plan_sha256: str | None = None
    implementer_session_id: str | None = None
    reviewer_session_id: str | None = None
    implementation_started: bool = False
    review_started: bool = False
    repository_directory: Path | None = field(default=None, repr=False)
    checkpoint_path: Path | None = field(default=None, repr=False)
    expected_branch: str | None = field(default=None, repr=False)
    expected_head: str | None = field(default=None, repr=False)
    expected_origin_url: str | None = field(default=None, repr=False)
    permitted_files: tuple[str, ...] = field(default=(), repr=False)
    implementation: ImplementationOutcome | None = field(default=None, repr=False)
    terminal_results: TestRunResult | None = field(default=None, repr=False)
    evidence: ChangeEvidence | None = field(default=None, repr=False)
    review: ReviewResult | None = field(default=None, repr=False)
    reviewed_packet: bytes | None = field(default=None, repr=False)
    timeout: TimeoutDiagnostic | None = None

    @property
    def ready_for_publication(self) -> bool:
        return self.outcome == "READY_FOR_PUBLICATION"


class _Stop(Exception):
    def __init__(self, code, *, timeout=None):
        super().__init__(code)
        self.timeout = timeout


def _git_stop(code, seconds, operation="git_verification"):
    return _Stop(code, timeout=TimeoutDiagnostic(operation, seconds) if code == "GIT_TIMEOUT" else None)


class _CheckpointFailure(Exception):
    pass


def _code(value, fallback):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value) else fallback


def _session(value):
    if not isinstance(value, str) or len(value) != 36:
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.int != 0 and str(parsed) == value.lower()


def _snapshot(repository, plan, task, timeout, max_bytes):
    state = check_publish_preflight(repository, plan["expected_branch"], plan["expected_head"],
                                    plan["expected_origin_url"], timeout)
    if not state.passed:
        raise _git_stop(_code(state.error_code, "GIT_STATE_UNVERIFIED"), timeout)
    evidence = collect_change_evidence(repository, task["permitted_files"], max_bytes, timeout_seconds=timeout)
    if not isinstance(evidence, ChangeEvidence) or evidence.passed is not True:
        raise _git_stop(_code(getattr(evidence, "error_code", None), "EVIDENCE_INVALID"), timeout)
    if (evidence.error_code is not None or not isinstance(evidence.changed_paths, tuple)
            or not evidence.changed_paths or not isinstance(evidence.tracked_diff, bytes)
            or not isinstance(evidence.untracked_files, tuple)):
        raise _Stop("EVIDENCE_INVALID")
    if not set(evidence.changed_paths) <= set(task["permitted_files"]):
        raise _Stop("OUT_OF_SCOPE")
    if evidence.fingerprint != _evidence_fingerprint(evidence):
        raise _Stop("EVIDENCE_INVALID")
    for relative in evidence.changed_paths:
        target = _safe_location(repository, relative)
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue  # Tracked deletions are represented completely in the diff.
        if not stat.S_ISREG(mode):
            raise _Stop("UNSAFE_PATH")
    return evidence


def _context(repository, plan, evidence, timeout, max_bytes):
    """Read full changed files; deleted files use their verified base blob."""
    result, total = [], 0
    for relative in evidence.changed_paths:
        total += len(relative.encode("utf-8"))
        remaining = max_bytes - total
        if remaining < 0:
            raise _Stop("CONTEXT_TOO_LARGE")
        target = _safe_location(repository, relative)
        try:
            before = target.lstat()
        except FileNotFoundError:
            temporary = _external_temporary_directory(repository)
            if temporary is None:
                raise _Stop("UNSAFE_TEMP_LOCATION")
            environment = os.environ.copy()
            if any(name in environment for name in _REPOSITORY_ENVIRONMENT):
                raise _Stop("UNSAFE_GIT_ENVIRONMENT")
            environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
            code, contents, error = _git_query(
                repository, ["cat-file", "blob", f"{plan['expected_head']}:{relative}"],
                timeout, temporary, environment, remaining,
            )
            if error or code != 0:
                if error == "GIT_TIMEOUT":
                    raise _git_stop(error, timeout)
                raise _Stop("CONTEXT_TOO_LARGE" if error == "GIT_OUTPUT_TOO_LARGE" else "CONTEXT_UNAVAILABLE")
        else:
            if not stat.S_ISREG(before.st_mode):
                raise _Stop("UNSAFE_PATH")
            if before.st_size > remaining:
                raise _Stop("CONTEXT_TOO_LARGE")
            with target.open("rb") as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise _Stop("UNSAFE_PATH")
                contents = source.read(remaining + 1)
                closed = os.fstat(source.fileno())
            _safe_location(repository, relative)
            after = target.lstat()
            signatures = {(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) for s in (before, opened, closed, after)}
            if len(signatures) != 1 or len(contents) != before.st_size:
                raise _Stop("CONTEXT_CHANGED")
        total += len(contents)
        if total > max_bytes:
            raise _Stop("CONTEXT_TOO_LARGE")
        result.append((relative, contents))
    return result


def execute_first_task_through_review(
    frozen_plan_path: str | Path, approved_sha256: str, repository_directory: str | Path,
    checkpoint_path: str | Path, git_timeout_seconds: float,
    evidence_max_bytes: int, packet_max_bytes: int,
) -> FirstTaskReviewResult:
    """Review only the first task, retaining the original single-task interface."""
    return _execute_task_through_review(
        frozen_plan_path, approved_sha256, repository_directory, checkpoint_path,
        git_timeout_seconds, evidence_max_bytes, packet_max_bytes)


def _task_preflight(frozen, approval, repository, timeout, position):
    if position is None:
        return preflight_frozen_plan(frozen, approval, repository, timeout)
    try:
        plan = load_frozen_plan(frozen, approval)
    except (PlanIntegrityError, OSError):
        raise _Stop("PLAN_INVALID_OR_UNAPPROVED") from None
    if plan["schema_version"] != 2 or plan_fingerprint(plan) != approval:
        raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
    return _preflight_verified_plan(plan, approval, repository, timeout, position[1])


def _execute_task_through_review(
    frozen_plan_path, approved_sha256, repository_directory, checkpoint_path,
    git_timeout_seconds, evidence_max_bytes, packet_max_bytes, *, position=None, prior_sessions=(), prior_publications=(),
) -> FirstTaskReviewResult:
    """Return an in-memory handoff for D16b; never stage, publish or complete.

All execution settings come exclusively from the approved schema-v2 plan.
evidence_max_bytes also bounds total selected context bytes and path names.
packet_max_bytes bounds the canonical packet, not the frozen instruction text.
Successful checkpoints remain running at publishing. A checkpoint failure stops
all further execution and is never retried. Ignored files follow Git semantics.
"""
    phase = "preflight"
    values = {}
    state = None
    destination = None
    created = False
    preflight = None

    def remember_timeout(diagnostic):
        if diagnostic is not None and values.get("timeout") is None:
            values["timeout"] = diagnostic
            show_timeout(diagnostic, values.get("batch_id") or getattr(preflight, "batch_id", None),
                         values.get("task_id") or (f"task-number-{position[0] + 1}" if position else None),
                         prior_publications)

    def finish(error=None, checkpoint_error=None, ready=False):
        outcome = "READY_FOR_PUBLICATION" if ready else (
            "STOPPED" if values.get("implementation_started") else "STOPPED_BEFORE_IMPLEMENTATION"
        )
        return FirstTaskReviewResult(outcome, phase, error, checkpoint_error, **values)

    def persist(**changes):
        nonlocal state
        candidate = dict(state, **changes)
        try:
            update_checkpoint(destination, candidate)
        except (OSError, CheckpointError):
            raise _CheckpointFailure from None
        state = candidate

    try:
        if any(type(limit) is not int or limit <= 0 for limit in (evidence_max_bytes, packet_max_bytes)):
            raise _Stop("INVALID_MAX_BYTES")
        preflight = _task_preflight(frozen_plan_path, approved_sha256, repository_directory, git_timeout_seconds, position)
        if not preflight.ready:
            raise _git_stop(_code(preflight.error_code, "PREFLIGHT_FAILED"), git_timeout_seconds, "git_preflight")
        try:
            plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        except PlanIntegrityError:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED") from None
        except OSError:
            raise _Stop("PLAN_IO_FAILED") from None
        if plan["schema_version"] != 2 or plan_fingerprint(plan) != approved_sha256:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
        index = 0 if position is None else position[0]
        starting_head = plan["expected_head"] if position is None else position[1]
        task = plan["tasks"][index]
        if task["order"] != index + 1:
            raise _Stop("FIRST_TASK_INVALID")
        git_state = {"expected_branch": plan["expected_branch"], "expected_head": starting_head,
                     "expected_origin_url": plan["expected_origin_url"]}
        repository = Path(repository_directory).resolve(strict=True)
        requested_destination = Path(checkpoint_path)
        if os.path.lexists(requested_destination):
            raise _Stop("CHECKPOINT_EXISTS")
        destination = requested_destination.resolve()
        if destination == repository or repository in destination.parents:
            raise _Stop("UNSAFE_CHECKPOINT_LOCATION")
        values.update(batch_id=plan["batch_id"], task_id=task["task_id"], plan_sha256=approved_sha256,
                      repository_directory=repository, checkpoint_path=destination,
                      expected_branch=plan["expected_branch"], expected_head=starting_head,
                      expected_origin_url=plan["expected_origin_url"], permitted_files=tuple(task["permitted_files"]))
        state = {"schema_version": 1, "batch_id": plan["batch_id"], "frozen_plan_sha256": approved_sha256,
                 "expected_branch": plan["expected_branch"], "base_head": starting_head,
                 "status": "pending", "current_task_id": None, "stage": "pending",
                 "implementer_session_id": None, "reviewer_session_id": None, "stop_reason_code": None}
        try:
            create_checkpoint(destination, state)
        except FileExistsError:
            raise _Stop("CHECKPOINT_EXISTS") from None
        except (OSError, CheckpointError):
            raise _Stop("CHECKPOINT_CREATE_FAILED") from None
        created = True
        # Recheck critical starting state immediately before the first invocation.
        preflight = _task_preflight(frozen_plan_path, approved_sha256, repository, git_timeout_seconds, position)
        if not preflight.ready:
            raise _git_stop(_code(preflight.error_code, "PREFLIGHT_FAILED"), git_timeout_seconds, "git_preflight")
        prompt = task["review"]["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or any(0xD800 <= ord(c) <= 0xDFFF for c in prompt):
            raise _Stop("REVIEW_INSTRUCTIONS_INVALID")
        phase = "implementation"
        persist(status="running", current_task_id=task["task_id"], stage="implementation")
        settings = task["implementation"]
        values["implementation_started"] = True
        implementation = run_implementation(settings["prompt"], settings["model"], settings["reasoning_effort"],
                                            repository, task["limits"]["implementation_seconds"], session_id=None)
        values["implementation"] = implementation
        if isinstance(implementation, ImplementationOutcome) and implementation.timed_out is True:
            remember_timeout(TimeoutDiagnostic("implementation", task["limits"]["implementation_seconds"],
                                               implementation.elapsed_seconds))
        if isinstance(implementation, ImplementationOutcome) and _session(implementation.session_id):
            values["implementer_session_id"] = implementation.session_id
            persist(implementer_session_id=implementation.session_id)
            if implementation.session_id.lower() in prior_sessions:
                raise _Stop("BATCH_SESSION_REUSED")
        if isinstance(implementation, ImplementationOutcome) and implementation.timed_out is True:
            raise _Stop("PROCESS_TIMEOUT")
        if (not isinstance(implementation, ImplementationOutcome) or implementation.succeeded is not True
                or type(implementation.exit_status) is not int or implementation.exit_status != 0
                or implementation.timed_out is not False or implementation.cleanup_result != "not_needed"
                or implementation.error_code is not None or not _session(implementation.session_id)
                or not isinstance(implementation.report, ImplementationReport)
                or type(implementation.elapsed_seconds) not in (int, float)
                or not math.isfinite(implementation.elapsed_seconds) or implementation.elapsed_seconds < 0):
            raise _Stop(_code(getattr(implementation, "error_code", None), "IMPLEMENTATION_INVALID"))
        phase = "evidence"
        values["evidence"] = _snapshot(repository, git_state, task, git_timeout_seconds, evidence_max_bytes)
        phase = "testing"
        persist(stage="testing")
        tests = run_approved_tests(repository, task["test_commands"], task["limits"]["test_seconds"])
        values["terminal_results"] = tests
        if isinstance(tests, TestRunResult) and isinstance(tests.command_results, tuple):
            for command in tests.command_results:
                if isinstance(command, TestCommandResult) and command.timed_out is True:
                    raise _Stop("TEST_TIMEOUT", timeout=TimeoutDiagnostic(
                        "tests", task["limits"]["test_seconds"], command.elapsed_seconds))
        evidence = _snapshot(repository, git_state, task, git_timeout_seconds, evidence_max_bytes)
        values["evidence"] = evidence
        if not isinstance(tests, TestRunResult) or tests.passed is not True:
            if isinstance(tests, TestRunResult) and isinstance(tests.command_results, tuple):
                if any(getattr(result, "cleanup_result", None) != "not_needed" for result in tests.command_results):
                    raise _Stop("TEST_CLEANUP_UNCERTAIN")
            raise _Stop("TESTS_NOT_PASSED")
        context = _context(repository, git_state, evidence, git_timeout_seconds, evidence_max_bytes)
        if _snapshot(repository, git_state, task, git_timeout_seconds, evidence_max_bytes) != evidence:
            raise _Stop("EVIDENCE_CHANGED")
        phase = "review"
        persist(stage="review")
        try:
            packet = build_review_packet(evidence, task["acceptance_criteria"], task["test_commands"], tests, context, packet_max_bytes)
        except ValueError:
            raise _Stop("REVIEW_PACKET_INVALID") from None
        values["reviewed_packet"] = packet
        settings = task["review"]
        values["review_started"] = True
        review_started_at = time.monotonic()
        review = run_review_packet(evidence, task["acceptance_criteria"], task["test_commands"], tests, context, packet_max_bytes,
                                   model=settings["model"], reasoning_effort=settings["reasoning_effort"],
                                   repository_directory=repository, timeout_seconds=task["limits"]["review_seconds"],
                                   session_id=None, review_prompt=settings["prompt"])
        values["review"] = review
        if isinstance(review, ReviewResult) and review.timed_out is True:
            remember_timeout(TimeoutDiagnostic("review", task["limits"]["review_seconds"],
                                               time.monotonic() - review_started_at))
        if isinstance(review, ReviewResult) and _session(review.session_id):
            values["reviewer_session_id"] = review.session_id
            if review.session_id.lower() == implementation.session_id.lower():
                raise _Stop("REVIEW_SESSION_NOT_INDEPENDENT")
            persist(reviewer_session_id=review.session_id)
            if review.session_id.lower() in prior_sessions:
                raise _Stop("BATCH_SESSION_REUSED")
        if isinstance(review, ReviewResult) and review.timed_out is True:
            raise _Stop("PROCESS_TIMEOUT")
        if not isinstance(review, ReviewResult) or review.approved is not True:
            raise _Stop(_code(getattr(review, "error_code", None), "REVIEW_NOT_APPROVED"))
        if _review_snapshot(packet, review, packet_max_bytes) != evidence:
            raise _Stop("REVIEW_BINDING_MISMATCH")
        if (_snapshot(repository, git_state, task, git_timeout_seconds, evidence_max_bytes) != evidence
                or _context(repository, git_state, evidence, git_timeout_seconds, evidence_max_bytes) != context):
            raise _Stop("EVIDENCE_CHANGED_AFTER_REVIEW")
        phase = "publishing"
        persist(stage="publishing")
        return finish(ready=True)
    except _CheckpointFailure:
        return finish("CHECKPOINT_WRITE_FAILED", "CHECKPOINT_WRITE_FAILED")
    except Exception as error:
        reason = _code(str(error), "EXECUTION_FAILED") if isinstance(error, (_Stop, _StageFailure)) else "EXECUTION_FAILED"
        remember_timeout(getattr(error, "timeout", None))
        checkpoint_error = None
        if created:
            try:
                persist(status="stopped", stop_reason_code=reason,
                        **({"timeout": values["timeout"].record()} if values.get("timeout") is not None else {}))
            except _CheckpointFailure:
                checkpoint_error = "CHECKPOINT_WRITE_FAILED"
        return finish(reason, checkpoint_error)


@dataclass(frozen=True)
class SingleTaskPublishResult:
    """Publication truth is independent of checkpoint/report finalization.

    Invocation flags record gate calls; only the D13 result observes mutations.
    commit_sha and remote_sha are copied only from observed valid D13 SHA fields.
    checkpoint_finalized means completed state was written and read back.
    stage retains the primary failure stage when a later report is attempted.
    """

    outcome: str
    stage: str
    error_code: str | None = None
    checkpoint_error_code: str | None = None
    report_error_code: str | None = None
    published: bool = False
    checkpoint_finalized: bool = False
    staging_invoked: bool = False
    publishing_invoked: bool = False
    reports_exported: bool = False
    commit_sha: str | None = None
    remote_sha: str | None = None
    staging: GitStageResult | None = field(default=None, repr=False)
    publishing: GitPublishResult | None = field(default=None, repr=False)
    report_paths: tuple[Path, ...] = field(default=(), repr=False)
    partial_report_paths: tuple[Path, ...] = field(default=(), repr=False)
    timeout: TimeoutDiagnostic | None = None


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def _publication_handoff(handoff, plan, approved_sha256, evidence_limit, packet_limit, *, position=None):
    """Bind the entire retained packet and result metadata to the frozen task."""
    task = plan["tasks"][0 if position is None else position[0]]
    starting_head = plan["expected_head"] if position is None else position[1]
    if (not isinstance(handoff, FirstTaskReviewResult) or handoff.outcome != "READY_FOR_PUBLICATION"
            or handoff.stage != "publishing" or handoff.error_code is not None
            or handoff.checkpoint_error_code is not None or handoff.timeout is not None or handoff.implementation_started is not True
            or handoff.review_started is not True or handoff.batch_id != plan["batch_id"]
            or handoff.task_id != task["task_id"] or handoff.plan_sha256 != approved_sha256
            or handoff.expected_branch != plan["expected_branch"] or handoff.expected_head != starting_head
            or handoff.expected_origin_url != plan["expected_origin_url"]
            or handoff.permitted_files != tuple(task["permitted_files"])
            or not _session(handoff.implementer_session_id) or not _session(handoff.reviewer_session_id)
            or handoff.implementer_session_id.lower() == handoff.reviewer_session_id.lower()):
        raise _Stop("HANDOFF_INVALID")
    implementation = handoff.implementation
    if (not isinstance(implementation, ImplementationOutcome) or implementation.succeeded is not True
            or type(implementation.exit_status) is not int or implementation.exit_status != 0
            or implementation.timed_out is not False or implementation.cleanup_result != "not_needed"
            or implementation.error_code is not None or implementation.session_id != handoff.implementer_session_id
            or not isinstance(implementation.report, ImplementationReport)
            or type(implementation.elapsed_seconds) not in (int, float)
            or not math.isfinite(implementation.elapsed_seconds) or implementation.elapsed_seconds < 0):
        raise _Stop("IMPLEMENTATION_INVALID")
    claims = implementation.report
    if (claims.confidence not in ("LOW", "MEDIUM", "HIGH")
            or any(not isinstance(value, str) or not value.strip() or len(value) > 2000
                   for value in (claims.summary, claims.approach))
            or not isinstance(claims.files_reported, tuple) or len(claims.files_reported) > 100
            or any(not isinstance(value, str) or not value.strip() or len(value) > 512 for value in claims.files_reported)
            or not isinstance(claims.concerns, tuple) or len(claims.concerns) > 100
            or any(not isinstance(value, str) or not value.strip() or len(value) > 2000 for value in claims.concerns)):
        raise _Stop("IMPLEMENTATION_INVALID")
    evidence = _review_snapshot(handoff.reviewed_packet, handoff.review, packet_limit)
    if evidence != handoff.evidence or handoff.review.session_id != handoff.reviewer_session_id:
        raise _Stop("REVIEW_BINDING_MISMATCH")
    if not evidence.changed_paths or not set(evidence.changed_paths) <= set(task["permitted_files"]):
        raise _Stop("OUT_OF_SCOPE")
    size = len(evidence.tracked_diff) + sum(len(path.encode("utf-8")) for path in evidence.changed_paths)
    size += sum(len(item.contents) for item in evidence.untracked_files)
    if size > evidence_limit:
        raise _Stop("EVIDENCE_TOO_LARGE")
    data = json.loads(handoff.reviewed_packet)
    context = [(item["path"], base64.b64decode(item["contents_base64"], validate=True))
               for item in data["caller_supplied_review_inputs"]["selected_code_context"]["files"]]
    if build_review_packet(evidence, task["acceptance_criteria"], task["test_commands"],
                           handoff.terminal_results, context, packet_limit) != handoff.reviewed_packet:
        raise _Stop("FROZEN_TASK_PACKET_MISMATCH")


def _report_task(handoff, task, checkpoint, staged, publishing, publishing_invoked,
                 *, record_unattempted=True):
    """Explicit safe projection: never dump adapter objects, prompts or bytes."""
    implementation, tests, review = handoff.implementation, handoff.terminal_results, handoff.review
    claims = implementation.report if isinstance(implementation, ImplementationOutcome) else None
    verdict = review.verdict if isinstance(review, ReviewResult) else None
    complete = checkpoint["status"] == "completed"
    pending = checkpoint["stage"] == "pending"
    record = {
        "task_id": task["task_id"], "title": task["title"],
        "objective": "Not separately recorded in the frozen plan; see title and acceptance criteria.",
        "acceptance_criteria": task["acceptance_criteria"], "permitted_files": task["permitted_files"],
        "implementer": {"model": task["implementation"]["model"], "reasoning_effort": task["implementation"]["reasoning_effort"],
                        "session_id": handoff.implementer_session_id,
                        "session_mode": "fresh" if handoff.implementation_started else None},
        "reviewer": {"model": task["review"]["model"], "reasoning_effort": task["review"]["reasoning_effort"],
                     "session_id": handoff.reviewer_session_id,
                     "session_mode": "fresh" if handoff.review_started else None},
        "status": "completed" if complete else "pending" if pending else "stopped",
        "failure_stage": None if complete or pending else checkpoint["stage"],
        "stop_reason_code": None if pending else checkpoint["stop_reason_code"],
        "implementation_started": handoff.implementation_started, "review_started": handoff.review_started,
        "implementation": {
            "succeeded": implementation.succeeded, "exit_status": implementation.exit_status,
            "elapsed_seconds": implementation.elapsed_seconds, "timed_out": implementation.timed_out,
            "cleanup_result": implementation.cleanup_result, "error_code": implementation.error_code,
            "session_id": implementation.session_id,
            "model_claims": {"summary": claims.summary, "approach": claims.approach,
                             "files_reported": list(claims.files_reported), "model_reported_confidence": claims.confidence,
                             "concerns": list(claims.concerns)} if claims is not None else None,
        } if isinstance(implementation, ImplementationOutcome) else None,
        "evidence": {"passed": handoff.evidence.passed, "changed_paths": list(handoff.evidence.changed_paths),
                     "fingerprint": handoff.evidence.fingerprint, "error_code": handoff.evidence.error_code}
                    if isinstance(handoff.evidence, ChangeEvidence) else None,
        "approved_command_identities": [_command_identity(command) for command in task["test_commands"]],
        "terminal_results": {
            "passed": tests.passed, "commands_executed": tests.commands_executed, "first_failed_index": tests.first_failed_index,
            "command_results": [{"index": result.index, "command_identity": result.command_identity,
                                 "exit_status": result.exit_status, "elapsed_seconds": result.elapsed_seconds,
                                 "timed_out": result.timed_out, "cleanup_result": result.cleanup_result}
                                for result in tests.command_results],
        } if isinstance(tests, TestRunResult) else None,
        "review": {"approved": review.approved, "exit_status": review.exit_status, "timed_out": review.timed_out,
                   "cleanup_result": review.cleanup_result, "error_code": review.error_code, "session_id": review.session_id,
                   "reviewed_packet_sha256": review.reviewed_packet_sha256,
                   "reviewed_evidence_fingerprint": review.reviewed_evidence_fingerprint,
                   "model_verdict": {"decision": verdict.decision, "blocking_issues": list(verdict.blocking_issues),
                                     "nonblocking_notes": list(verdict.nonblocking_notes),
                                     "checked_requirements": list(verdict.checked_requirements), "model_reported_confidence": verdict.confidence}
                                    if verdict is not None else None} if isinstance(review, ReviewResult) else None,
        "staging": None, "publishing": None, "preserved_evidence_location": None,
    }
    if staged is not None:
        record["staging"] = {"passed": staged.passed, "error_code": staged.error_code,
                             "staging_attempted": staged.staging_attempted,
                             "staged_paths": None if staged.staged_paths is None else list(staged.staged_paths),
                             "staged_changes_sha256": staged.staged_changes_sha256,
                             "staged_state_error_code": staged.staged_state_error_code}
    if publishing is not None:
        record["publishing"] = {"outcome": publishing.outcome, "error_code": publishing.error_code,
                                "commit_attempted": publishing.commit_attempted, "push_attempted": publishing.push_attempted,
                                "commit_sha": publishing.commit_sha, "remote_sha": publishing.remote_sha,
                                "observed_head_sha": publishing.observed_head_sha}
    elif not publishing_invoked and record_unattempted:
        # Known control flow, not a fabricated successful Git result. When a
        # gate was invoked but returned no result, these observations stay None.
        record["publishing"] = {"outcome": "NOT_ATTEMPTED", "error_code": checkpoint["stop_reason_code"],
                                "commit_attempted": False, "push_attempted": False,
                                "commit_sha": None, "remote_sha": None, "observed_head_sha": None}
    return record


@dataclass(frozen=True)
class StoppedBatchReportResult:
    reports_exported: bool = False
    error_code: str | None = None
    report_paths: tuple[Path, ...] = field(default=(), repr=False)
    partial_report_paths: tuple[Path, ...] = field(default=(), repr=False)


def export_stopped_batch_report(
    handoff: FirstTaskReviewResult, frozen_plan_path: str | Path, approved_sha256: str,
    metadata_directory: str | Path, repository_identifier: str, report_max_bytes: int,
) -> StoppedBatchReportResult:
    """Report an actual terminal D16a stop without changing Git or checkpoint state.

Unavailable observations stay None. Invalid/inconsistent observations fail report
validation rather than being repaired. No checkpoint is manufactured or retried.
"""
    try:
        if (not isinstance(handoff, FirstTaskReviewResult)
                or handoff.outcome not in ("STOPPED", "STOPPED_BEFORE_IMPLEMENTATION")
                or handoff.error_code is None):
            raise _Stop("STOPPED_HANDOFF_INVALID")
        plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        if (plan["schema_version"] != 2 or plan_fingerprint(plan) != approved_sha256
                or len(plan["tasks"]) != 1):
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
        task = plan["tasks"][0]
        if handoff.checkpoint_path is None:
            raise _Stop("REPORT_CHECKPOINT_UNAVAILABLE")
        repository = Path(handoff.repository_directory).resolve(strict=True)
        checkpoint_path = Path(handoff.checkpoint_path).resolve(strict=True)
        if checkpoint_path.is_relative_to(repository):
            raise _Stop("UNSAFE_CHECKPOINT_LOCATION")
        state = load_checkpoint(checkpoint_path)
        if state["status"] != "stopped":
            raise _Stop("REPORT_CHECKPOINT_NOT_TERMINAL")
        expected = {"batch_id": plan["batch_id"], "frozen_plan_sha256": approved_sha256,
                    "expected_branch": plan["expected_branch"], "base_head": plan["expected_head"],
                    "implementer_session_id": handoff.implementer_session_id,
                    "reviewer_session_id": handoff.reviewer_session_id,
                    "stop_reason_code": handoff.error_code}
        if (any(state[name] != value for name, value in expected.items())
                or state["current_task_id"] not in (None, task["task_id"])
                or handoff.batch_id != plan["batch_id"] or handoff.task_id != task["task_id"]
                or handoff.plan_sha256 != approved_sha256):
            raise _Stop("REPORT_CHECKPOINT_MISMATCH")
        record = _report_task(handoff, task, state, None, None, False, record_unattempted=False)
        paths = export_batch_report(
            state, {"repository_id": repository_identifier, "started_at": None, "ended_at": None},
            [record], [], report_max_bytes, destination=metadata_directory, repository_directory=repository,
        )
        return StoppedBatchReportResult(True, report_paths=paths)
    except ReportExportError as error:
        return StoppedBatchReportResult(error_code="REPORT_EXPORT_FAILED", partial_report_paths=tuple(error.partial_paths[:4]))
    except BatchReportError:
        return StoppedBatchReportResult(error_code="REPORT_INPUT_INVALID_OR_TOO_LARGE")
    except _Stop as error:
        return StoppedBatchReportResult(error_code=_code(str(error), "REPORT_FAILED"))
    except KeyboardInterrupt:
        return StoppedBatchReportResult(error_code="REPORT_EXPORT_INTERRUPTED")
    except (OSError, CheckpointError, PlanIntegrityError, ValueError, TypeError, AttributeError):
        return StoppedBatchReportResult(error_code="REPORT_UNAVAILABLE")


def publish_single_task_batch(
    handoff: FirstTaskReviewResult, frozen_plan_path: str | Path, approved_sha256: str,
    git_timeout_seconds: float, evidence_max_bytes: int, packet_max_bytes: int, report_max_bytes: int,
    metadata_directory: str | Path, repository_identifier: str,
) -> SingleTaskPublishResult:
    """Publish and finalize an approved one-task batch without retries."""
    return _publish_task(handoff, frozen_plan_path, approved_sha256, git_timeout_seconds,
                         evidence_max_bytes, packet_max_bytes, report_max_bytes,
                         metadata_directory, repository_identifier)


def _publish_task(
    handoff, frozen_plan_path, approved_sha256, git_timeout_seconds, evidence_max_bytes,
    packet_max_bytes, report_max_bytes, metadata_directory, repository_identifier, *, position=None, prior_publications=(),
) -> SingleTaskPublishResult:
    """Complete one approved task using only D13 mutations and D14 finalization.

Commit message policy: 'auto-coding: <frozen-task-id>'. No execution settings
may be substituted. Approval provenance and adapter observations remain trusted
inputs, not independently authenticated history. D13's shared capture budget is
max(evidence_max_bytes, packet_max_bytes); retained evidence must fit its own limit.
Timestamps/events unavailable from D16a remain None/empty. A failed checkpoint
write is not retried; reports require an actual terminal checkpoint and may thus
be unavailable. Reporting failure never changes verified publication truth.
"""
    outcome, phase, reason = "STOPPED_BEFORE_STAGING", "validation", None
    values = {}
    plan = state = repository = checkpoint_path = directory = None
    authenticated = storage_failed = False

    def persist(*, complete=False, **changes):
        nonlocal state, storage_failed
        candidate = dict(state, **changes)
        try:
            if load_checkpoint(checkpoint_path) != state:
                raise CheckpointError("checkpoint changed")
            update_checkpoint(checkpoint_path, candidate, completion_verified=complete)
            if load_checkpoint(checkpoint_path) != candidate:
                raise CheckpointError("checkpoint write not verified")
        except (OSError, CheckpointError):
            storage_failed = True
            raise _CheckpointFailure from None
        state = candidate

    try:
        if any(type(limit) is not int or limit <= 0 for limit in (evidence_max_bytes, packet_max_bytes, report_max_bytes)):
            raise _Stop("INVALID_MAX_BYTES")
        try:
            valid_timeout = (type(git_timeout_seconds) in (int, float) and math.isfinite(git_timeout_seconds)
                             and git_timeout_seconds > 0)
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise _Stop("INVALID_TIMEOUT")
        try:
            plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        except (PlanIntegrityError, OSError):
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED") from None
        if plan["schema_version"] != 2 or plan_fingerprint(plan) != approved_sha256:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
        if position is None and len(plan["tasks"]) != 1:
            raise _Stop("SINGLE_TASK_BATCH_REQUIRED")
        index = 0 if position is None else position[0]
        starting_head = plan["expected_head"] if position is None else position[1]
        task = plan["tasks"][index]
        if task["order"] != index + 1:
            raise _Stop("FIRST_TASK_INVALID")
        _publication_handoff(handoff, plan, approved_sha256, evidence_max_bytes, packet_max_bytes, position=position)
        repository = Path(handoff.repository_directory).resolve(strict=True)
        if repository != handoff.repository_directory:
            raise _Stop("HANDOFF_REPOSITORY_INVALID")
        checkpoint_path = Path(handoff.checkpoint_path).resolve(strict=True)
        if checkpoint_path == repository or repository in checkpoint_path.parents:
            raise _Stop("UNSAFE_CHECKPOINT_LOCATION")
        try:
            state = load_checkpoint(checkpoint_path)
        except (OSError, CheckpointError):
            raise _Stop("CHECKPOINT_INVALID") from None
        expected = {"schema_version": 1, "batch_id": plan["batch_id"], "frozen_plan_sha256": approved_sha256,
                    "expected_branch": plan["expected_branch"], "base_head": starting_head,
                    "status": "running", "current_task_id": task["task_id"], "stage": "publishing",
                    "implementer_session_id": handoff.implementer_session_id, "reviewer_session_id": handoff.reviewer_session_id,
                    "stop_reason_code": None}
        if state != expected:
            raise _Stop("CHECKPOINT_HANDOFF_MISMATCH")
        authenticated = True
        if (not isinstance(repository_identifier, str) or len(repository_identifier) > 256
                or not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository_identifier)):
            raise _Stop("REPORT_REPOSITORY_IDENTIFIER_INVALID")
        try:
            directory = Path(metadata_directory).resolve(strict=True)
        except (OSError, ValueError, TypeError):
            raise _Stop("REPORT_DIRECTORY_INVALID") from None
        if not directory.is_dir() or directory == repository or repository in directory.parents:
            raise _Stop("UNSAFE_REPORT_LOCATION")
        if any(os.path.lexists(directory / name) for name in ("batch-report.json", "batch-report.md")):
            raise _Stop("REPORTS_ALREADY_EXIST")
        phase = "staging"
        persist(stage="staging")
        values["staging_invoked"] = True
        outcome = "STAGING_FAILED_OR_UNCERTAIN"
        staged = stage_reviewed_changes(repository, plan["expected_branch"], starting_head,
                                        plan["expected_origin_url"], git_timeout_seconds, task["permitted_files"],
                                        handoff.reviewed_packet, handoff.review, max(evidence_max_bytes, packet_max_bytes))
        values["staging"] = staged if isinstance(staged, GitStageResult) else None
        if isinstance(staged, GitStageResult) and staged.timeout is not None:
            values["timeout"] = staged.timeout
            show_timeout(staged.timeout, plan["batch_id"], task["task_id"], prior_publications)
        if (not isinstance(staged, GitStageResult) or staged.passed is not True or staged.error_code is not None or staged.timeout is not None
                or staged.staging_attempted is not True or staged.staged_state_error_code is not None
                or staged.staged_paths != handoff.evidence.changed_paths
                or not isinstance(staged.staged_changes_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", staged.staged_changes_sha256)):
            raise _Stop(_code(getattr(staged, "error_code", None), "STAGING_UNVERIFIED"))
        outcome = "STOPPED_AFTER_STAGING"
        phase = "publishing"
        persist(stage="publishing")
        values["publishing_invoked"] = True
        outcome = "PUBLICATION_AMBIGUOUS"
        publishing = publish_reviewed_changes(repository, plan["expected_branch"], starting_head,
                                             plan["expected_origin_url"], git_timeout_seconds, staged,
                                             handoff.reviewed_packet, handoff.review, max(evidence_max_bytes, packet_max_bytes),
                                             f"auto-coding: {task['task_id']}")
        values["publishing"] = publishing if isinstance(publishing, GitPublishResult) else None
        if isinstance(publishing, GitPublishResult) and publishing.timeout is not None:
            values["timeout"] = publishing.timeout
            show_timeout(publishing.timeout, plan["batch_id"], task["task_id"], prior_publications)
        if not isinstance(publishing, GitPublishResult):
            raise _Stop("PUBLICATION_UNVERIFIED")
        if _sha(publishing.commit_sha):
            values["commit_sha"] = publishing.commit_sha
        if _sha(publishing.remote_sha):
            values["remote_sha"] = publishing.remote_sha
        if publishing.outcome != "PUBLISHED":
            if publishing.outcome in {"COMMIT_FAILED_OR_AMBIGUOUS", "COMMIT_CREATED_NOT_PUSHED", "PUSH_FAILED_OR_AMBIGUOUS"}:
                outcome = publishing.outcome
            elif publishing.outcome == "NOT_ATTEMPTED":
                outcome = "STOPPED_AFTER_STAGING"
            raise _Stop(_code(publishing.error_code, "PUBLICATION_UNVERIFIED"))
        if (publishing.error_code is not None or publishing.timeout is not None or publishing.commit_attempted is not True or publishing.push_attempted is not True
                or not _sha(publishing.commit_sha) or publishing.commit_sha != publishing.remote_sha
                or publishing.observed_head_sha != publishing.commit_sha):
            raise _Stop("PUBLICATION_UNVERIFIED")
        values["published"] = True
        outcome, phase = "PUBLISHED_CHECKPOINT_FAILED", "checkpoint"
        persist(complete=True, status="completed", stage="finished", current_task_id=None, stop_reason_code=None)
        values["checkpoint_finalized"] = True
        outcome = "PUBLISHED_REPORT_FAILED"
    except _CheckpointFailure:
        reason = "CHECKPOINT_FINALIZATION_FAILED" if values.get("published") else "CHECKPOINT_WRITE_FAILED"
        values["checkpoint_error_code"] = reason
    except Exception as error:
        reason = _code(str(error), "FINALIZATION_FAILED") if isinstance(error, (_Stop, _StageFailure)) else "FINALIZATION_FAILED"
    if reason is not None and authenticated and not storage_failed and not values.get("published"):
        try:
            persist(status="stopped", stop_reason_code=reason,
                    **({"timeout": values["timeout"].record()} if values.get("timeout") is not None else {}))
        except _CheckpointFailure:
            values["checkpoint_error_code"] = "CHECKPOINT_WRITE_FAILED"
    if authenticated:
        try:
            terminal = load_checkpoint(checkpoint_path)
            if any(terminal[name] != expected[name] for name in (
                "batch_id", "frozen_plan_sha256", "expected_branch", "base_head", "implementer_session_id", "reviewer_session_id",
            )):
                raise _Stop("REPORT_CHECKPOINT_MISMATCH")
            if terminal["status"] not in ("stopped", "completed"):
                raise _Stop("REPORT_CHECKPOINT_NOT_TERMINAL")
            if terminal["status"] == "completed" and not values.get("published"):
                raise _Stop("REPORT_CHECKPOINT_MISMATCH")
            if directory is None or directory == repository or repository in directory.parents:
                raise _Stop("REPORT_DIRECTORY_INVALID")
            report_task = _report_task(handoff, task, terminal, values.get("staging"), values.get("publishing"),
                                       values.get("publishing_invoked", False))
            metadata = {"repository_id": repository_identifier, "started_at": None, "ended_at": None}
            if reason is None:
                phase = "reporting"
            build_batch_report(terminal, metadata, [report_task], [], report_max_bytes)
            paths = export_batch_report(terminal, metadata, [report_task], [], report_max_bytes,
                                        destination=directory, repository_directory=repository)
            values.update(reports_exported=True, report_paths=paths)
            if values.get("published") and values.get("checkpoint_finalized"):
                outcome = "PUBLISHED_FINALIZED"
        except ReportExportError as error:
            values["report_error_code"] = "REPORT_EXPORT_FAILED"
            values["partial_report_paths"] = tuple(error.partial_paths[:4])
        except BatchReportError:
            values["report_error_code"] = "REPORT_INPUT_INVALID_OR_TOO_LARGE"
        except KeyboardInterrupt:
            values["report_error_code"] = "REPORT_EXPORT_INTERRUPTED"
        except _Stop as error:
            values["report_error_code"] = _code(str(error), "REPORT_FAILED")
        except (OSError, CheckpointError, ValueError, TypeError, AttributeError):
            values["report_error_code"] = "REPORT_UNAVAILABLE"
    if reason is None and values.get("report_error_code") is not None:
        reason = values["report_error_code"]
    return SingleTaskPublishResult(outcome, phase, reason, **values)


@dataclass(frozen=True)
class SequentialBatchResult:
    outcome: str
    stage: str
    error_code: str | None = None
    checkpoint_error_code: str | None = None
    report_error_code: str | None = None
    completed_task_ids: tuple[str, ...] = ()
    # Verified publications include a task whose later metadata finalization failed.
    publications: tuple[tuple[str, str, str], ...] = ()
    current_task_id: str | None = None
    checkpoint_path: Path | None = field(default=None, repr=False)
    report_paths: tuple[Path, ...] = field(default=(), repr=False)
    partial_report_paths: tuple[Path, ...] = field(default=(), repr=False)
    timeout: TimeoutDiagnostic | None = None


def _pending_task_record(task):
    return _report_task(FirstTaskReviewResult("NOT_STARTED", "pending"), task,
                        {"status": "stopped", "stage": "pending", "stop_reason_code": None},
                        None, None, False, record_unattempted=False)


def _check_report_history(records, limit):
    if _packet_json_size(records, limit) > limit:
        raise _Stop("REPORT_HISTORY_TOO_LARGE")


def execute_sequential_batch(
    frozen_plan_path, approved_sha256, repository_directory, metadata_directory,
    repository_identifier, git_timeout_seconds, evidence_max_bytes, packet_max_bytes, report_max_bytes,
) -> SequentialBatchResult:
    """Execute one approved plan in order, stopping on the first uncertain result.

The v2 batch checkpoint retains the original approved base and a verified SHA
chain. Each task has an ordinary v1 checkpoint and reports in task-NNNN/. Fresh
A/B sessions and the original frozen file are used throughout; no derived plans,
resume, retry or repair are performed. All resource limits remain in force.
"""
    phase, reason = "preflight", None
    state = path = directory = repository = None
    created = storage_failed = False
    records, completed, publications = [], [], []
    paths = partial_paths = ()
    checkpoint_error = report_error = None
    current = None
    handoff = published = None
    sessions = set()
    execution_invoked = False
    timeout_detail = None

    def persist(candidate, *, complete=False):
        nonlocal state, storage_failed
        try:
            if load_checkpoint(path) != state:
                raise CheckpointError("batch checkpoint changed")
            update_checkpoint(path, candidate, completion_verified=complete)
            if load_checkpoint(path) != candidate:
                raise CheckpointError("batch checkpoint write not verified")
        except (OSError, CheckpointError, KeyboardInterrupt):
            storage_failed = True
            raise _CheckpointFailure from None
        state = candidate

    try:
        if any(type(limit) is not int or limit <= 0 for limit in (evidence_max_bytes, packet_max_bytes, report_max_bytes)):
            raise _Stop("INVALID_MAX_BYTES")
        preflight = preflight_frozen_plan(frozen_plan_path, approved_sha256, repository_directory, git_timeout_seconds)
        if preflight.ready is not True or preflight.error_code is not None:
            timeout_detail = getattr(preflight, "timeout", None)
            if timeout_detail is not None:
                show_timeout(timeout_detail, preflight.batch_id)
            raise _Stop(_code(preflight.error_code, "PREFLIGHT_FAILED"))
        plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        if plan["schema_version"] != 2 or plan_fingerprint(plan) != approved_sha256:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
        repository = Path(repository_directory).resolve(strict=True)
        frozen = Path(frozen_plan_path).resolve(strict=True)
        if frozen.is_relative_to(repository) or Path(frozen_plan_path).absolute().is_relative_to(repository):
            raise _Stop("UNSAFE_FROZEN_PLAN_LOCATION")
        requested = Path(metadata_directory).absolute()
        directory = requested.resolve(strict=True)
        if (requested != directory or ".." in requested.parts or not directory.is_dir()
                or directory.is_relative_to(repository)):
            raise _Stop("UNSAFE_METADATA_DIRECTORY")
        for component in (*directory.parents, directory):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise _Stop("UNSAFE_METADATA_DIRECTORY")
        if (not isinstance(repository_identifier, str) or len(repository_identifier) > 256
                or not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository_identifier)):
            raise _Stop("REPORT_REPOSITORY_IDENTIFIER_INVALID")
        path = directory / "batch-checkpoint.json"
        tasks = plan["tasks"]
        if len(tasks) > 100:  # Existing report schema limit; never truncate a plan.
            raise _Stop("BATCH_METADATA_TOO_LARGE")
        for target in (path, directory / "batch-report.json", directory / "batch-report.md",
                       *(directory / f"task-{i:04d}" for i in range(1, len(tasks) + 1))):
            if os.path.lexists(target):
                raise _Stop("METADATA_DESTINATION_EXISTS")
        state = dict(schema_version=2, batch_id=plan["batch_id"], frozen_plan_sha256=approved_sha256,
                     expected_branch=plan["expected_branch"], base_head=plan["expected_head"], status="pending",
                     current_task_id=None, stage="pending", implementer_session_id=None, reviewer_session_id=None,
                     stop_reason_code=None, task_ids=[task["task_id"] for task in tasks], publications=[],
                     current_head=plan["expected_head"])
        try:
            validate_checkpoint_capacity(state)
        except CheckpointError:
            raise _Stop("BATCH_METADATA_TOO_LARGE") from None
        for task in tasks:
            record = _pending_task_record(task)
            _check_report_history(record, report_max_bytes)
            records.append(record)
            _check_report_history(records, report_max_bytes)
        # Validate representability before Codex or checkpoint creation. This is
        # a capacity check, never an exported or stored execution observation.
        metadata = {"repository_id": repository_identifier, "started_at": None, "ended_at": None}
        build_batch_report(dict(state, status="stopped", stop_reason_code="CAPACITY_CHECK"),
                           metadata, records, [], report_max_bytes)
        create_checkpoint(path, state)
        created = True
        for index, task in enumerate(tasks):
            handoff = published = None
            execution_invoked = False
            current, phase = task["task_id"], "implementation"
            starting_head = state["current_head"]
            persist(dict(state, status="running", current_task_id=current, stage="implementation"))
            child = directory / f"task-{index + 1:04d}"
            child.mkdir()  # No overwrite and no automatic reuse of task artifacts.
            execution_invoked = True
            handoff = _execute_task_through_review(
                frozen_plan_path, approved_sha256, repository, child / "batch-checkpoint.json",
                git_timeout_seconds, evidence_max_bytes, packet_max_bytes, position=(index, starting_head),
                prior_sessions=sessions, prior_publications=tuple(item[0] for item in publications))
            timeout_detail = handoff.timeout
            if (handoff.outcome != "READY_FOR_PUBLICATION" or handoff.error_code is not None
                    or handoff.checkpoint_error_code is not None or handoff.timeout is not None):
                phase = handoff.stage
                raise _Stop(_code(handoff.error_code, "REVIEW_HANDOFF_INVALID"))
            phase = "publishing"
            published = _publish_task(
                handoff, frozen_plan_path, approved_sha256, git_timeout_seconds, evidence_max_bytes,
                packet_max_bytes, report_max_bytes, child, repository_identifier, position=(index, starting_head),
                prior_publications=tuple(item[0] for item in publications))
            if not isinstance(published, SingleTaskPublishResult):
                raise _Stop("PUBLICATION_UNVERIFIED")
            timeout_detail = published.timeout
            observed = published.publishing
            if published.published is True and (
                    not isinstance(observed, GitPublishResult) or observed.outcome != "PUBLISHED"
                    or observed.error_code is not None or observed.commit_attempted is not True
                    or observed.push_attempted is not True or not _sha(observed.commit_sha)
                    or observed.commit_sha != observed.remote_sha or observed.observed_head_sha != observed.commit_sha
                    or observed.commit_sha != published.commit_sha or observed.remote_sha != published.remote_sha):
                raise _Stop("PUBLICATION_UNVERIFIED")
            finalized = (published.outcome == "PUBLISHED_FINALIZED" and published.error_code is None
                         and published.checkpoint_error_code is None and published.report_error_code is None
                         and published.published is True and published.checkpoint_finalized is True
                         and published.reports_exported is True and published.staging_invoked is True
                         and published.publishing_invoked is True
                         and published.report_paths == (child / "batch-report.json", child / "batch-report.md"))
            entries = state["publications"]
            if published.published is True:
                if not _sha(published.commit_sha) or published.commit_sha != published.remote_sha:
                    raise _Stop("PUBLICATION_UNVERIFIED")
                publications.append((current, published.commit_sha, published.remote_sha))
                entries = [*entries, dict(task_id=current, starting_head=starting_head,
                                         commit_sha=published.commit_sha, remote_sha=published.remote_sha,
                                         finalized=finalized)]
            if not finalized:
                phase = published.stage
                # Keep verified publication truth even when later metadata failed.
                candidate = dict(state, publications=entries,
                                 current_head=entries[-1]["commit_sha"] if entries else state["base_head"],
                                 status="stopped", stage=phase if phase in {"staging", "publishing"} else "publishing",
                                 stop_reason_code=_code(published.error_code, "PUBLICATION_UNVERIFIED"))
                if timeout_detail is not None:
                    candidate["timeout"] = timeout_detail.record()
                persist(candidate)
                raise _Stop(candidate["stop_reason_code"])
            completed.append(current)
            sessions.update((handoff.implementer_session_id.lower(), handoff.reviewer_session_id.lower()))
            last = index == len(tasks) - 1
            persist(dict(state, publications=entries, current_head=published.commit_sha,
                         status="completed" if last else "running", stage="finished" if last else "implementation",
                         current_task_id=None if last else tasks[index + 1]["task_id"]), complete=last)
            terminal = load_checkpoint(handoff.checkpoint_path)
            if terminal["status"] != "completed":
                raise _Stop("TASK_CHECKPOINT_CHANGED")
            records[index] = _report_task(handoff, task, terminal, published.staging, published.publishing, True)
            _check_report_history(records, report_max_bytes)
        current, phase = None, "reporting"
    except _CheckpointFailure:
        reason = checkpoint_error = "BATCH_CHECKPOINT_WRITE_FAILED"
    except (Exception, KeyboardInterrupt) as error:
        reason = _code(str(error), "BATCH_EXECUTION_FAILED") if isinstance(error, _Stop) else (
            "BATCH_INTERRUPTED_STATE_UNCERTAIN" if isinstance(error, KeyboardInterrupt) else "BATCH_EXECUTION_FAILED")
    if reason is not None and created and not storage_failed and state["status"] != "completed":
        # A terminal checkpoint is written once. Failed writes are never retried.
        if state["status"] != "stopped":
            stage = phase if phase in {"implementation", "testing", "review", "staging", "publishing"} else state["stage"]
            try:
                persist(dict(state, status="stopped", stage=stage, stop_reason_code=reason,
                             **({"timeout": timeout_detail.record()} if timeout_detail is not None else {})))
            except _CheckpointFailure:
                checkpoint_error = "BATCH_CHECKPOINT_WRITE_FAILED"
        try:
            index = state["task_ids"].index(state["current_task_id"])
            current = state["current_task_id"]
            task = tasks[index]
            # Preflight can return known no-start observations before assigning
            # task identity. Preserve them rather than marking execution unknown.
            stopped_before_identity = (
                handoff is not None and handoff.task_id is None
                and handoff.outcome == "STOPPED_BEFORE_IMPLEMENTATION"
                and handoff.implementation_started is False and handoff.review_started is False)
            observation = handoff if handoff is not None and (
                handoff.task_id == task["task_id"] or stopped_before_identity) else FirstTaskReviewResult(
                "NO_RESULT", phase, implementation_started=None if execution_invoked else False,
                review_started=None if execution_invoked else False)
            task_publication = published if observation is handoff else None
            records[index] = _report_task(observation, task, state,
                                           None if task_publication is None else task_publication.staging,
                                           None if task_publication is None else task_publication.publishing,
                                           False if task_publication is None else task_publication.publishing_invoked,
                                           record_unattempted=False)
        except (ValueError, AttributeError, TypeError):
            if state["current_task_id"] is not None:
                report_error = "REPORT_OBSERVATIONS_UNAVAILABLE"
    if created:
        try:
            terminal = load_checkpoint(path)
            if storage_failed or terminal != state or terminal["status"] not in {"stopped", "completed"}:
                raise _Stop("REPORT_CHECKPOINT_NOT_TERMINAL")
            if report_error is not None:
                raise _Stop(report_error)
            paths = export_batch_report(terminal, metadata, records, [], report_max_bytes,
                                        destination=directory, repository_directory=repository)
        except ReportExportError as error:
            report_error, partial_paths = "REPORT_EXPORT_FAILED", tuple(error.partial_paths[:4])
        except (Exception, KeyboardInterrupt) as error:
            report_error = _code(str(error), "REPORT_UNAVAILABLE") if isinstance(error, _Stop) else "REPORT_UNAVAILABLE"
    reason = reason or report_error
    outcome = ("COMPLETED" if reason is None else "PUBLISHED_NOT_FINALIZED"
               if state is not None and state["status"] == "completed" else "PARTIALLY_COMPLETED" if publications else "STOPPED")
    return SequentialBatchResult(outcome, phase, reason, checkpoint_error, report_error,
                                 tuple(completed), tuple(publications), current, path, paths, partial_paths,
                                 timeout=timeout_detail)
