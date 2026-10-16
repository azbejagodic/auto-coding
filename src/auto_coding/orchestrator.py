"""Review a first frozen task, then separately publish a single-task batch.

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

from auto_coding.batch_state import CheckpointError, create_checkpoint, load_checkpoint, update_checkpoint
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
from auto_coding.plan_preflight import preflight_frozen_plan
from auto_coding.review import ReviewResult, _evidence_fingerprint, build_review_packet, run_review_packet
from auto_coding.test_runner import TestRunResult, _command_identity, run_approved_tests


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

    @property
    def ready_for_publication(self) -> bool:
        return self.outcome == "READY_FOR_PUBLICATION"


class _Stop(Exception):
    pass


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
        raise _Stop(_code(state.error_code, "GIT_STATE_UNVERIFIED"))
    evidence = collect_change_evidence(repository, task["permitted_files"], max_bytes)
    if not isinstance(evidence, ChangeEvidence) or evidence.passed is not True:
        raise _Stop(_code(getattr(evidence, "error_code", None), "EVIDENCE_INVALID"))
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
        preflight = preflight_frozen_plan(frozen_plan_path, approved_sha256, repository_directory, git_timeout_seconds)
        if not preflight.ready:
            raise _Stop(_code(preflight.error_code, "PREFLIGHT_FAILED"))
        try:
            plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        except PlanIntegrityError:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED") from None
        except OSError:
            raise _Stop("PLAN_IO_FAILED") from None
        if plan["schema_version"] != 2 or plan_fingerprint(plan) != approved_sha256:
            raise _Stop("PLAN_INVALID_OR_UNAPPROVED")
        task = plan["tasks"][0]
        if task["order"] != 1:
            raise _Stop("FIRST_TASK_INVALID")
        repository = Path(repository_directory).resolve(strict=True)
        requested_destination = Path(checkpoint_path)
        if os.path.lexists(requested_destination):
            raise _Stop("CHECKPOINT_EXISTS")
        destination = requested_destination.resolve()
        if destination == repository or repository in destination.parents:
            raise _Stop("UNSAFE_CHECKPOINT_LOCATION")
        values.update(batch_id=plan["batch_id"], task_id=task["task_id"], plan_sha256=approved_sha256,
                      repository_directory=repository, checkpoint_path=destination,
                      expected_branch=plan["expected_branch"], expected_head=plan["expected_head"],
                      expected_origin_url=plan["expected_origin_url"], permitted_files=tuple(task["permitted_files"]))
        state = {"schema_version": 1, "batch_id": plan["batch_id"], "frozen_plan_sha256": approved_sha256,
                 "expected_branch": plan["expected_branch"], "base_head": plan["expected_head"],
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
        preflight = preflight_frozen_plan(frozen_plan_path, approved_sha256, repository, git_timeout_seconds)
        if not preflight.ready:
            raise _Stop(_code(preflight.error_code, "PREFLIGHT_FAILED"))
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
        if isinstance(implementation, ImplementationOutcome) and _session(implementation.session_id):
            values["implementer_session_id"] = implementation.session_id
            persist(implementer_session_id=implementation.session_id)
        if (not isinstance(implementation, ImplementationOutcome) or implementation.succeeded is not True
                or type(implementation.exit_status) is not int or implementation.exit_status != 0
                or implementation.timed_out is not False or implementation.cleanup_result != "not_needed"
                or implementation.error_code is not None or not _session(implementation.session_id)
                or not isinstance(implementation.report, ImplementationReport)
                or type(implementation.elapsed_seconds) not in (int, float)
                or not math.isfinite(implementation.elapsed_seconds) or implementation.elapsed_seconds < 0):
            raise _Stop(_code(getattr(implementation, "error_code", None), "IMPLEMENTATION_INVALID"))
        phase = "evidence"
        values["evidence"] = _snapshot(repository, plan, task, git_timeout_seconds, evidence_max_bytes)
        phase = "testing"
        persist(stage="testing")
        tests = run_approved_tests(repository, task["test_commands"], task["limits"]["test_seconds"])
        values["terminal_results"] = tests
        evidence = _snapshot(repository, plan, task, git_timeout_seconds, evidence_max_bytes)
        values["evidence"] = evidence
        if not isinstance(tests, TestRunResult) or tests.passed is not True:
            if isinstance(tests, TestRunResult) and isinstance(tests.command_results, tuple):
                if any(getattr(result, "timed_out", None) is True for result in tests.command_results):
                    raise _Stop("TEST_TIMEOUT")
                if any(getattr(result, "cleanup_result", None) != "not_needed" for result in tests.command_results):
                    raise _Stop("TEST_CLEANUP_UNCERTAIN")
            raise _Stop("TESTS_NOT_PASSED")
        context = _context(repository, plan, evidence, git_timeout_seconds, evidence_max_bytes)
        if _snapshot(repository, plan, task, git_timeout_seconds, evidence_max_bytes) != evidence:
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
        review = run_review_packet(evidence, task["acceptance_criteria"], task["test_commands"], tests, context, packet_max_bytes,
                                   model=settings["model"], reasoning_effort=settings["reasoning_effort"],
                                   repository_directory=repository, timeout_seconds=task["limits"]["review_seconds"],
                                   session_id=None, review_prompt=settings["prompt"])
        values["review"] = review
        if isinstance(review, ReviewResult) and _session(review.session_id):
            values["reviewer_session_id"] = review.session_id
            if review.session_id.lower() == implementation.session_id.lower():
                raise _Stop("REVIEW_SESSION_NOT_INDEPENDENT")
            persist(reviewer_session_id=review.session_id)
        if not isinstance(review, ReviewResult) or review.approved is not True:
            raise _Stop(_code(getattr(review, "error_code", None), "REVIEW_NOT_APPROVED"))
        if _review_snapshot(packet, review, packet_max_bytes) != evidence:
            raise _Stop("REVIEW_BINDING_MISMATCH")
        if (_snapshot(repository, plan, task, git_timeout_seconds, evidence_max_bytes) != evidence
                or _context(repository, plan, evidence, git_timeout_seconds, evidence_max_bytes) != context):
            raise _Stop("EVIDENCE_CHANGED_AFTER_REVIEW")
        phase = "publishing"
        persist(stage="publishing")
        return finish(ready=True)
    except _CheckpointFailure:
        return finish("CHECKPOINT_WRITE_FAILED", "CHECKPOINT_WRITE_FAILED")
    except Exception as error:
        reason = _code(str(error), "EXECUTION_FAILED") if isinstance(error, (_Stop, _StageFailure)) else "EXECUTION_FAILED"
        checkpoint_error = None
        if created:
            try:
                persist(status="stopped", stop_reason_code=reason)
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


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def _publication_handoff(handoff, plan, approved_sha256, evidence_limit, packet_limit):
    """Bind the entire retained packet and result metadata to the frozen task."""
    task = plan["tasks"][0]
    if (not isinstance(handoff, FirstTaskReviewResult) or handoff.outcome != "READY_FOR_PUBLICATION"
            or handoff.stage != "publishing" or handoff.error_code is not None
            or handoff.checkpoint_error_code is not None or handoff.implementation_started is not True
            or handoff.review_started is not True or handoff.batch_id != plan["batch_id"]
            or handoff.task_id != task["task_id"] or handoff.plan_sha256 != approved_sha256
            or handoff.expected_branch != plan["expected_branch"] or handoff.expected_head != plan["expected_head"]
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


def _report_task(handoff, task, checkpoint, staged, publishing, publishing_invoked):
    """Explicit safe projection: never dump adapter objects, prompts or bytes."""
    implementation, tests, review = handoff.implementation, handoff.terminal_results, handoff.review
    claims, verdict = implementation.report, review.verdict
    complete = checkpoint["status"] == "completed"
    record = {
        "task_id": task["task_id"], "title": task["title"],
        "objective": "Not separately recorded in the frozen plan; see title and acceptance criteria.",
        "acceptance_criteria": task["acceptance_criteria"], "permitted_files": task["permitted_files"],
        "implementer": {"model": task["implementation"]["model"], "reasoning_effort": task["implementation"]["reasoning_effort"],
                        "session_id": handoff.implementer_session_id, "session_mode": "fresh"},
        "reviewer": {"model": task["review"]["model"], "reasoning_effort": task["review"]["reasoning_effort"],
                     "session_id": handoff.reviewer_session_id, "session_mode": "fresh"},
        "status": "completed" if complete else "stopped",
        "failure_stage": None if complete else checkpoint["stage"],
        "stop_reason_code": checkpoint["stop_reason_code"],
        "implementation_started": handoff.implementation_started, "review_started": handoff.review_started,
        "implementation": {
            "succeeded": implementation.succeeded, "exit_status": implementation.exit_status,
            "elapsed_seconds": implementation.elapsed_seconds, "timed_out": implementation.timed_out,
            "cleanup_result": implementation.cleanup_result, "error_code": implementation.error_code,
            "session_id": implementation.session_id,
            "model_claims": {"summary": claims.summary, "approach": claims.approach,
                             "files_reported": list(claims.files_reported), "model_reported_confidence": claims.confidence,
                             "concerns": list(claims.concerns)},
        },
        "evidence": {"passed": handoff.evidence.passed, "changed_paths": list(handoff.evidence.changed_paths),
                     "fingerprint": handoff.evidence.fingerprint, "error_code": handoff.evidence.error_code},
        "approved_command_identities": [_command_identity(command) for command in task["test_commands"]],
        "terminal_results": {
            "passed": tests.passed, "commands_executed": tests.commands_executed, "first_failed_index": tests.first_failed_index,
            "command_results": [{"index": result.index, "command_identity": result.command_identity,
                                 "exit_status": result.exit_status, "elapsed_seconds": result.elapsed_seconds,
                                 "timed_out": result.timed_out, "cleanup_result": result.cleanup_result}
                                for result in tests.command_results],
        },
        "review": {"approved": review.approved, "exit_status": review.exit_status, "timed_out": review.timed_out,
                   "cleanup_result": review.cleanup_result, "error_code": review.error_code, "session_id": review.session_id,
                   "reviewed_packet_sha256": review.reviewed_packet_sha256,
                   "reviewed_evidence_fingerprint": review.reviewed_evidence_fingerprint,
                   "model_verdict": {"decision": verdict.decision, "blocking_issues": list(verdict.blocking_issues),
                                     "nonblocking_notes": list(verdict.nonblocking_notes),
                                     "checked_requirements": list(verdict.checked_requirements), "model_reported_confidence": verdict.confidence}},
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
    elif not publishing_invoked:
        # Known control flow, not a fabricated successful Git result. When a
        # gate was invoked but returned no result, these observations stay None.
        record["publishing"] = {"outcome": "NOT_ATTEMPTED", "error_code": checkpoint["stop_reason_code"],
                                "commit_attempted": False, "push_attempted": False,
                                "commit_sha": None, "remote_sha": None, "observed_head_sha": None}
    return record


def publish_single_task_batch(
    handoff: FirstTaskReviewResult, frozen_plan_path: str | Path, approved_sha256: str,
    git_timeout_seconds: float, evidence_max_bytes: int, packet_max_bytes: int, report_max_bytes: int,
    metadata_directory: str | Path, repository_identifier: str,
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
        if len(plan["tasks"]) != 1:
            raise _Stop("SINGLE_TASK_BATCH_REQUIRED")
        task = plan["tasks"][0]
        if task["order"] != 1:
            raise _Stop("FIRST_TASK_INVALID")
        _publication_handoff(handoff, plan, approved_sha256, evidence_max_bytes, packet_max_bytes)
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
                    "expected_branch": plan["expected_branch"], "base_head": plan["expected_head"],
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
        staged = stage_reviewed_changes(repository, plan["expected_branch"], plan["expected_head"],
                                        plan["expected_origin_url"], git_timeout_seconds, task["permitted_files"],
                                        handoff.reviewed_packet, handoff.review, max(evidence_max_bytes, packet_max_bytes))
        values["staging"] = staged if isinstance(staged, GitStageResult) else None
        if (not isinstance(staged, GitStageResult) or staged.passed is not True or staged.error_code is not None
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
        publishing = publish_reviewed_changes(repository, plan["expected_branch"], plan["expected_head"],
                                             plan["expected_origin_url"], git_timeout_seconds, staged,
                                             handoff.reviewed_packet, handoff.review, max(evidence_max_bytes, packet_max_bytes),
                                             f"auto-coding: {task['task_id']}")
        values["publishing"] = publishing if isinstance(publishing, GitPublishResult) else None
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
        if (publishing.error_code is not None or publishing.commit_attempted is not True or publishing.push_attempted is not True
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
            persist(status="stopped", stop_reason_code=reason)
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
