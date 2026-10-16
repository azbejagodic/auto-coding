"""Execute only the first frozen task through independent review, never publish.

The supplied approval fingerprint is a trusted caller assertion. Checks observe
snapshots, not locks or containment; live Codex sandbox enforcement is a separate
verification. Evidence substeps use their enclosing implementation/testing/review
checkpoint stage, preserving D14a's schema. No resume or recovery is attempted.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import stat
import uuid

from auto_coding.batch_state import CheckpointError, create_checkpoint, update_checkpoint
from auto_coding.change_evidence import (
    ChangeEvidence, _external_temporary_directory, _safe_location, collect_change_evidence,
)
from auto_coding.git_publish import (
    _REPOSITORY_ENVIRONMENT, _StageFailure, _git_query, _review_snapshot, check_publish_preflight,
)
from auto_coding.integrations.codex import ImplementationOutcome, ImplementationReport, run_implementation
from auto_coding.plan_freeze import PlanIntegrityError, load_frozen_plan, plan_fingerprint
from auto_coding.plan_preflight import preflight_frozen_plan
from auto_coding.review import ReviewResult, _evidence_fingerprint, build_review_packet, run_review_packet
from auto_coding.test_runner import TestRunResult, run_approved_tests


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
