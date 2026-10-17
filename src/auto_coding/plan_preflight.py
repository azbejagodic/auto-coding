"""Read-only execution readiness for an explicitly approved frozen v2 plan.

The supplied fingerprint must come from a trusted human-approval channel. D04
storage checks integrity, not approval provenance: an arbitrary file and its
self-computed hash are not independent evidence of human approval. This gate
does not run tasks or mutate checkpoint, plan, repository or remote state.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path

from auto_coding.change_evidence import _external_temporary_directory
from auto_coding.git_publish import (
    _REPOSITORY_ENVIRONMENT, _git_query, check_publish_preflight,
)
from auto_coding.plan_freeze import PlanIntegrityError, load_frozen_plan, plan_fingerprint
from auto_coding.plans import PlanValidationError
from auto_coding.operations import TimeoutDiagnostic


@dataclass(frozen=True)
class PlanPreflightResult:
    ready: bool
    error_code: str | None = None
    batch_id: str | None = None
    plan_sha256: str | None = None
    branch: str | None = None
    starting_head: str | None = None
    repository_directory: Path | None = None
    origin_url: str | None = None
    test_timeouts: tuple[tuple[str, int], ...] = ()
    timeout: TimeoutDiagnostic | None = None


def preflight_frozen_plan(
    frozen_plan_path: str | Path,
    approved_sha256: str,
    repository_directory: str | Path,
    git_timeout_seconds: float,
) -> PlanPreflightResult:
    """Return verified identity and ordered (task ID, test seconds) pairs.

Each task's approved test_seconds applies separately to every ordered command.
Version 1 remains loadable but is not runnable. Failure diagnostics contain no
prompts, raw Git output or exception messages. Ignored files remain ignored.
This is a snapshot, not a lock: D16 must recheck critical state immediately
before execution and publishing. No hash alone proves human authorization.
"""
    try:
        valid_timeout = (type(git_timeout_seconds) in (int, float)
                         and math.isfinite(git_timeout_seconds) and git_timeout_seconds > 0)
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        return PlanPreflightResult(False, "INVALID_TIMEOUT")
    try:
        plan = load_frozen_plan(frozen_plan_path, approved_sha256)
        canonical_fingerprint = plan_fingerprint(plan)
        if canonical_fingerprint != approved_sha256:
            return PlanPreflightResult(False, "PLAN_FINGERPRINT_MISMATCH")
    except (PlanIntegrityError, PlanValidationError, ValueError, TypeError, RecursionError):
        return PlanPreflightResult(False, "PLAN_INVALID_OR_UNAPPROVED")
    except OSError:
        return PlanPreflightResult(False, "PLAN_IO_FAILED")
    if plan["schema_version"] != 2:
        return PlanPreflightResult(False, "PLAN_SCHEMA_NOT_RUNNABLE")
    return _preflight_verified_plan(plan, approved_sha256, repository_directory,
                                    git_timeout_seconds, plan["expected_head"])


def _preflight_verified_plan(plan, approved_sha256, repository_directory,
                             git_timeout_seconds, starting_head):
    """Internal controller check at a previously verified task boundary.

The caller must first validate the original frozen plan and derive starting_head
from the preceding verified publication. This never changes the approved plan.
"""
    git_state = check_publish_preflight(
        repository_directory, plan["expected_branch"], starting_head,
        plan["expected_origin_url"], git_timeout_seconds,
    )
    if not git_state.passed:
        return PlanPreflightResult(False, git_state.error_code, batch_id=plan["batch_id"],
                                   timeout=TimeoutDiagnostic("git_preflight", git_timeout_seconds)
                                   if git_state.error_code == "GIT_TIMEOUT" else None)
    try:
        repository = Path(repository_directory).resolve(strict=True)
        temporary = _external_temporary_directory(repository)
        if temporary is None:
            return PlanPreflightResult(False, "UNSAFE_TEMP_LOCATION")
        environment = os.environ.copy()
        if any(name in environment for name in _REPOSITORY_ENVIRONMENT):
            return PlanPreflightResult(False, "UNSAFE_GIT_ENVIRONMENT")
        environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
        code, output, error = _git_query(
            repository,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            git_timeout_seconds, temporary, environment,
        )
        if error:
            return PlanPreflightResult(False, error, batch_id=plan["batch_id"],
                                       timeout=TimeoutDiagnostic("git_preflight", git_timeout_seconds)
                                       if error == "GIT_TIMEOUT" else None)
        if code != 0:
            return PlanPreflightResult(False, "GIT_FAILED")
        if output:
            return PlanPreflightResult(False, "WORKTREE_NOT_CLEAN")
        return PlanPreflightResult(
            True, batch_id=plan["batch_id"], plan_sha256=approved_sha256,
            branch=plan["expected_branch"], starting_head=starting_head.lower(),
            repository_directory=repository, origin_url=plan["expected_origin_url"],
            test_timeouts=tuple((task["task_id"], task["limits"]["test_seconds"]) for task in plan["tasks"]),
        )
    except (OSError, ValueError, TypeError):
        return PlanPreflightResult(False, "GIT_IO_FAILED")
