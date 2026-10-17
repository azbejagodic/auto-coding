"""Manual continuation prepares a new run; it never resumes a stopped run.

Metadata records are controller observations, not authenticated history. Verify
their consistency and immutable Git objects/remote refs before preparing a clone.
Checks are snapshots, not locks. Failures preserve newly created partial paths.
"""

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

from auto_coding.batch_report import BatchReportError, build_batch_report
from auto_coding.batch_state import CheckpointError, _unique_fields, _validate as validate_checkpoint
from auto_coding.git_publish import _REPOSITORY_ENVIRONMENT, _check_publish_state, _git_query, _raw_index_entries
from auto_coding.change_evidence import _external_temporary_directory
from auto_coding.plan_freeze import PlanIntegrityError, load_frozen_plan, plan_fingerprint
from auto_coding.plans import PlanValidationError, _branch, _origin_url, validate_plan
from auto_coding.test_runner import _command_identity


class ContinuationError(ValueError):
    """Carries a fixed reason code; never an exception transcript."""


@dataclass(frozen=True)
class ContinuationInspection:
    safe: bool = False
    error_code: str | None = None
    batch_id: str | None = None
    status: str | None = None
    published_task_ids: tuple[str, ...] = ()
    base_sha: str | None = None
    stopped_task_id: str | None = None
    stop_reason_code: str | None = None
    never_started_task_ids: tuple[str, ...] = ()
    unknown_task_ids: tuple[str, ...] = ()
    baseline_only: bool = False
    branch: str | None = None
    origin: str | None = field(default=None, repr=False)
    source_checkpoint_sha256: str | None = None
    source_report_sha256: str | None = None
    reported_review_decision: str | None = None


@dataclass(frozen=True)
class ContinuationPreparation:
    prepared: bool
    inspection: ContinuationInspection
    error_code: str | None = None
    dry_run: bool = False
    plan_sha256: str | None = None
    repository: Path | None = None
    metadata_directory: Path | None = None
    new_batch_id: str | None = None
    branch: str | None = None


def _fail(code):
    raise ContinuationError(code)


def _plain_path(value):
    requested = Path(value).absolute()
    if ".." in requested.parts:
        _fail("UNSAFE_CONTINUATION_PATH")
    for component in (*reversed(requested.parents), requested):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            _fail("UNSAFE_CONTINUATION_PATH")
    resolved = requested.resolve()
    if resolved != requested:
        _fail("UNSAFE_CONTINUATION_PATH")
    return resolved


def _read(path, limit):
    if not stat.S_ISREG(path.lstat().st_mode):
        _fail("SOURCE_FILE_NOT_REGULAR")
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        _fail("CONTINUATION_INPUT_TOO_LARGE")
    return raw


def _json(raw):
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_fields)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _environment(timeout, max_bytes):
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 86400
    except OverflowError:
        valid = False
    if not valid:
        _fail("INVALID_TIMEOUT")
    if type(max_bytes) is not int or not 0 < max_bytes <= 64 * 1024 * 1024:
        _fail("INVALID_MAX_BYTES")
    if any(name in os.environ for name in _REPOSITORY_ENVIRONMENT):
        _fail("UNSAFE_GIT_ENVIRONMENT")
    return dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")


def _query(repository, arguments, timeout, temporary, environment):
    code, output, error = _git_query(repository, arguments, timeout, temporary, environment)
    if error:
        _fail(error)
    if code != 0:
        _fail("CONTINUATION_GIT_FAILED")
    return output


def _temporary_parent(repository, source_metadata):
    directory = _external_temporary_directory(repository)
    if directory is None or directory.is_relative_to(source_metadata):
        _fail("UNSAFE_TEMP_LOCATION")
    return directory


def inspect_continuation(source_plan, source_approved_sha256, source_metadata, repository,
                         git_timeout_seconds=120, report_max_bytes=2097152):
    """Read-only inspection. A dirty source is allowed but never used for copying.

    Requires the original independently supplied approval, complete canonical
    reports and a stopped sequential checkpoint. Git HEAD and the source remote
    branch must still equal the recorded published base (or original baseline).
    """
    result = ContinuationInspection()
    try:
        environment = _environment(git_timeout_seconds, report_max_bytes)
        root, metadata, frozen = map(_plain_path, (repository, source_metadata, source_plan))
        if not root.is_dir() or not metadata.is_dir() or metadata.is_relative_to(root) or frozen.is_relative_to(root):
            _fail("UNSAFE_SOURCE_LOCATION")
        plan = load_frozen_plan(frozen, source_approved_sha256)
        if plan["schema_version"] != 2 or plan_fingerprint(plan) != source_approved_sha256:
            _fail("SOURCE_PLAN_UNAPPROVED")
        raw_checkpoint = _read(_plain_path(metadata / "batch-checkpoint.json"), 16384)
        state = validate_checkpoint(_json(raw_checkpoint))
        if state["schema_version"] != 2 or state["status"] != "stopped":
            _fail("STOPPED_SEQUENTIAL_BATCH_REQUIRED")
        if (state["batch_id"] != plan["batch_id"] or state["frozen_plan_sha256"] != source_approved_sha256
                or state["expected_branch"] != plan["expected_branch"] or state["base_head"] != plan["expected_head"]
                or state["task_ids"] != [task["task_id"] for task in plan["tasks"]]):
            _fail("SOURCE_CHECKPOINT_PLAN_MISMATCH")
        raw_report = _read(_plain_path(metadata / "batch-report.json"), report_max_bytes)
        data = _json(raw_report)
        if not isinstance(data, dict) or data.keys() != {"schema_version", "provenance", "checkpoint", "batch", "tasks", "events"}:
            _fail("SOURCE_REPORT_INVALID")
        if _canonical(data["checkpoint"]) != _canonical(state):
            _fail("SOURCE_REPORT_CHECKPOINT_MISMATCH")
        # Remove only documented derived fields, then use the existing validator
        # and compare the complete rebuilt reports, including derived values.
        batch = {key: value for key, value in data["batch"].items()
                 if key not in {"planned_tasks", "completed_tasks", "stopped_tasks"}}
        if not isinstance(data["tasks"], list) or len(data["tasks"]) != len(plan["tasks"]):
            _fail("SOURCE_REPORT_INVALID")
        records = []
        for record in data["tasks"]:
            copy = dict(record)
            if copy["publishing"] is not None:
                copy["publishing"] = {key: value for key, value in copy["publishing"].items() if key != "remote_verification"}
            records.append(copy)
        rebuilt = build_batch_report(state, batch, records, data["events"], report_max_bytes)
        if rebuilt.json_bytes != _canonical(data) or rebuilt.markdown_bytes != _read(
                _plain_path(metadata / "batch-report.md"), report_max_bytes):
            _fail("SOURCE_REPORT_INCONSISTENT")
        published = state["publications"]
        base = state["current_head"]
        result = ContinuationInspection(
            batch_id=plan["batch_id"], status=state["status"],
            published_task_ids=tuple(entry["task_id"] for entry in published), base_sha=base,
            stopped_task_id=state["current_task_id"], stop_reason_code=state["stop_reason_code"],
            never_started_task_ids=tuple(record["task_id"] for record in records if record["implementation_started"] is False),
            unknown_task_ids=tuple(record["task_id"] for record in records if record["implementation_started"] is None),
            baseline_only=not published, branch=plan["expected_branch"], origin=plan["expected_origin_url"],
            source_checkpoint_sha256=hashlib.sha256(raw_checkpoint).hexdigest(),
            source_report_sha256=hashlib.sha256(raw_report).hexdigest())
        stopped_record = next((record for record in records if record["task_id"] == state["current_task_id"]), None)
        if stopped_record is not None and stopped_record["review"] is not None:
            verdict = stopped_record["review"]["model_verdict"]
            result = replace(result, reported_review_decision=None if verdict is None else verdict["decision"])
        if any(not entry["finalized"] for entry in published):
            _fail("SOURCE_PUBLICATION_NOT_FINALIZED")
        for task, record in zip(plan["tasks"], records):
            if (any(record[name] != task[name] for name in ("task_id", "title", "acceptance_criteria", "permitted_files"))
                    or record["approved_command_identities"] != [_command_identity(command) for command in task["test_commands"]]
                    or any(record[role][name] != task[settings][name]
                           for role, settings in (("implementer", "implementation"), ("reviewer", "review"))
                           for name in ("model", "reasoning_effort"))):
                _fail("SOURCE_REPORT_PLAN_MISMATCH")
            publication = record["publishing"]
            if publication is not None and publication["outcome"] not in {"NOT_ATTEMPTED", "PUBLISHED"}:
                _fail("SOURCE_PUBLICATION_AMBIGUOUS")
            for flag, field in (("implementation_started", "implementation"), ("review_started", "review")):
                if record[flag] is None or (record[flag] and record[field] is None):
                    _fail("SOURCE_EXECUTION_UNCERTAIN")
            observed = [record[name] for name in ("implementation", "review") if record[name] is not None]
            if record["terminal_results"] is not None:
                observed.extend(record["terminal_results"]["command_results"])
            if any(item["cleanup_result"] not in {"not_needed", "taskkill_completed", "launch_failed"} for item in observed):
                _fail("SOURCE_PROCESS_CLEANUP_UNCERTAIN")
            if (record["failure_stage"] in {"testing", "review", "staging", "publishing"}
                    and record["terminal_results"] is None):
                _fail("SOURCE_EXECUTION_UNCERTAIN")
        current_index = state["task_ids"].index(state["current_task_id"]) if state["current_task_id"] is not None else None
        if current_index is not None:
            current_path = _plain_path(metadata / f"task-{current_index + 1:04d}" / "batch-checkpoint.json")
            current_record = records[current_index]
            if current_path.exists():
                current = validate_checkpoint(_json(_read(current_path, 16384)))
                if (current["schema_version"] != 1 or current["status"] != "stopped"
                        or current["current_task_id"] != state["current_task_id"] or current["base_head"] != base
                        or current["stage"] != state["stage"]
                        or any(current[name] != state[name] for name in ("batch_id", "frozen_plan_sha256", "expected_branch", "stop_reason_code"))
                        or current["implementer_session_id"] != current_record["implementer"]["session_id"]
                        or current["reviewer_session_id"] != current_record["reviewer"]["session_id"]):
                    _fail("SOURCE_TASK_CHECKPOINT_MISMATCH")
            elif current_record["implementation_started"] is not False or current_record["review_started"] is not False:
                _fail("SOURCE_TASK_CHECKPOINT_UNAVAILABLE")
        with tempfile.TemporaryDirectory(dir=_temporary_parent(root, metadata)) as temporary:
            def verify_state():
                check = _check_publish_state(root, result.branch, base, result.origin,
                                             git_timeout_seconds, allow_staged=True)
                if not check.passed:
                    _fail(check.error_code)
            verify_state()
            for index, entry in enumerate(published):
                child = validate_checkpoint(_json(_read(_plain_path(
                    metadata / f"task-{index + 1:04d}" / "batch-checkpoint.json"), 16384)))
                record = records[index]
                if (child["schema_version"] != 1 or child["status"] != "completed"
                        or any(child[name] != state[name] for name in ("batch_id", "frozen_plan_sha256", "expected_branch"))
                        or child["base_head"] != entry["starting_head"]
                        or child["implementer_session_id"] != record["implementer"]["session_id"]
                        or child["reviewer_session_id"] != record["reviewer"]["session_id"]):
                    _fail("SOURCE_TASK_CHECKPOINT_MISMATCH")
                commit = _query(root, ["--no-replace-objects", "cat-file", "commit", entry["commit_sha"]],
                                git_timeout_seconds, temporary, environment)
                headers, separator, _ = commit.partition(b"\n\n")
                if not separator or [line for line in headers.split(b"\n") if line.startswith(b"parent ")] != [
                        b"parent " + entry["starting_head"].lower().encode("ascii")]:
                    _fail("SOURCE_COMMIT_CHAIN_MISMATCH")
                raw = _query(root, ["--no-replace-objects", "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                                   "--raw", "--abbrev=40", "-z", "--ita-visible-in-index",
                                   entry["starting_head"], entry["commit_sha"], "--"],
                             git_timeout_seconds, temporary, environment)
                if (not raw or tuple(sorted(_raw_index_entries(raw))) != tuple(record["evidence"]["changed_paths"])
                        or hashlib.sha256(raw).hexdigest() != record["staging"]["staged_changes_sha256"]):
                    _fail("SOURCE_COMMIT_EVIDENCE_MISMATCH")
            verify_state()
        return replace(result, safe=True)
    except ContinuationError as error:
        return replace(result, error_code=str(error))
    except PlanIntegrityError:
        return replace(result, error_code="SOURCE_PLAN_UNAPPROVED")
    except (CheckpointError, BatchReportError, PlanValidationError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return replace(result, error_code="SOURCE_METADATA_INVALID")
    except OSError:
        return replace(result, error_code="CONTINUATION_IO_FAILED")


_REFERENCE_NOTICE = "Manual continuation reference; not approval or authenticated execution history."
_REFERENCE_FIELDS = {"schema_version", "provenance", "source_batch_id", "source_plan_sha256",
                     "source_checkpoint_sha256", "source_report_sha256", "source_repository", "source_metadata",
                     "published_task_ids", "new_batch_id", "proposed_plan_sha256", "base_sha",
                     "branch", "origin", "repository", "metadata_directory"}


def check_continuation_binding(plan, approved_sha256, repository, metadata):
    """Check an optional preparation reference; it never supplies approval."""
    path = Path(metadata) / "continuation.json"
    if not os.path.lexists(path):
        return
    data = _json(_read(_plain_path(path), 16384))
    if (not isinstance(data, dict) or data.keys() != _REFERENCE_FIELDS or type(data["schema_version"]) is not int
            or data["schema_version"] != 1 or data["provenance"] != _REFERENCE_NOTICE):
        _fail("CONTINUATION_REFERENCE_INVALID")
    for key in ("source_batch_id", "new_batch_id"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", data[key]):
            _fail("CONTINUATION_REFERENCE_INVALID")
    for key in ("source_plan_sha256", "source_checkpoint_sha256", "source_report_sha256", "proposed_plan_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            _fail("CONTINUATION_REFERENCE_INVALID")
    for key in ("source_repository", "source_metadata", "repository", "metadata_directory"):
        value = data[key]
        if (not isinstance(value, str) or len(value) > 4096 or not Path(value).is_absolute()
                or any(ord(c) < 32 or ord(c) == 127 for c in value)):
            _fail("CONTINUATION_REFERENCE_INVALID")
    _branch(data["branch"])
    _origin_url(data["origin"])
    if (data["new_batch_id"] == data["source_batch_id"] or data["new_batch_id"] != plan["batch_id"]
            or data["proposed_plan_sha256"] != approved_sha256 or plan_fingerprint(plan) != approved_sha256
            or data["base_sha"] != plan["expected_head"] or data["branch"] != plan["expected_branch"]
            or data["origin"] != plan["expected_origin_url"] or data["repository"] != str(Path(repository).resolve())
            or data["metadata_directory"] != str(Path(metadata).resolve())):
        _fail("CONTINUATION_PLAN_MISMATCH")
    if (not isinstance(data["published_task_ids"], list) or len(data["published_task_ids"]) > 100
            or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", item)
                   for item in data["published_task_ids"])
            or any(task["task_id"] in data["published_task_ids"] for task in plan["tasks"])):
        _fail("PUBLISHED_TASK_REPEATED")
    return data


def prepare_continuation(source_plan, source_approved_sha256, source_metadata, repository,
                         proposed_plan, destination, metadata_directory, *, git_timeout_seconds=120,
                         report_max_bytes=2097152, dry_run=False):
    """Create only new local paths/branch. Never freeze, execute or push a plan.

    The caller reviews a new plan and subsequently uses normal freeze/preflight/
    run. The new remote branch must be explicitly published by the user first.
    """
    inspection = inspect_continuation(source_plan, source_approved_sha256, source_metadata, repository,
                                      git_timeout_seconds, report_max_bytes)
    if not inspection.safe:
        return ContinuationPreparation(False, inspection, inspection.error_code)
    target = metadata = None
    fingerprint = None
    try:
        if type(dry_run) is not bool:
            _fail("INVALID_DRY_RUN")
        target, metadata, source_root, source_directory, proposed = map(
            _plain_path, (destination, metadata_directory, repository, source_metadata, proposed_plan))
        for path in (target, metadata):
            if os.path.lexists(path) or not path.parent.is_dir():
                _fail("NEW_DESTINATION_REQUIRED")
            if any(path.is_relative_to(root) for root in (source_root, source_directory)):
                _fail("UNSAFE_CONTINUATION_LOCATION")
        if target.is_relative_to(metadata) or metadata.is_relative_to(target):
            _fail("UNSAFE_CONTINUATION_LOCATION")
        plan = validate_plan(_json(_read(proposed, 1048576)))
        if (plan["schema_version"] != 2 or plan["batch_id"] == inspection.batch_id
                or plan["expected_head"] != inspection.base_sha or plan["expected_origin_url"] != inspection.origin
                or plan["expected_branch"] == inspection.branch or len(plan["expected_branch"]) > 512
                or plan["expected_branch"].casefold() in {"main", "master"}):
            _fail("NEW_PLAN_STATE_MISMATCH")
        if any(task["task_id"] in inspection.published_task_ids for task in plan["tasks"]):
            _fail("PUBLISHED_TASK_REPEATED")
        fingerprint = plan_fingerprint(plan)
        reference = dict(schema_version=1, provenance=_REFERENCE_NOTICE,
                         source_batch_id=inspection.batch_id, source_plan_sha256=source_approved_sha256,
                         source_checkpoint_sha256=inspection.source_checkpoint_sha256,
                         source_report_sha256=inspection.source_report_sha256,
                         source_repository=str(source_root), source_metadata=str(source_directory),
                         published_task_ids=list(inspection.published_task_ids), new_batch_id=plan["batch_id"],
                         proposed_plan_sha256=fingerprint, base_sha=inspection.base_sha, branch=plan["expected_branch"],
                         origin=inspection.origin, repository=str(target), metadata_directory=str(metadata))
        raw_reference = _canonical(reference)
        if len(raw_reference) > 16384 or any(len(reference[name]) > 4096 for name in (
                "source_repository", "source_metadata", "repository", "metadata_directory")):
            _fail("CONTINUATION_INPUT_TOO_LARGE")
        environment = _environment(git_timeout_seconds, report_max_bytes)
        with tempfile.TemporaryDirectory(dir=_temporary_parent(source_root, source_directory)) as temporary:
            code, output, error = _git_query(source_root, ["ls-remote", "--exit-code", "--refs", "origin",
                                                         f"refs/heads/{plan['expected_branch']}"],
                                            git_timeout_seconds, temporary, environment)
            if error:
                _fail(error)
            if code == 0:
                _fail("NEW_REMOTE_BRANCH_REQUIRED")
            if code != 2 or output:
                _fail("CONTINUATION_GIT_FAILED")
            if inspect_continuation(source_plan, source_approved_sha256, source_metadata, repository,
                                    git_timeout_seconds, report_max_bytes) != inspection:
                _fail("SOURCE_CHANGED_DURING_PREPARATION")
            if dry_run:
                return ContinuationPreparation(False, inspection, dry_run=True, plan_sha256=fingerprint,
                                               repository=target, metadata_directory=metadata,
                                               new_batch_id=plan["batch_id"], branch=plan["expected_branch"])
            # Exclusive reservations. Partial new directories remain on failure;
            # no reset, deletion, overwrite or automatic repair is attempted.
            target.mkdir()
            metadata.mkdir()
            _query(source_root, ["-c", f"core.hooksPath={temporary}", "clone", "--no-checkout", "--no-hardlinks",
                                 "--no-recurse-submodules", "--single-branch", "--branch", inspection.branch,
                                 "--origin", "origin", "--", inspection.origin, str(target)],
                   git_timeout_seconds, temporary, environment)
            _query(target, ["-c", f"core.hooksPath={temporary}", "checkout", "--no-recurse-submodules", "-b",
                            plan["expected_branch"], inspection.base_sha, "--"], git_timeout_seconds, temporary, environment)
            check = _check_publish_state(target, plan["expected_branch"], inspection.base_sha, inspection.origin,
                                         git_timeout_seconds, remote_branch=inspection.branch)
            # The new branch is local only. Reuse every state gate while checking
            # the explicitly selected source ref; normal run gates still require
            # the NEW remote branch to exist and match the new approved plan.
            if not check.passed:
                _fail(check.error_code)
            if _query(target, ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                      git_timeout_seconds, temporary, environment):
                _fail("PREPARED_REPOSITORY_DIRTY")
            if inspect_continuation(source_plan, source_approved_sha256, source_metadata, repository,
                                    git_timeout_seconds, report_max_bytes) != inspection:
                _fail("SOURCE_CHANGED_DURING_PREPARATION")
            # A clone can take time. Recheck aliases before writing the audit
            # reference as well; these observations do not provide locking.
            _plain_path(target)
            _plain_path(metadata)
            with (metadata / "continuation.json").open("xb") as output:
                output.write(raw_reference)
                output.flush()
                os.fsync(output.fileno())
        return ContinuationPreparation(True, inspection, plan_sha256=fingerprint,
                                       repository=target, metadata_directory=metadata,
                                       new_batch_id=plan["batch_id"], branch=plan["expected_branch"])
    except ContinuationError as error:
        code = str(error)
    except (PlanValidationError, CheckpointError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        code = "NEW_PLAN_INVALID"
    except OSError:
        code = "CONTINUATION_IO_FAILED"
    return ContinuationPreparation(False, inspection, code, plan_sha256=fingerprint,
                                   repository=target, metadata_directory=metadata)
