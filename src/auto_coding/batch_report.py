"""Bounded batch report generation from explicit, caller-supplied records.

Records use the result fields of the existing adapters, not raw outputs or
dataclass dumps. This module validates consistency, not authenticity, plan
provenance, or whether model summaries contain secrets. Sharing needs user
review. No execution history is inferred from a checkpoint.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

from auto_coding.batch_state import CheckpointError, _validate as _checkpoint
from auto_coding.plans import PlanValidationError, _path


class BatchReportError(ValueError):
    """Invalid, inconsistent, or oversized report input."""


class ReportExportError(OSError):
    """Export failed; partial_paths lists files that could not be removed."""

    def __init__(self, partial_paths: tuple[Path, ...] = ()):
        super().__init__("REPORT_EXPORT_FAILED")
        self.partial_paths = partial_paths


@dataclass(frozen=True, repr=False)
class BatchReport:
    json_bytes: bytes
    markdown_bytes: bytes


def _text(value, limit=2000):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(c) < 32 and c not in "\n\t" for c in value)
            or any(0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise BatchReportError("invalid bounded text")
    return value


def _pattern(pattern, limit):
    def validate(value):
        _text(value, limit)
        if not re.fullmatch(pattern, value):
            raise BatchReportError("invalid structured identifier")
        return value
    return validate


_ID = _pattern(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", 64)
_REASON = _pattern(r"[A-Z][A-Z0-9_]{0,63}", 64)
_SHA256 = _pattern(r"[0-9a-f]{64}", 64)
_SHA = _pattern(r"[0-9a-fA-F]{40}", 40)
_MODEL = _pattern(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", 128)
_REPO = _pattern(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", 256)


def _enum(*options):
    def validate(value):
        if not isinstance(value, str) or value not in options:
            raise BatchReportError("invalid enumerated value")
        return value
    return validate


def _boolean(value):
    if type(value) is not bool:
        raise BatchReportError("invalid boolean")
    return value


def _integer(value):
    if type(value) is not int or not -(2**31) <= value < 2**32:
        raise BatchReportError("invalid integer")
    return value


def _duration(value):
    if (type(value) not in (int, float) or value < 0
            or value > 10**9 or not math.isfinite(value)):
        raise BatchReportError("invalid duration")
    return value


def _uuid(value):
    _text(value, 36)
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise BatchReportError("invalid session UUID") from None
    if parsed.int == 0 or str(parsed) != value.lower():
        raise BatchReportError("invalid session UUID")
    return value


def _file_path(value):
    _text(value, 512)
    try:
        _path(value, "path")
    except PlanValidationError:
        raise BatchReportError("invalid repository-relative path") from None
    return value


def _timestamp(value):
    _text(value, 40)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise BatchReportError("invalid UTC timestamp") from None
    if "T" not in value or parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BatchReportError("timestamp must explicitly use UTC")
    return value


def _optional(schema):
    return ("optional", schema)


def _array(schema, limit=100):
    return ("array", schema, limit)


_CONFIDENCE = _enum("LOW", "MEDIUM", "HIGH")
_CLEANUP = _enum("not_needed", "taskkill_completed", "taskkill_failed", "taskkill_timed_out")
_ROLE = {
    "model": _MODEL, "reasoning_effort": _enum("low", "medium", "high", "xhigh"),
    "session_id": _optional(_uuid), "session_mode": _optional(_enum("fresh", "resumed")),
}
_IMPLEMENTATION = {
    "succeeded": _boolean, "exit_status": _optional(_integer),
    "elapsed_seconds": _duration, "timed_out": _boolean, "cleanup_result": _CLEANUP,
    "error_code": _optional(_REASON), "session_id": _optional(_uuid),
    "model_claims": _optional({
        "summary": _text, "approach": _text, "files_reported": _array(_file_path),
        "model_reported_confidence": _CONFIDENCE, "concerns": _array(_text),
    }),
}
_TERMINAL = {
    "passed": _boolean, "commands_executed": _integer, "first_failed_index": _optional(_integer),
    "command_results": _array({
        "index": _integer, "command_identity": _SHA256,
        "exit_status": _optional(_integer), "elapsed_seconds": _duration,
        "timed_out": _boolean, "cleanup_result": _CLEANUP,
    }),
}
_REVIEW = {
    "approved": _boolean, "exit_status": _optional(_integer), "timed_out": _boolean,
    "cleanup_result": _CLEANUP, "error_code": _optional(_REASON),
    "session_id": _optional(_uuid), "reviewed_packet_sha256": _optional(_SHA256),
    "reviewed_evidence_fingerprint": _optional(_SHA256),
    "model_verdict": _optional({
        "decision": _enum("APPROVED", "REJECTED", "UNCERTAIN"),
        "blocking_issues": _array(_text), "nonblocking_notes": _array(_text),
        "checked_requirements": _array(_text), "model_reported_confidence": _CONFIDENCE,
    }),
}
_STAGING = {
    "passed": _boolean, "error_code": _optional(_REASON), "staging_attempted": _boolean,
    "staged_paths": _optional(_array(_file_path)), "staged_changes_sha256": _optional(_SHA256),
    "staged_state_error_code": _optional(_REASON),
}
_PUBLISH = {
    "outcome": _enum("NOT_ATTEMPTED", "COMMIT_FAILED_OR_AMBIGUOUS",
                     "COMMIT_CREATED_NOT_PUSHED", "PUSH_FAILED_OR_AMBIGUOUS", "PUBLISHED"),
    "error_code": _optional(_REASON), "commit_attempted": _boolean, "push_attempted": _boolean,
    "commit_sha": _optional(_SHA), "remote_sha": _optional(_SHA),
    "observed_head_sha": _optional(_SHA),
}
_TASK = {
    "task_id": _ID, "title": _text, "objective": _text,
    "acceptance_criteria": _array(_text), "permitted_files": _array(_file_path),
    "implementer": _ROLE, "reviewer": _ROLE,
    "status": _enum("pending", "running", "stopped", "completed"),
    "failure_stage": _optional(_enum("implementation", "testing", "review", "staging", "publishing")),
    "stop_reason_code": _optional(_REASON),
    "implementation_started": _optional(_boolean), "review_started": _optional(_boolean),
    "implementation": _optional(_IMPLEMENTATION),
    "evidence": _optional({"passed": _boolean, "changed_paths": _array(_file_path),
                           "fingerprint": _optional(_SHA256), "error_code": _optional(_REASON)}),
    "approved_command_identities": _array(_SHA256), "terminal_results": _optional(_TERMINAL),
    "review": _optional(_REVIEW), "staging": _optional(_STAGING), "publishing": _optional(_PUBLISH),
    "preserved_evidence_location": _optional(lambda v: _text(v, 512)),
}
_BATCH = {"repository_id": _REPO, "started_at": _optional(_timestamp), "ended_at": _optional(_timestamp)}
_EVENT = {
    "timestamp": _timestamp, "run_id": _ID, "stage": _ID,
    "status": _enum("started", "succeeded", "failed", "stopped"),
    "batch_id": _optional(_ID), "task_id": _optional(_ID), "reason_code": _optional(_REASON),
}
_PROVENANCE = (
    "Machine-result fields are explicit caller-supplied observations, not independently authenticated here. "
    "Implementation model_claims and review model_verdict, including confidence, are unverified model reports. "
    "Null means unavailable; no execution or success is inferred. Command hashes record trusted-runner identity, "
    "not independent proof of execution. Frozen-plan provenance is not authenticated by this report. "
    "Bounded text is not semantically checked for secrets; user review is required before sharing."
)


class _Budget:
    def __init__(self, limit):
        self.remaining = limit

    def charge(self, size):
        self.remaining -= size
        if self.remaining < 0:
            raise BatchReportError("REPORT_TOO_LARGE")


def _copy(value, schema, budget):
    """Validate and budget entries before allocating intermediate collections."""
    if isinstance(schema, tuple) and schema[0] == "optional":
        if value is None:
            budget.charge(4)
            return None
        return _copy(value, schema[1], budget)
    if isinstance(schema, dict):
        if not isinstance(value, dict) or value.keys() != schema.keys():
            raise BatchReportError("invalid report record fields")
        budget.charge(2)
        result = {}
        for key, child in schema.items():
            budget.charge(len(key) + 3)
            result[key] = _copy(value[key], child, budget)
        return result
    if isinstance(schema, tuple):
        if not isinstance(value, list) or len(value) > schema[2]:
            raise BatchReportError("invalid bounded array")
        budget.charge(2)
        result = []
        for item in value:
            result.append(_copy(item, schema[1], budget))
        return result
    result = schema(value)
    # Scalars are already bounded; this small serialization is never of a collection.
    budget.charge(len(json.dumps(result, ensure_ascii=True, allow_nan=False)))
    return result


def _require(condition):
    if not condition:
        raise BatchReportError("inconsistent report records")


def _successful_process(result):
    return result["exit_status"] == 0 and not result["timed_out"] and result["cleanup_result"] == "not_needed"


def _check_task(task):
    criteria, paths = task["acceptance_criteria"], task["permitted_files"]
    _require(criteria and paths and len(set(criteria)) == len(criteria) and len(set(paths)) == len(paths))
    for name, started in (("implementer", "implementation_started"), ("reviewer", "review_started")):
        role = task[name]
        if role["session_id"] is not None:
            _require(task[started] is True and role["session_mode"] is not None)
        if role["session_mode"] is not None:
            _require(task[started] is True)
        if role["session_mode"] == "resumed":
            _require(role["session_id"] is not None)
    a, b = task["implementer"]["session_id"], task["reviewer"]["session_id"]
    _require(a is None or b is None or a.lower() != b.lower())
    for field, role, started in (("implementation", "implementer", "implementation_started"),
                                 ("review", "reviewer", "review_started")):
        result = task[field]
        if result is not None:
            _require(task[started] is True and result["session_id"] == task[role]["session_id"])
    implementation = task["implementation"]
    if implementation and implementation["succeeded"]:
        _require(_successful_process(implementation) and implementation["error_code"] is None)
    evidence = task["evidence"]
    if evidence:
        _require(len(set(evidence["changed_paths"])) == len(evidence["changed_paths"]))
        _require(set(evidence["changed_paths"]) <= set(paths))
        if evidence["passed"]:
            _require(evidence["fingerprint"] is not None and evidence["error_code"] is None)
    terminal, approved = task["terminal_results"], task["approved_command_identities"]
    if terminal:
        commands = terminal["command_results"]
        _require(terminal["commands_executed"] == len(commands) <= len(approved))
        first_failure = None
        for index, result in enumerate(commands, 1):
            _require(result["index"] == index and result["command_identity"] == approved[index - 1])
            if not _successful_process(result):
                _require(first_failure is None and index == len(commands))
                first_failure = index
        _require(terminal["first_failed_index"] == first_failure)
        _require(terminal["passed"] == (bool(approved) and len(commands) == len(approved) and first_failure is None))
    review = task["review"]
    if review:
        verdict = review["model_verdict"]
        if review["reviewed_evidence_fingerprint"] is not None:
            _require(evidence is not None and review["reviewed_evidence_fingerprint"] == evidence["fingerprint"])
        if verdict and verdict["decision"] == "APPROVED":
            # Negative/uncertain findings may describe checks in the model's
            # own words. An approval still requires exact, complete coverage,
            # even when the adapter did not accept that model verdict.
            _require(len(verdict["checked_requirements"]) == len(criteria)
                     and set(verdict["checked_requirements"]) == set(criteria))
        if review["approved"]:
            _require(_successful_process(review) and review["error_code"] is None and verdict is not None)
            _require(verdict["decision"] == "APPROVED" and not verdict["blocking_issues"])
            _require(len(verdict["checked_requirements"]) == len(criteria)
                     and set(verdict["checked_requirements"]) == set(criteria))
    staging, publishing = task["staging"], task["publishing"]
    if staging and staging["passed"]:
        _require(staging["staging_attempted"] and staging["error_code"] is None
                 and staging["staged_state_error_code"] is None and staging["staged_changes_sha256"] is not None)
        _require(evidence is not None and staging["staged_paths"] == evidence["changed_paths"])
    if publishing:
        outcome = publishing["outcome"]
        _require(not publishing["push_attempted"] or publishing["commit_attempted"])
        _require(publishing["push_attempted"] or publishing["remote_sha"] is None)
        if outcome == "NOT_ATTEMPTED":
            _require(not publishing["commit_attempted"] and not publishing["push_attempted"]
                     and publishing["commit_sha"] is None and publishing["remote_sha"] is None)
        else:
            _require(publishing["commit_attempted"])
        if outcome == "COMMIT_FAILED_OR_AMBIGUOUS":
            _require(not publishing["push_attempted"] and publishing["commit_sha"] is None)
        if outcome == "COMMIT_CREATED_NOT_PUSHED":
            _require(not publishing["push_attempted"] and publishing["observed_head_sha"] is not None)
        if outcome in {"PUSH_FAILED_OR_AMBIGUOUS", "PUBLISHED"}:
            _require(publishing["push_attempted"] and publishing["commit_sha"] is not None)
        if publishing["commit_sha"] is not None:
            _require(publishing["observed_head_sha"] == publishing["commit_sha"])
        if outcome == "PUBLISHED":
            _require(publishing["error_code"] is None and publishing["remote_sha"] == publishing["commit_sha"])
            _require(staging is not None and staging["passed"] and review is not None and review["approved"])
            _require(evidence is not None and evidence["passed"] and terminal is not None and terminal["passed"])
            _require(review["reviewed_packet_sha256"] is not None
                     and review["reviewed_evidence_fingerprint"] is not None)
        else:
            _require(publishing["error_code"] is not None)
    if task["status"] == "stopped":
        _require(task["failure_stage"] is not None and task["stop_reason_code"] is not None)
    else:
        _require(task["failure_stage"] is None and task["stop_reason_code"] is None)
    if task["status"] == "pending":
        _require(task["implementation_started"] in (None, False) and task["review_started"] in (None, False))
        _require(all(task[f] is None for f in ("implementation", "evidence", "terminal_results", "review", "staging", "publishing")))
    if task["status"] == "completed":
        _require(implementation is not None and implementation["succeeded"])
        _require(evidence is not None and evidence["passed"] and terminal is not None and terminal["passed"])
        _require(review is not None and review["approved"] and publishing is not None and publishing["outcome"] == "PUBLISHED")


def _escaped(value):
    text = html.escape(str(value), quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text).replace("\n", "<br>").replace("\t", "    ")


def _markdown(data, limit):
    parts, size = [], 0

    def append(line):
        nonlocal size
        encoded = (line + "\n").encode("utf-8")
        size += len(encoded)
        if size > limit:
            raise BatchReportError("REPORT_TOO_LARGE")
        parts.append(encoded)

    def render(value, indent=0):
        if isinstance(value, dict):
            for key, child in sorted(value.items()):
                label = _escaped(key.replace("_", " "))
                if isinstance(child, (dict, list)):
                    append("  " * indent + "- **" + label + "**:")
                    render(child, indent + 1)
                else:
                    append("  " * indent + "- **" + label + "**: " + ("Unavailable" if child is None else _escaped(child)))
        elif isinstance(value, list):
            if not value:
                append("  " * indent + "- (none recorded)")
            for item in value:
                if isinstance(item, dict):
                    append("  " * indent + "- Record:")
                    render(item, indent + 1)
                else:
                    append("  " * indent + "- " + _escaped(item))

    append("# Auto-Coding batch report")
    append("")
    append(_escaped(_PROVENANCE))
    append("")
    for key in ("checkpoint", "batch"):
        append("## " + key.title())
        append("")
        render(data[key])
        append("")
    for index, task in enumerate(data["tasks"], 1):
        append("## Task " + str(index) + ": " + _escaped(task["task_id"]) + " — " + _escaped(task["title"]))
        append("")
        render(task)
        append("")
    append("## Operational events")
    append("")
    render(data["events"])
    return b"".join(parts)


def build_batch_report(checkpoint, batch, tasks, events, max_bytes: int) -> BatchReport:
    """Build schema v1 JSON and Markdown; each output must fit max_bytes.

All schema keys are required; unavailable optional values must be explicit None.
Tasks are supplied in approved order (not independently authenticated here).
Task records use _TASK's bounded field contract, mirroring safe adapter fields
with explicit model_claims/model_verdict labels; no raw result objects are dumped.
Operational events use the logger's fields, with absent optional keys set to None.
"""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise BatchReportError("max_bytes must be a positive integer")
    budget = _Budget(max_bytes)
    try:
        checkpoint = _checkpoint(checkpoint)
    except CheckpointError:
        raise BatchReportError("invalid checkpoint") from None
    _require(checkpoint["status"] in {"completed", "stopped"})
    budget.charge(len(json.dumps(checkpoint, ensure_ascii=True)))
    batch = _copy(batch, _BATCH, budget)
    tasks = _copy(tasks, _array(_TASK), budget)
    events = _copy(events, _array(_EVENT, 1000), budget)
    _require(tasks and len({t["task_id"] for t in tasks}) == len(tasks))
    if batch["started_at"] is not None and batch["ended_at"] is not None:
        _require(datetime.fromisoformat(batch["started_at"]) <= datetime.fromisoformat(batch["ended_at"]))
    for task in tasks:
        _check_task(task)
        publishing = task["publishing"]
        if publishing is not None:
            publishing["remote_verification"] = (
                "verified" if publishing["outcome"] == "PUBLISHED" else
                "mismatch" if publishing["remote_sha"] is not None and publishing["commit_sha"] is not None
                and publishing["remote_sha"] != publishing["commit_sha"] else "not_confirmed"
            )
    completed = sum(t["status"] == "completed" for t in tasks)
    stopped = sum(t["status"] == "stopped" for t in tasks)
    if checkpoint["status"] == "completed":
        _require(completed == len(tasks))
    sequential = checkpoint["schema_version"] == 2
    if sequential:
        _require([task["task_id"] for task in tasks] == checkpoint["task_ids"])
        entries = checkpoint["publications"]
        for entry, task in zip(entries, tasks):
            _require(task["status"] == ("completed" if entry["finalized"] else "stopped")
                     and task["publishing"] is not None and task["publishing"]["outcome"] == "PUBLISHED")
            _require(task["publishing"]["commit_sha"] == entry["commit_sha"]
                     and task["publishing"]["remote_sha"] == entry["remote_sha"])
        _require(completed == sum(entry["finalized"] for entry in entries))
    current = checkpoint["current_task_id"]
    if current is not None:
        matching = [t for t in tasks if t["task_id"] == current]
        _require(len(matching) == 1 and matching[0]["status"] in {"running", "stopped"})
        task = matching[0]
        if task["status"] == "stopped":
            _require(task["failure_stage"] == checkpoint["stage"]
                     and task["stop_reason_code"] == checkpoint["stop_reason_code"])
        if not sequential:
            for field, role in (("implementer_session_id", "implementer"), ("reviewer_session_id", "reviewer")):
                _require(checkpoint[field] == task[role]["session_id"])
    identifiers = {t["task_id"] for t in tasks}
    for event in events:
        _require(event["batch_id"] in (None, checkpoint["batch_id"]))
        _require(event["task_id"] is None or event["task_id"] in identifiers)
    batch.update(planned_tasks=len(tasks), completed_tasks=completed, stopped_tasks=stopped)
    data = {"schema_version": 1, "provenance": _PROVENANCE, "checkpoint": checkpoint,
            "batch": batch, "tasks": tasks, "events": events}
    # Input collections have already been incrementally bounded. The final
    # canonical size check remains authoritative, including exact equality.
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(raw) > max_bytes:
        raise BatchReportError("REPORT_TOO_LARGE")
    return BatchReport(raw, _markdown(data, max_bytes))


def export_batch_report(
    checkpoint, batch, tasks, events, max_bytes: int, *,
    destination: str | Path, repository_directory: str | Path,
) -> tuple[Path, Path]:
    """Export validated reports to an existing directory outside the target.

Each complete file is published atomically without overwrite using a hard link.
The pair is not filesystem-atomic. Ordinary failures roll back files created by
this call; ReportExportError.partial_paths identifies any rollback failures.
An abrupt process/power loss can leave one report or temporary files. No retry,
locking, directory fsync, or crash-durability guarantee is provided.
"""
    report = build_batch_report(checkpoint, batch, tasks, events, max_bytes)
    repository = Path(repository_directory).resolve(strict=True)
    directory = Path(destination).resolve(strict=True)
    if (not repository.is_dir() or not directory.is_dir() or directory == repository
            or repository in directory.parents):
        raise BatchReportError("UNSAFE_EXPORT_LOCATION")
    paths = (directory / "batch-report.json", directory / "batch-report.md")
    created, temporary = [], []
    try:
        if any(os.path.lexists(path) for path in paths):
            raise FileExistsError("report destination already exists")
        for path, raw in zip(paths, (report.json_bytes, report.markdown_bytes)):
            with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".batch-report-", delete=False) as output:
                source = Path(output.name)
                temporary.append(source)
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.link(source, path)
            created.append(path)
        for source in temporary:
            source.unlink()
        return paths
    except BaseException as error:
        remaining = []
        for path in created + temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                remaining.append(path)
        if isinstance(error, (KeyboardInterrupt, SystemExit)) and not remaining:
            raise
        raise ReportExportError(tuple(remaining)) from None
