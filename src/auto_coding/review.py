"""Run one independent, read-only Codex review and validate its verdict."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
import uuid

from auto_coding.integrations.session_events import MAX_JSONL_BYTES, session_from_jsonl
from auto_coding.integrations.windows_process import run_windows_process


_MAX_VERDICT_BYTES = 64 * 1024
_FIELDS = {
    "decision", "blocking_issues", "nonblocking_notes", "checked_requirements", "confidence",
}
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVED", "REJECTED", "UNCERTAIN"]},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "nonblocking_notes": {"type": "array", "items": {"type": "string"}},
        "checked_requirements": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
    },
    "required": sorted(_FIELDS),
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ReviewVerdict:
    decision: str
    blocking_issues: tuple[str, ...]
    nonblocking_notes: tuple[str, ...]
    checked_requirements: tuple[str, ...]
    confidence: str


@dataclass(frozen=True)
class ReviewResult:
    approved: bool
    verdict: ReviewVerdict | None
    error_code: str | None
    exit_status: int | None
    timed_out: bool
    cleanup_result: str
    session_id: str | None = None


def _validate_inputs(
    prompt: object, model: object, reasoning_effort: object,
    timeout_seconds: object, acceptance_criteria: object,
) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("review prompt must be a nonempty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("review model must be a nonempty string")
    if reasoning_effort not in ("low", "medium", "high", "xhigh"):
        raise ValueError("reasoning_effort is invalid")
    try:
        valid_timeout = (
            not isinstance(timeout_seconds, bool)
            and isinstance(timeout_seconds, (int, float))
            and math.isfinite(timeout_seconds)
            and timeout_seconds > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout_seconds must be positive and finite")
    if not isinstance(acceptance_criteria, list) or not acceptance_criteria or any(
        not isinstance(criterion, str) or not criterion.strip()
        for criterion in acceptance_criteria
    ):
        raise ValueError("acceptance_criteria must be a nonempty string array")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _parse_verdict(data: object) -> ReviewVerdict:
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError("invalid verdict fields")
    if data["decision"] not in ("APPROVED", "REJECTED", "UNCERTAIN"):
        raise ValueError("invalid decision")
    if data["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError("invalid confidence")
    for field in ("blocking_issues", "nonblocking_notes", "checked_requirements"):
        values = data[field]
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError("invalid verdict array")
    return ReviewVerdict(
        decision=data["decision"],
        blocking_issues=tuple(data["blocking_issues"]),
        nonblocking_notes=tuple(data["nonblocking_notes"]),
        checked_requirements=tuple(data["checked_requirements"]),
        confidence=data["confidence"],
    )


def run_review(
    prompt: str,
    model: str,
    reasoning_effort: str,
    repository_directory: str | Path,
    timeout_seconds: float,
    acceptance_criteria: list[str],
    *,
    session_id: str | None = None,
) -> ReviewResult:
    """Run a fresh or explicitly resumed reviewer through the same verdict gate."""
    _validate_inputs(prompt, model, reasoning_effort, timeout_seconds, acceptance_criteria)
    if session_id is not None:
        try:
            parsed_id = uuid.UUID(session_id) if isinstance(session_id, str) else None
        except ValueError:
            parsed_id = None
        if parsed_id is None or parsed_id.int == 0 or str(parsed_id) != session_id.lower():
            return ReviewResult(False, None, "SESSION_ID_INVALID", None, False, "not_needed")
    try:
        repository = Path(repository_directory).resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if temporary_root == repository or repository in temporary_root.parents:
            return ReviewResult(False, None, "UNSAFE_TEMP_LOCATION", None, False, "not_needed")

        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            schema_path = Path(temporary) / "verdict-schema.json"
            verdict_path = Path(temporary) / "verdict.json"
            schema_path.write_text(json.dumps(_VERDICT_SCHEMA), encoding="utf-8")
            command = ["codex", "--ask-for-approval", "never", "exec"]
            if session_id is not None:
                command.append("resume")
            command.extend([
                "--model", model,
                "-c", f"model_reasoning_effort={reasoning_effort}",
            ])
            if session_id is not None:
                command.extend(["-c", 'sandbox_mode="read-only"'])
            else:
                command.extend(["--sandbox", "read-only"])
            command.extend([
                "--output-schema", str(schema_path),
                "--output-last-message", str(verdict_path),
                "--json",
            ])
            if session_id is not None:
                command.append(session_id)
            command.append("-")
            with tempfile.TemporaryFile(mode="w+b", dir=temporary) as output_file:
                process = run_windows_process(
                    command,
                    input_bytes=prompt.encode("utf-8"),
                    cwd=repository_directory,
                    timeout_seconds=timeout_seconds,
                    stdout_file=output_file,
                )
                if process.timed_out:
                    return ReviewResult(False, None, "PROCESS_TIMEOUT", process.exit_status, True,
                                        process.cleanup_result)
                if process.cleanup_result == "launch_failed":
                    return ReviewResult(False, None, "PROCESS_LAUNCH_FAILED", process.exit_status,
                                        False, process.cleanup_result)
                if process.cleanup_result != "not_needed":
                    return ReviewResult(False, None, "PROCESS_CLEANUP_UNCERTAIN",
                                        process.exit_status, False, process.cleanup_result)
                if process.exit_status != 0:
                    return ReviewResult(False, None, "PROCESS_FAILED", process.exit_status,
                                        False, process.cleanup_result)

                output_file.seek(0)
                returned_id, error = session_from_jsonl(output_file.read(MAX_JSONL_BYTES + 1))
                if error is not None:
                    return ReviewResult(False, None, error, 0, False, "not_needed")
                if session_id is not None and returned_id.lower() != session_id.lower():
                    return ReviewResult(False, None, "SESSION_ID_MISMATCH", 0, False,
                                        "not_needed")

            try:
                with verdict_path.open("rb") as verdict_file:
                    raw = verdict_file.read(_MAX_VERDICT_BYTES + 1)
            except FileNotFoundError:
                return ReviewResult(False, None, "VERDICT_MISSING", 0, False, "not_needed")
            if len(raw) > _MAX_VERDICT_BYTES:
                return ReviewResult(False, None, "VERDICT_TOO_LARGE", 0, False, "not_needed")
            try:
                data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                return ReviewResult(False, None, "VERDICT_INVALID_JSON", 0, False, "not_needed")
            try:
                verdict = _parse_verdict(data)
            except (TypeError, ValueError):
                return ReviewResult(False, None, "VERDICT_INVALID", 0, False, "not_needed")
            if verdict.decision != "APPROVED":
                return ReviewResult(False, verdict, None, 0, False, "not_needed", returned_id)
            if verdict.blocking_issues:
                return ReviewResult(False, verdict, "BLOCKING_ISSUES", 0, False,
                                    "not_needed", returned_id)
            if not all(
                criterion in verdict.checked_requirements for criterion in acceptance_criteria
            ):
                return ReviewResult(False, verdict, "CRITERIA_NOT_COVERED", 0, False,
                                    "not_needed", returned_id)
            return ReviewResult(True, verdict, None, 0, False, "not_needed", returned_id)
    except OSError:
        return ReviewResult(False, None, "TEMP_IO_FAILED", None, False, "not_needed")
