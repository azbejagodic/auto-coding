"""Run one Codex implementation prompt."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
import time
import uuid

from auto_coding.integrations.session_events import (
    MAX_JSONL_BYTES as _MAX_JSONL_BYTES,
    session_from_jsonl as _session_from_jsonl,
)
from auto_coding.integrations.windows_process import run_windows_process


_MAX_REPORT_BYTES = 64 * 1024
_TEXT_LIMIT = 2000
_PATH_LIMIT = 512
_MAX_ITEMS = 100
_REPORT_FIELDS = {"summary", "approach", "files_reported", "confidence", "concerns"}
_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": _TEXT_LIMIT},
        "approach": {"type": "string", "minLength": 1, "maxLength": _TEXT_LIMIT},
        "files_reported": {"type": "array", "maxItems": _MAX_ITEMS,
                           "items": {"type": "string", "minLength": 1,
                                     "maxLength": _PATH_LIMIT}},
        "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "concerns": {"type": "array", "maxItems": _MAX_ITEMS,
                     "items": {"type": "string", "minLength": 1,
                               "maxLength": _TEXT_LIMIT}},
    },
    "required": sorted(_REPORT_FIELDS),
    "additionalProperties": False,
}


@dataclass(frozen=True, repr=False)
class ImplementationReport:
    """Unverified model claims; callers must check before displaying them."""

    summary: str
    approach: str
    files_reported: tuple[str, ...]
    confidence: str
    concerns: tuple[str, ...]


@dataclass(frozen=True)
class ImplementationOutcome:
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    succeeded: bool
    cleanup_result: str = "not_needed"
    session_id: str | None = None
    error_code: str | None = None
    report: ImplementationReport | None = field(default=None, repr=False)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _parse_report(raw: bytes) -> tuple[ImplementationReport | None, str | None]:
    if len(raw) > _MAX_REPORT_BYTES:
        return None, "REPORT_TOO_LARGE"
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, "REPORT_INVALID_JSON"
    if not isinstance(data, dict) or set(data) != _REPORT_FIELDS:
        return None, "REPORT_INVALID"
    if any(not isinstance(data[name], str) or not data[name].strip()
           or len(data[name]) > _TEXT_LIMIT for name in ("summary", "approach")):
        return None, "REPORT_INVALID"
    if data["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return None, "REPORT_INVALID"
    for name, limit in (("files_reported", _PATH_LIMIT), ("concerns", _TEXT_LIMIT)):
        values = data[name]
        if not isinstance(values, list) or len(values) > _MAX_ITEMS or any(
            not isinstance(value, str) or not value.strip() or len(value) > limit
            for value in values
        ):
            return None, "REPORT_INVALID"
    return ImplementationReport(data["summary"], data["approach"],
                                tuple(data["files_reported"]), data["confidence"],
                                tuple(data["concerns"])), None


def run_implementation(
    prompt: str,
    model: str,
    reasoning_effort: str,
    repository_directory: str | Path,
    timeout_seconds: float,
    *,
    session_id: str | None = None,
) -> ImplementationOutcome:
    """Run a fresh or explicitly identified Codex session and verify its ID."""
    started = time.monotonic()
    if session_id is not None:
        try:
            parsed_id = uuid.UUID(session_id) if isinstance(session_id, str) else None
        except ValueError:
            parsed_id = None
        if parsed_id is None or parsed_id.int == 0 or str(parsed_id) != session_id.lower():
            return ImplementationOutcome(None, time.monotonic() - started, False, False,
                                         error_code="SESSION_ID_INVALID")
    try:
        repository = Path(repository_directory).resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if temporary_root == repository or repository in temporary_root.parents:
            return ImplementationOutcome(None, time.monotonic() - started, False, False,
                                         error_code="UNSAFE_TEMP_LOCATION")
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            schema_path = Path(temporary) / "implementation-schema.json"
            report_path = Path(temporary) / "implementation-report.json"
            schema_path.write_text(json.dumps(_REPORT_SCHEMA), encoding="utf-8")
            command = ["codex", "--ask-for-approval", "never", "exec"]
            if session_id is not None:
                command.append("resume")
            command.extend(["--model", model, "-c",
                            f"model_reasoning_effort={reasoning_effort}"])
            if session_id is not None:
                command.extend(["-c", 'sandbox_mode="workspace-write"'])
            else:
                command.extend(["--sandbox", "workspace-write"])
            command.extend(["--json", "--output-schema", str(schema_path),
                            "--output-last-message", str(report_path)])
            if session_id is not None:
                command.append(session_id)
            command.append("-")

            with tempfile.TemporaryFile(mode="w+b", dir=temporary) as output_file:
                result = run_windows_process(
                    command,
                    input_bytes=prompt.encode("utf-8"),
                    cwd=repository_directory,
                    timeout_seconds=timeout_seconds,
                    stdout_file=output_file,
                )
                if result.timed_out:
                    error = "PROCESS_TIMEOUT"
                elif result.cleanup_result == "launch_failed":
                    error = "PROCESS_LAUNCH_FAILED"
                elif result.cleanup_result != "not_needed":
                    error = "PROCESS_CLEANUP_UNCERTAIN"
                elif result.exit_status != 0:
                    error = "PROCESS_FAILED"
                else:
                    error = None
                if error is not None:
                    return ImplementationOutcome(result.exit_status, time.monotonic() - started,
                                                 result.timed_out, False, result.cleanup_result,
                                                 error_code=error)

                output_file.seek(0)
                returned_id, error = _session_from_jsonl(output_file.read(_MAX_JSONL_BYTES + 1))
                if error is None and session_id is not None and returned_id.lower() != session_id.lower():
                    return ImplementationOutcome(result.exit_status, time.monotonic() - started,
                                                 False, False, result.cleanup_result,
                                                 error_code="SESSION_ID_MISMATCH")
                if error is not None:
                    return ImplementationOutcome(result.exit_status, time.monotonic() - started,
                                                 False, False, result.cleanup_result,
                                                 error_code=error)

            try:
                with report_path.open("rb") as report_file:
                    raw = report_file.read(_MAX_REPORT_BYTES + 1)
            except FileNotFoundError:
                return ImplementationOutcome(0, time.monotonic() - started, False, False,
                                             result.cleanup_result, returned_id,
                                             "REPORT_MISSING")
            report, error = _parse_report(raw)
            return ImplementationOutcome(0, time.monotonic() - started, False,
                                         error is None, result.cleanup_result,
                                         returned_id, error, report)
    except OSError:
        return ImplementationOutcome(None, time.monotonic() - started, False, False,
                                     error_code="OUTPUT_UNAVAILABLE")
