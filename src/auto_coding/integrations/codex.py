"""Run one Codex implementation prompt."""

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
import uuid

from auto_coding.integrations.windows_process import run_windows_process


_MAX_JSONL_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ImplementationOutcome:
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    succeeded: bool
    cleanup_result: str = "not_needed"
    session_id: str | None = None
    error_code: str | None = None


def _session_from_jsonl(output: bytes) -> tuple[str | None, str | None]:
    if len(output) > _MAX_JSONL_BYTES:
        return None, "OUTPUT_TOO_LARGE"

    session_id = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "SESSION_EVIDENCE_MALFORMED"
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            return None, "SESSION_EVIDENCE_MALFORMED"
        if event["type"] != "thread.started":
            continue
        value = event.get("thread_id")
        if not isinstance(value, str):
            return None, "SESSION_ID_INVALID"
        try:
            parsed = uuid.UUID(value)
        except ValueError:
            return None, "SESSION_ID_INVALID"
        if parsed.int == 0 or str(parsed) != value.lower():
            return None, "SESSION_ID_INVALID"
        if session_id is not None:
            return None, "SESSION_ID_AMBIGUOUS"
        session_id = value

    if session_id is None:
        return None, "SESSION_ID_MISSING"
    return session_id, None


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
        command = [
            "codex", "--ask-for-approval", "never", "exec", "resume",
            "--model", model,
            "-c", f"model_reasoning_effort={reasoning_effort}",
            "-c", 'sandbox_mode="workspace-write"',
            "--json", session_id, "-",
        ]
    else:
        command = [
            "codex", "--ask-for-approval", "never", "exec",
            "--model", model,
            "-c", f"model_reasoning_effort={reasoning_effort}",
            "--sandbox", "workspace-write",
            "--json", "-",
        ]
    try:
        repository = Path(repository_directory).resolve()
        temporary_root = Path(tempfile.gettempdir()).resolve()
        if temporary_root == repository or repository in temporary_root.parents:
            return ImplementationOutcome(None, time.monotonic() - started, False, False,
                                         error_code="UNSAFE_TEMP_LOCATION")
        with tempfile.TemporaryFile(mode="w+b", dir=temporary_root) as output_file:
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
            return ImplementationOutcome(result.exit_status, time.monotonic() - started,
                                         False, error is None, result.cleanup_result,
                                         returned_id, error)
    except OSError:
        return ImplementationOutcome(None, time.monotonic() - started, False, False,
                                     error_code="OUTPUT_UNAVAILABLE")
