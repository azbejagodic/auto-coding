"""Run one Codex implementation prompt."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import uuid

from auto_coding.integrations.session_events import (
    MAX_JSONL_BYTES as _MAX_JSONL_BYTES,
    session_from_jsonl as _session_from_jsonl,
)
from auto_coding.integrations.windows_process import run_windows_process


@dataclass(frozen=True)
class ImplementationOutcome:
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    succeeded: bool
    cleanup_result: str = "not_needed"
    session_id: str | None = None
    error_code: str | None = None


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
