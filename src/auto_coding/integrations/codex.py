"""Run one Codex implementation prompt."""

from dataclasses import dataclass
from pathlib import Path
import time

from auto_coding.integrations.windows_process import run_windows_process


@dataclass(frozen=True)
class ImplementationOutcome:
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    succeeded: bool
    cleanup_result: str = "not_needed"


def run_implementation(
    prompt: str,
    model: str,
    reasoning_effort: str,
    repository_directory: str | Path,
    timeout_seconds: float,
) -> ImplementationOutcome:
    """Run a fresh non-interactive Codex process without retaining its output."""
    command = [
        "codex", "--ask-for-approval", "never", "exec",
        "--model", model,
        "-c", f"model_reasoning_effort={reasoning_effort}",
        "--sandbox", "workspace-write",
        "-",
    ]
    started = time.monotonic()
    result = run_windows_process(
        command,
        input_bytes=prompt.encode("utf-8"),
        cwd=repository_directory,
        timeout_seconds=timeout_seconds,
    )
    return ImplementationOutcome(
        result.exit_status,
        time.monotonic() - started,
        result.timed_out,
        result.exit_status == 0 and not result.timed_out,
        result.cleanup_result,
    )
