"""Run one Codex implementation prompt.

Windows child-process-tree cleanup must be completed before unattended live execution.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time


@dataclass(frozen=True)
class ImplementationOutcome:
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    succeeded: bool


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
    try:
        result = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            cwd=repository_directory,
            timeout=timeout_seconds,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ImplementationOutcome(None, time.monotonic() - started, True, False)
    except OSError:
        return ImplementationOutcome(None, time.monotonic() - started, False, False)
    return ImplementationOutcome(
        result.returncode, time.monotonic() - started, False, result.returncode == 0
    )
