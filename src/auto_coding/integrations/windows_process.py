"""Run subprocesses with bounded Windows process-tree cleanup on timeout.

Taskkill cannot guarantee containment of descendants that already detached.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import BinaryIO


@dataclass(frozen=True)
class WindowsProcessResult:
    exit_status: int | None
    timed_out: bool
    cleanup_result: str


def run_windows_process(
    command: list[str],
    *,
    input_bytes: bytes,
    cwd: str | Path,
    timeout_seconds: float,
    cleanup_timeout_seconds: float = 5.0,
    stdout_file: BinaryIO | None = None,
) -> WindowsProcessResult:
    """Run once, optionally redirect stdout, and report timeout cleanup."""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout_file if stdout_file is not None else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            shell=False,
        )
    except OSError:
        return WindowsProcessResult(None, False, "launch_failed")

    try:
        process.communicate(input=input_bytes, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        cleanup_result = "taskkill_completed"
        try:
            taskkill = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
                timeout=cleanup_timeout_seconds,
            )
            if taskkill.returncode != 0:
                cleanup_result = "taskkill_failed"
        except subprocess.TimeoutExpired:
            cleanup_result = "taskkill_timed_out"
        except OSError:
            cleanup_result = "taskkill_failed"

        if cleanup_result != "taskkill_completed":
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=cleanup_timeout_seconds)
        except (OSError, subprocess.TimeoutExpired):
            return WindowsProcessResult(None, True, "parent_not_reaped")
        return WindowsProcessResult(None, True, cleanup_result)

    return WindowsProcessResult(process.returncode, False, "not_needed")
