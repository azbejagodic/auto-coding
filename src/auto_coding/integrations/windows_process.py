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
    stdin_write_status: str = "incomplete"


class _ObservedStdin:
    """Observe the existing pipe without adding buffering or a writer thread."""

    def __init__(self, stream: BinaryIO, expected_bytes: int):
        self.stream = stream
        self.expected_bytes = expected_bytes
        self.written_bytes = 0
        self.failed = False
        self.closed_successfully = False

    def write(self, data):
        try:
            written = self.stream.write(data)
        except (OSError, ValueError):
            self.failed = True
            # Observe failures without changing subprocess's exception
            # classification or suppression behavior.
            raise
        if type(written) is not int or written != len(data):
            self.failed = True
        elif not self.failed:
            self.written_bytes += written
        return written

    def close(self):
        try:
            self.stream.close()
        except (OSError, ValueError):
            self.failed = True
            raise
        self.closed_successfully = True

    def flush(self):
        try:
            return self.stream.flush()
        except (OSError, ValueError):
            self.failed = True
            raise

    def __getattr__(self, name):
        return getattr(self.stream, name)

    @property
    def status(self) -> str:
        if self.expected_bytes == 0:
            return "not_requested"
        if (not self.failed and self.closed_successfully
                and self.written_bytes == self.expected_bytes):
            return "complete"
        return "incomplete"


def run_windows_process(
    command: list[str],
    *,
    input_bytes: bytes,
    cwd: str | Path,
    timeout_seconds: float,
    cleanup_timeout_seconds: float = 5.0,
    stdout_file: BinaryIO | None = None,
) -> WindowsProcessResult:
    """Run once and report cleanup and observed stdin submission.

    Stdin status is not_requested for empty input, complete only when all
    bytes were observed written and the stream closed successfully, and
    incomplete otherwise. Pipe submission does not prove application
    consumption or understanding. Unobserved writes also remain incomplete.
    """
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
        return WindowsProcessResult(
            None, False, "launch_failed", "incomplete" if input_bytes else "not_requested",
        )

    stdin = _ObservedStdin(process.stdin, len(input_bytes))
    process.stdin = stdin

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
            return WindowsProcessResult(None, True, "parent_not_reaped", stdin.status)
        return WindowsProcessResult(None, True, cleanup_result, stdin.status)

    return WindowsProcessResult(process.returncode, False, "not_needed", stdin.status)
