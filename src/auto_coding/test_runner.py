"""Execute approved test commands without interpreting or modifying them."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

from auto_coding.integrations.windows_process import run_windows_process


@dataclass(frozen=True)
class TestCommandResult:
    index: int
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    cleanup_result: str


@dataclass(frozen=True)
class TestRunResult:
    passed: bool
    commands_executed: int
    command_results: tuple[TestCommandResult, ...]
    first_failed_index: int | None


def _validate_inputs(test_commands: object, timeout_seconds: object) -> list[list[str]]:
    if not isinstance(test_commands, list) or not test_commands:
        raise ValueError("test_commands must be a nonempty array")
    for index, command in enumerate(test_commands):
        if not isinstance(command, list) or not command:
            raise ValueError(f"test_commands[{index}] must be a nonempty argument array")
        if any(not isinstance(argument, str) or not argument.strip() for argument in command):
            raise ValueError(f"test_commands[{index}] arguments must be nonempty strings")
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
        raise ValueError("timeout_seconds must be a positive finite number")
    return test_commands


def run_approved_tests(
    repository_directory: str | Path,
    test_commands: list[list[str]],
    timeout_seconds: float,
) -> TestRunResult:
    """Run approved argument arrays sequentially and stop at the first failure."""
    commands = _validate_inputs(test_commands, timeout_seconds)
    results: list[TestCommandResult] = []

    for index, command in enumerate(commands, start=1):
        started = time.monotonic()
        process = run_windows_process(
            command,
            input_bytes=b"",
            cwd=repository_directory,
            timeout_seconds=timeout_seconds,
        )
        elapsed = time.monotonic() - started
        results.append(TestCommandResult(
            index=index,
            exit_status=process.exit_status,
            elapsed_seconds=elapsed,
            timed_out=process.timed_out,
            cleanup_result=process.cleanup_result,
        ))
        if (
            process.exit_status != 0
            or process.timed_out
            or process.cleanup_result != "not_needed"
        ):
            return TestRunResult(False, len(results), tuple(results), index)

    return TestRunResult(True, len(results), tuple(results), None)
