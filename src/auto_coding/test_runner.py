"""Execute approved test commands without interpreting or modifying them."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

from auto_coding.integrations.windows_process import run_windows_process
from auto_coding.plans import _test_contract
from auto_coding.test_protection import check_test_protection


@dataclass(frozen=True)
class TestDiagnostic:
    """Fixed controller codes; never process output or a model diagnosis.

    APPLICATION_FAILURE is the approved harness's exit-status-v1 assertion,
    with verified declared inputs, not independent proof of a code defect.
    """

    classification: str
    reason_code: str
    protocol: str | None = None

    def record(self) -> dict:
        return {"classification": self.classification, "reason_code": self.reason_code, "protocol": self.protocol}


def _diagnose(process, protocol=None, protection_error=None):
    if process.timed_out is True:
        return TestDiagnostic("TIMEOUT", "TEST_TIMEOUT", protocol)
    if process.cleanup_result == "launch_failed":
        return TestDiagnostic("LAUNCH_FAILURE", "TEST_LAUNCH_FAILED", protocol)
    if process.cleanup_result != "not_needed":
        return TestDiagnostic("CLEANUP_FAILURE", "TEST_CLEANUP_UNCERTAIN", protocol)
    if process.timed_out is not False or type(process.exit_status) is not int:
        return TestDiagnostic("UNKNOWN", "TEST_RESULT_INVALID", protocol)
    if protection_error:
        return TestDiagnostic("INFRASTRUCTURE_FAILURE", protection_error, protocol)
    if process.exit_status == 0:
        return TestDiagnostic("PASSED", "TEST_PASSED", protocol)
    if protocol == "exit-status-v1":
        if process.exit_status == 10:
            return TestDiagnostic("APPLICATION_FAILURE", "APPROVED_FUNCTIONAL_FAILURE", protocol)
        if process.exit_status == 20:
            return TestDiagnostic("INFRASTRUCTURE_FAILURE", "APPROVED_INFRASTRUCTURE_FAILURE", protocol)
        return TestDiagnostic("UNKNOWN", "UNRECOGNIZED_EXIT_STATUS", protocol)
    return TestDiagnostic("UNKNOWN", "TEST_PROTOCOL_UNAVAILABLE", protocol)


@dataclass(frozen=True)
class TestCommandResult:
    index: int
    exit_status: int | None
    elapsed_seconds: float
    timed_out: bool
    cleanup_result: str
    command_identity: str = ""
    diagnostic: TestDiagnostic | None = None


@dataclass(frozen=True)
class TestRunResult:
    passed: bool
    commands_executed: int
    command_results: tuple[TestCommandResult, ...]
    first_failed_index: int | None
    diagnostic: TestDiagnostic | None = None


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


def _command_identity(command: list[str]) -> str:
    """Hash the executable and ordered argv with an unambiguous encoding."""
    digest = hashlib.sha256()
    digest.update(b"auto-coding-test-command-v1\0")
    digest.update(len(command).to_bytes(8, "big"))
    for argument in command:
        encoded = argument.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def run_approved_tests(
    repository_directory: str | Path,
    test_commands: list[list[str]],
    timeout_seconds: float,
    *, test_contract: dict | None = None,
) -> TestRunResult:
    """Run original argv once, with optional approval-bound status semantics.

    exit-status-v1: 0=pass, 10=application assertion/functional failure,
    20=infrastructure failure; every other nonzero status is UNKNOWN. Only an
    explicitly approved, trusted harness may use this protocol. Text on either
    output stream is discarded and cannot activate classification. An omitted
    contract preserves execution and pass/fail behavior but never permits repair.
    """
    commands = _validate_inputs(test_commands, timeout_seconds)
    results: list[TestCommandResult] = []
    if test_contract is not None:
        _test_contract(test_contract, len(commands))

    for index, command in enumerate(commands, start=1):
        protocol = None if test_contract is None else test_contract["failure_protocols"][index - 1]
        if test_contract is not None:
            protection = check_test_protection(repository_directory, test_contract, len(commands))
            if not protection.passed:
                diagnostic = TestDiagnostic("INFRASTRUCTURE_FAILURE", protection.error_code, protocol)
                return TestRunResult(False, len(results), tuple(results), None, diagnostic)
        command_identity = _command_identity(command)
        started = time.monotonic()
        process = run_windows_process(
            command,
            input_bytes=b"",
            cwd=repository_directory,
            timeout_seconds=timeout_seconds,
        )
        elapsed = time.monotonic() - started
        protection_error = None
        if test_contract is not None:
            protection_error = check_test_protection(repository_directory, test_contract, len(commands)).error_code
        diagnostic = _diagnose(process, protocol, protection_error)
        results.append(TestCommandResult(
            index=index,
            exit_status=process.exit_status,
            elapsed_seconds=elapsed,
            timed_out=process.timed_out,
            cleanup_result=process.cleanup_result,
            command_identity=command_identity,
            diagnostic=diagnostic,
        ))
        if (
            process.exit_status != 0
            or process.timed_out
            or process.cleanup_result != "not_needed"
            or diagnostic.classification != "PASSED"
        ):
            return TestRunResult(False, len(results), tuple(results), index, diagnostic)

    return TestRunResult(True, len(results), tuple(results), None, TestDiagnostic("PASSED", "APPROVED_TESTS_PASSED"))
