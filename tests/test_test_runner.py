"""Tests for execution of approved terminal test commands."""

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import call, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import WindowsProcessResult
from auto_coding.test_runner import run_approved_tests
from auto_coding.test_protection import TestProtectionResult


class ApprovedTestRunnerTests(unittest.TestCase):
    @staticmethod
    def expected_command_identity(command: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(b"auto-coding-test-command-v1\0")
        digest.update(len(command).to_bytes(8, "big"))
        for argument in command:
            encoded = argument.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def test_command_identity_is_stable_and_covers_executable_and_ordered_arguments(self) -> None:
        original = ["tool", "first", "second"]
        variants = (
            ["other-tool", "first", "second"],
            ["tool", "changed", "second"],
            ["tool", "second", "first"],
            ["tool", "firstsecond"],
        )
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(0, False, "not_needed")
            first = run_approved_tests("workspace", [original], 5)
            repeated = run_approved_tests("workspace", [original.copy()], 5)
            changed = [
                run_approved_tests("workspace", [command], 5).command_results[0].command_identity
                for command in variants
            ]

        identity = first.command_results[0].command_identity
        self.assertEqual(identity, self.expected_command_identity(original))
        self.assertEqual(identity, repeated.command_results[0].command_identity)
        self.assertEqual(len(identity), 64)
        self.assertEqual(len(set([identity, *changed])), 1 + len(variants))

    def test_one_passing_command_preserves_argv_and_workspace(self) -> None:
        command = ["py", "-m", "unittest", "tests/test_example.py"]
        workspace = Path(__file__).resolve().parents[1]
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(0, False, "not_needed")
            result = run_approved_tests(workspace, [command], 60)

        run.assert_called_once_with(
            command, input_bytes=b"", cwd=workspace, timeout_seconds=60
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.commands_executed, 1)
        self.assertEqual(result.first_failed_index, None)
        self.assertEqual(result.command_results[0].index, 1)
        self.assertEqual(result.command_results[0].exit_status, 0)
        self.assertGreaterEqual(result.command_results[0].elapsed_seconds, 0)
        self.assertFalse(result.command_results[0].timed_out)
        self.assertEqual(result.command_results[0].cleanup_result, "not_needed")
        self.assertEqual(
            result.command_results[0].command_identity,
            self.expected_command_identity(command),
        )

    def test_multiple_commands_run_in_exact_order(self) -> None:
        commands = [["tool", "first arg"], ["another-tool", "--exact"]]
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.side_effect = [
                WindowsProcessResult(0, False, "not_needed"),
                WindowsProcessResult(0, False, "not_needed"),
            ]
            result = run_approved_tests("workspace", commands, 15)

        self.assertEqual(
            run.call_args_list,
            [
                call(commands[0], input_bytes=b"", cwd="workspace", timeout_seconds=15),
                call(commands[1], input_bytes=b"", cwd="workspace", timeout_seconds=15),
            ],
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.commands_executed, 2)
        self.assertEqual([item.index for item in result.command_results], [1, 2])

    def test_nonzero_exit_stops_remaining_commands(self) -> None:
        commands = [["failing-test"], ["must-not-run"]]
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(3, False, "not_needed")
            result = run_approved_tests("workspace", commands, 10)

        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.commands_executed, 1)
        self.assertEqual(result.first_failed_index, 1)
        self.assertEqual(result.command_results[0].exit_status, 3)
        self.assertEqual(
            result.command_results[0].command_identity,
            self.expected_command_identity(commands[0]),
        )

    def test_timeout_stops_remaining_commands(self) -> None:
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(None, True, "taskkill_completed")
            result = run_approved_tests("workspace", [["slow"], ["later"]], 1)

        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.commands_executed, 1)
        self.assertEqual(result.first_failed_index, 1)
        self.assertTrue(result.command_results[0].timed_out)
        self.assertEqual(result.command_results[0].cleanup_result, "taskkill_completed")
        self.assertEqual(
            result.command_results[0].command_identity,
            self.expected_command_identity(["slow"]),
        )

    def test_missing_executable_fails_safely_and_stops(self) -> None:
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(None, False, "launch_failed")
            result = run_approved_tests("workspace", [["missing"], ["later"]], 5)

        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.first_failed_index, 1)
        self.assertEqual(result.command_results[0].exit_status, None)
        self.assertEqual(result.command_results[0].cleanup_result, "launch_failed")
        self.assertEqual(
            result.command_results[0].command_identity,
            self.expected_command_identity(["missing"]),
        )

    def test_failed_or_uncertain_cleanup_stops_remaining_commands(self) -> None:
        for cleanup in ("taskkill_failed", "taskkill_timed_out", "parent_not_reaped", "unknown"):
            with self.subTest(cleanup=cleanup):
                with patch("auto_coding.test_runner.run_windows_process") as run:
                    run.return_value = WindowsProcessResult(0, False, cleanup)
                    result = run_approved_tests("workspace", [["first"], ["later"]], 5)
                run.assert_called_once()
                self.assertFalse(result.passed)
                self.assertEqual(result.first_failed_index, 1)
                self.assertEqual(result.command_results[0].cleanup_result, cleanup)

    def test_invalid_commands_and_timeouts_are_rejected(self) -> None:
        for commands in ([], ["py"], [[]], [[""]], [[" "]], [["py", 1]], None):
            with self.subTest(commands=commands):
                with patch("auto_coding.test_runner.run_windows_process") as run:
                    with self.assertRaises(ValueError):
                        run_approved_tests("workspace", commands, 5)
                    run.assert_not_called()
        for timeout in (0, -1, True, float("inf"), float("nan"), "5"):
            with self.subTest(timeout=timeout):
                with patch("auto_coding.test_runner.run_windows_process") as run:
                    with self.assertRaises(ValueError):
                        run_approved_tests("workspace", [["py"]], timeout)
                    run.assert_not_called()

    def test_results_do_not_include_command_prompts_or_process_output(self) -> None:
        secret_command = ["test-tool", "private prompt"]
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(1, False, "not_needed")
            result = run_approved_tests("workspace", [secret_command], 5)

        rendered = repr(result)
        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("private stdout", rendered)
        self.assertNotIn("private stderr", rendered)


class ApprovedTestDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name).resolve()
        self.artifact = self.repository / "approved-check.py"
        self.contents = b"# An approved harness and its declared inputs are immutable.\n"
        self.artifact.write_bytes(self.contents)
        self.contract = {"inputs_complete": True, "failure_protocols": ["exit-status-v1"], "protected_files": [
            {"path": self.artifact.name, "sha256": hashlib.sha256(self.contents).hexdigest(), "size_bytes": len(self.contents)}]}

    def run_result(self, process, contract=True):
        with patch("auto_coding.test_runner.run_windows_process", return_value=process) as run:
            result = run_approved_tests(self.repository, [["approved-harness"]], 10,
                                        **({"test_contract": self.contract} if contract else {}))
        run.assert_called_once_with(["approved-harness"], input_bytes=b"", cwd=self.repository, timeout_seconds=10)
        if not result.passed:
            self.assertEqual(result.command_results[0].diagnostic, result.diagnostic)
        return result

    def test_approved_protocol_distinguishes_functional_infrastructure_and_unknown(self):
        for status, classification in ((0, "PASSED"), (10, "APPLICATION_FAILURE"),
                                       (20, "INFRASTRUCTURE_FAILURE"), (1, "UNKNOWN"), (99, "UNKNOWN")):
            with self.subTest(status=status):
                result = self.run_result(WindowsProcessResult(status, False, "not_needed"))
                self.assertEqual(result.diagnostic.classification, classification)
                self.assertEqual(result.passed, status == 0)

    def test_arbitrary_existing_and_explicitly_opaque_commands_are_unknown(self):
        result = self.run_result(WindowsProcessResult(10, False, "not_needed"), contract=False)
        self.assertEqual(result.diagnostic.classification, "UNKNOWN")
        self.assertEqual(result.diagnostic.reason_code, "TEST_PROTOCOL_UNAVAILABLE")
        self.contract["failure_protocols"] = ["opaque"]
        self.assertEqual(self.run_result(WindowsProcessResult(10, False, "not_needed")).diagnostic.classification, "UNKNOWN")

    def test_timeouts_launch_failures_and_uncertain_cleanup_override_exit_protocol(self):
        cases = ((WindowsProcessResult(None, True, "taskkill_completed"), "TIMEOUT"),
                 (WindowsProcessResult(10, True, "taskkill_failed"), "TIMEOUT"),
                 (WindowsProcessResult(None, False, "launch_failed"), "LAUNCH_FAILURE"),
                 (WindowsProcessResult(10, False, "taskkill_failed"), "CLEANUP_FAILURE"),
                 (WindowsProcessResult(None, False, "parent_not_reaped"), "CLEANUP_FAILURE"),
                 (WindowsProcessResult(None, False, "not_needed"), "UNKNOWN"))
        for process, classification in cases:
            with self.subTest(classification=classification, process=process):
                result = self.run_result(process)
                self.assertFalse(result.passed)
                self.assertEqual(result.diagnostic.classification, classification)

    def test_modified_protected_input_prevents_any_test_launch(self):
        self.artifact.write_bytes(b"untrusted")
        with patch("auto_coding.test_runner.run_windows_process") as run:
            result = run_approved_tests(self.repository, [["approved-harness"]], 10, test_contract=self.contract)
        run.assert_not_called()
        self.assertEqual((result.commands_executed, result.command_results, result.first_failed_index), (0, (), None))
        self.assertEqual(result.diagnostic.reason_code, "PROTECTED_TEST_INPUT_CHANGED")

    def test_mutation_during_test_cannot_pass_even_with_zero_exit(self):
        self.contract["failure_protocols"] *= 2

        def mutate(*args, **kwargs):
            self.artifact.write_bytes(b"X" * len(self.contents))
            return WindowsProcessResult(0, False, "not_needed")

        with patch("auto_coding.test_runner.run_windows_process", side_effect=mutate) as run:
            result = run_approved_tests(self.repository, [["first"], ["must-not-run"]], 10, test_contract=self.contract)
        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.first_failed_index, 1)
        self.assertEqual(result.command_results[0].exit_status, 0)
        self.assertEqual(result.diagnostic.classification, "INFRASTRUCTURE_FAILURE")
        self.assertEqual(result.diagnostic.reason_code, "PROTECTED_TEST_INPUT_CHANGED")

    def test_invalid_protocol_contract_prevents_launch(self):
        self.contract["failure_protocols"] = ["stderr-marker"]
        with patch("auto_coding.test_runner.run_windows_process") as run:
            with self.assertRaises(ValueError):
                run_approved_tests(self.repository, [["approved-harness"]], 10, test_contract=self.contract)
        run.assert_not_called()

    def test_input_integrity_loss_before_next_command_preserves_actual_execution_count(self):
        self.contract["failure_protocols"] *= 2
        with patch("auto_coding.test_runner.check_test_protection", side_effect=[
                TestProtectionResult(True), TestProtectionResult(True),
                TestProtectionResult(False, "PROTECTED_TEST_INPUT_UNAVAILABLE")]), \
             patch("auto_coding.test_runner.run_windows_process", return_value=WindowsProcessResult(0, False, "not_needed")) as run:
            result = run_approved_tests(self.repository, [["first"], ["must-not-run"]], 10, test_contract=self.contract)
        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.commands_executed, 1)
        self.assertIsNone(result.first_failed_index)  # No second command was executed.
        self.assertEqual(result.command_results[0].diagnostic.classification, "PASSED")
        self.assertEqual(result.diagnostic.classification, "INFRASTRUCTURE_FAILURE")

    def test_real_inline_assertion_protocol_needs_no_file_and_preserves_argv(self):
        code = "import sys\ntry:\n assert 1 == 2\nexcept AssertionError:\n sys.exit(10)\nexcept Exception:\n sys.exit(20)"
        command = [sys.executable, "-c", code]
        self.contract["protected_files"] = []
        result = run_approved_tests(self.repository, [command], 10, test_contract=self.contract)
        self.assertFalse(result.passed)
        self.assertEqual(result.diagnostic.classification, "APPLICATION_FAILURE")
        self.assertEqual(result.command_results[0].command_identity, ApprovedTestRunnerTests.expected_command_identity(command))

    def test_real_output_spoof_is_discarded_and_never_classifies_an_opaque_failure(self):
        code = ("import sys\n"
                "sys.stdout.write('APPLICATION_FAILURE AssertionError PRIVATE_SECRET\\n' * 4096)\n"
                "sys.stderr.write('APPROVED_FUNCTIONAL_FAILURE PRIVATE_SECRET\\n' * 4096)\n"
                "sys.exit(1)")
        self.contract.update(protected_files=[], failure_protocols=["opaque"])
        before = set(self.repository.iterdir())
        result = run_approved_tests(self.repository, [[sys.executable, "-c", code]], 10, test_contract=self.contract)
        self.assertEqual(result.diagnostic.classification, "UNKNOWN")
        self.assertNotIn("PRIVATE_SECRET", repr(result))
        self.assertLess(len(repr(result)), 1500)
        self.assertEqual(set(self.repository.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
