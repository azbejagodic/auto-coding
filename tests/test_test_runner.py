"""Tests for execution of approved terminal test commands."""

from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import WindowsProcessResult
from auto_coding.test_runner import run_approved_tests


class ApprovedTestRunnerTests(unittest.TestCase):
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

    def test_missing_executable_fails_safely_and_stops(self) -> None:
        with patch("auto_coding.test_runner.run_windows_process") as run:
            run.return_value = WindowsProcessResult(None, False, "launch_failed")
            result = run_approved_tests("workspace", [["missing"], ["later"]], 5)

        run.assert_called_once()
        self.assertFalse(result.passed)
        self.assertEqual(result.first_failed_index, 1)
        self.assertEqual(result.command_results[0].exit_status, None)
        self.assertEqual(result.command_results[0].cleanup_result, "launch_failed")

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


if __name__ == "__main__":
    unittest.main()
