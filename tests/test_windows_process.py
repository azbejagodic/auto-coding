"""Tests for bounded Windows subprocess cleanup."""

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import run_windows_process


class WindowsProcessTests(unittest.TestCase):
    def test_optional_binary_stdout_redirect_preserves_default(self) -> None:
        with tempfile.TemporaryFile(mode="w+b") as output_file:
            with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen:
                process = popen.return_value
                process.returncode = 0
                result = run_windows_process(
                    ["tool"], input_bytes=b"literal", cwd="repo", timeout_seconds=8,
                    stdout_file=output_file,
                )
            self.assertIs(popen.call_args.kwargs["stdout"], output_file)
            self.assertEqual(result.exit_status, 0)

    def test_normal_completion_uses_popen_without_shell(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen:
            process = popen.return_value
            process.returncode = 0
            result = run_windows_process(
                ["tool", "arg"], input_bytes=b"literal", cwd="repo", timeout_seconds=8
            )

        popen.assert_called_once_with(
            ["tool", "arg"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, cwd="repo", shell=False,
        )
        process.communicate.assert_called_once_with(input=b"literal", timeout=8)
        self.assertEqual(result.exit_status, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.cleanup_result, "not_needed")

    def test_timeout_runs_taskkill_and_reaps_parent(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen, patch(
            "auto_coding.integrations.windows_process.subprocess.run"
        ) as taskkill:
            process = popen.return_value
            process.pid = 1234
            process.communicate.side_effect = subprocess.TimeoutExpired("tool", 2, output=b"secret output")
            taskkill.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            result = run_windows_process(
                ["tool"], input_bytes=b"secret prompt", cwd="repo", timeout_seconds=2
            )

        popen.assert_called_once()
        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            shell=False, check=False, timeout=5.0,
        )
        process.wait.assert_called_once_with(timeout=5.0)
        process.kill.assert_not_called()
        self.assertTrue(result.timed_out)
        self.assertEqual(result.cleanup_result, "taskkill_completed")
        self.assertNotIn("secret prompt", repr(result))
        self.assertNotIn("secret output", repr(result))

    def test_taskkill_failure_is_reported_and_parent_is_reaped(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen, patch(
            "auto_coding.integrations.windows_process.subprocess.run"
        ) as taskkill:
            process = popen.return_value
            process.pid = 1234
            process.communicate.side_effect = subprocess.TimeoutExpired("tool", 2)
            taskkill.return_value = subprocess.CompletedProcess(args=[], returncode=1)
            result = run_windows_process(["tool"], input_bytes=b"", cwd="repo", timeout_seconds=2)

        popen.assert_called_once()
        taskkill.assert_called_once()
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=5.0)
        self.assertEqual(result.cleanup_result, "taskkill_failed")
        self.assertTrue(result.timed_out)

    def test_taskkill_timeout_is_reported(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen, patch(
            "auto_coding.integrations.windows_process.subprocess.run"
        ) as taskkill:
            process = popen.return_value
            process.pid = 1234
            process.communicate.side_effect = subprocess.TimeoutExpired("tool", 2)
            taskkill.side_effect = subprocess.TimeoutExpired("taskkill", 5)
            result = run_windows_process(["tool"], input_bytes=b"", cwd="repo", timeout_seconds=2)

        popen.assert_called_once()
        taskkill.assert_called_once()
        process.kill.assert_called_once()
        process.wait.assert_called_once_with(timeout=5.0)
        self.assertEqual(result.cleanup_result, "taskkill_timed_out")
        self.assertTrue(result.timed_out)

    def test_parent_reap_timeout_is_reported(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen, patch(
            "auto_coding.integrations.windows_process.subprocess.run"
        ) as taskkill:
            process = popen.return_value
            process.pid = 1234
            process.communicate.side_effect = subprocess.TimeoutExpired("tool", 2)
            process.wait.side_effect = subprocess.TimeoutExpired("tool", 5)
            taskkill.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            result = run_windows_process(["tool"], input_bytes=b"", cwd="repo", timeout_seconds=2)

        self.assertEqual(result.cleanup_result, "parent_not_reaped")
        self.assertTrue(result.timed_out)

    def test_process_launch_failure_is_reported(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen:
            popen.side_effect = FileNotFoundError("private path")
            result = run_windows_process(
                ["missing"], input_bytes=b"secret", cwd="repo", timeout_seconds=2
            )

        popen.assert_called_once()
        self.assertIsNone(result.exit_status)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.cleanup_result, "launch_failed")
        self.assertNotIn("private path", repr(result))

    @unittest.skipUnless(sys.platform == "win32", "Windows process-tree integration test")
    def test_real_timeout_terminates_python_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready.txt"
            survived = Path(directory) / "survived.txt"
            child_code = (
                "from pathlib import Path; import sys, time; "
                "Path(sys.argv[1]).write_text('ready'); time.sleep(3); "
                "Path(sys.argv[2]).write_text('survived')"
            )
            parent_code = (
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]]); "
                "time.sleep(30)"
            )
            result = run_windows_process(
                [sys.executable, "-c", parent_code, child_code, str(ready), str(survived)],
                input_bytes=b"", cwd=directory, timeout_seconds=1.5,
            )
            self.assertTrue(ready.exists())
            self.assertTrue(result.timed_out)
            self.assertEqual(result.cleanup_result, "taskkill_completed")
            time.sleep(3.2)
            self.assertFalse(survived.exists())


if __name__ == "__main__":
    unittest.main()
