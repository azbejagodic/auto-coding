"""Tests for bounded Windows subprocess cleanup."""

from pathlib import Path
import errno
import io
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import (
    WindowsProcessResult, _ObservedStdin, run_windows_process,
)


class WindowsProcessTests(unittest.TestCase):
    def run_with_stdin(self, stream, *, input_bytes=b"literal", exit_status=0):
        # Exercise the real suppression behavior used by communicate's Windows
        # writer, while keeping process creation and completion deterministic.
        stdin_write = subprocess.Popen._stdin_write
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen:
            process = popen.return_value
            process.stdin = stream
            process.returncode = exit_status
            process.communicate.side_effect = lambda input, timeout: stdin_write(process, input)
            result = run_windows_process(
                ["tool"], input_bytes=input_bytes, cwd="repo", timeout_seconds=8,
            )
        process.communicate.assert_called_once_with(input=input_bytes, timeout=8)
        return result

    def test_complete_stdin_write_is_independent_of_exit_status(self) -> None:
        for exit_status in (0, 7):
            with self.subTest(exit_status=exit_status):
                stream = Mock()
                stream.write.return_value = len(b"literal")
                result = self.run_with_stdin(stream, exit_status=exit_status)
                stream.write.assert_called_once_with(b"literal")
                stream.close.assert_called_once_with()
                self.assertEqual(result.stdin_write_status, "complete")
                self.assertEqual(result.exit_status, exit_status)
                self.assertFalse(result.timed_out)
                self.assertEqual(result.cleanup_result, "not_needed")

    def test_early_stdin_closure_remains_incomplete_and_preserves_value_error(self) -> None:
        stream = io.BytesIO()
        stream.close()
        observer = _ObservedStdin(stream, len(b"literal"))
        with self.assertRaises(ValueError):
            observer.write(b"literal")
        self.assertEqual(observer.status, "incomplete")
        with self.assertRaises(ValueError):
            self.run_with_stdin(stream, exit_status=0)

    def test_partial_stdin_write_is_incomplete_even_with_zero_exit(self) -> None:
        for written in (0, 2, None):
            with self.subTest(written=written):
                stream = Mock()
                stream.write.return_value = written
                result = self.run_with_stdin(stream)
                stream.write.assert_called_once_with(b"literal")
                stream.close.assert_called_once_with()
                self.assertEqual(result.stdin_write_status, "incomplete")
                self.assertEqual(result.exit_status, 0)

    def test_broken_pipe_and_windows_einval_are_recorded_before_suppression(self) -> None:
        for error in (BrokenPipeError("private payload"),
                      OSError(errno.EINVAL, "private payload")):
            with self.subTest(error_type=type(error).__name__, errno=error.errno):
                stream = Mock()
                stream.write.side_effect = error
                result = self.run_with_stdin(stream, input_bytes=b"secret prompt")
                stream.close.assert_called_once_with()
                self.assertEqual(result.stdin_write_status, "incomplete")
                self.assertEqual(result.exit_status, 0)
                self.assertNotIn("private payload", repr(result))
                self.assertNotIn("secret prompt", repr(result))

    def test_stdin_close_failure_prevents_complete_status(self) -> None:
        for error in (BrokenPipeError("private payload"),
                      OSError(errno.EINVAL, "private payload")):
            with self.subTest(error_type=type(error).__name__, errno=error.errno):
                stream = Mock()
                stream.write.return_value = len(b"literal")
                stream.close.side_effect = error
                result = self.run_with_stdin(stream)
                self.assertEqual(result.stdin_write_status, "incomplete")
                self.assertEqual(result.exit_status, 0)
                self.assertNotIn("private payload", repr(result))

    def test_observer_write_close_and_flush_preserve_original_exceptions(self) -> None:
        for operation in ("write", "close", "flush"):
            for error in (
                BrokenPipeError("private payload"), OSError(errno.EINVAL, "private payload"),
                OSError(errno.EIO, "private payload"), OSError(errno.EACCES, "private payload"),
                ValueError("private payload"),
            ):
                with self.subTest(operation=operation, error_type=type(error).__name__,
                                  errno=getattr(error, "errno", None)):
                    stream = Mock()
                    stream.write.return_value = len(b"literal")
                    observer = _ObservedStdin(stream, len(b"literal"))
                    if operation != "write":
                        observer.write(b"literal")
                    getattr(stream, operation).side_effect = error
                    arguments = (b"literal",) if operation == "write" else ()
                    with self.assertRaises(type(error)) as raised:
                        getattr(observer, operation)(*arguments)
                    self.assertIs(raised.exception, error)
                    self.assertEqual(observer.status, "incomplete")
                    # A later successful close must not erase an earlier failure.
                    stream.close.side_effect = None
                    observer.close()
                    self.assertEqual(observer.status, "incomplete")
                    result = WindowsProcessResult(0, False, "not_needed", observer.status)
                    self.assertNotIn("private payload", repr(result))
                    self.assertNotIn("literal", repr(result))

    def test_subprocess_does_not_suppress_unexpected_stdin_errors(self) -> None:
        for operation in ("write", "close"):
            for error in (OSError(errno.EIO, "private payload"),
                          OSError(errno.EACCES, "private payload"),
                          ValueError("private payload")):
                with self.subTest(operation=operation, error_type=type(error).__name__,
                                  errno=getattr(error, "errno", None)):
                    stream = Mock()
                    stream.write.return_value = len(b"literal")
                    getattr(stream, operation).side_effect = error
                    with self.assertRaises(type(error)) as raised:
                        self.run_with_stdin(stream, exit_status=0)
                    self.assertIs(raised.exception, error)

    def test_unobserved_stdin_writes_do_not_imply_completion(self) -> None:
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen:
            popen.return_value.returncode = 0
            result = run_windows_process(
                ["tool"], input_bytes=b"literal", cwd="repo", timeout_seconds=8,
            )
        self.assertEqual(result.stdin_write_status, "incomplete")
        self.assertEqual(result.exit_status, 0)

    def test_empty_input_does_not_request_stdin_writing(self) -> None:
        stream = Mock()
        result = self.run_with_stdin(stream, input_bytes=b"")
        stream.write.assert_not_called()
        stream.close.assert_called_once_with()
        self.assertEqual(result.stdin_write_status, "not_requested")

    def test_legacy_result_construction_does_not_imply_complete_stdin(self) -> None:
        self.assertEqual(
            WindowsProcessResult(0, False, "not_needed").stdin_write_status, "incomplete",
        )

    def test_completed_stdin_write_remains_complete_on_later_process_timeout(self) -> None:
        stdin_write = subprocess.Popen._stdin_write
        stream = Mock()
        stream.write.return_value = len(b"literal")
        with patch("auto_coding.integrations.windows_process.subprocess.Popen") as popen, patch(
            "auto_coding.integrations.windows_process.subprocess.run"
        ) as taskkill:
            process = popen.return_value
            process.stdin = stream
            process.pid = 1234

            def communicate(input, timeout):
                stdin_write(process, input)
                raise subprocess.TimeoutExpired("tool", timeout)

            process.communicate.side_effect = communicate
            taskkill.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            result = run_windows_process(
                ["tool"], input_bytes=b"literal", cwd="repo", timeout_seconds=2,
            )
        self.assertEqual(result.stdin_write_status, "complete")
        self.assertTrue(result.timed_out)
        self.assertEqual(result.cleanup_result, "taskkill_completed")
        process.wait.assert_called_once_with(timeout=5.0)
        stream.close.assert_called_once_with()

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
        self.assertEqual(result.stdin_write_status, "incomplete")
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
        self.assertEqual(result.stdin_write_status, "not_requested")

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
        self.assertEqual(result.stdin_write_status, "not_requested")

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
        self.assertEqual(result.stdin_write_status, "not_requested")

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
        self.assertEqual(result.stdin_write_status, "incomplete")
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
