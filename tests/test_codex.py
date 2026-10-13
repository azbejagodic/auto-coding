"""Mock-only tests for the Codex implementation adapter."""

from pathlib import Path
import json
import sys
import uuid
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.codex import run_implementation
from auto_coding.integrations.windows_process import WindowsProcessResult


class CodexRunnerTests(unittest.TestCase):
    SESSION_ID = "a26947d0-1111-4444-8888-849398fad344"

    def jsonl(self, session_id=None):
        return (json.dumps({"type": "thread.started", "thread_id": session_id or self.SESSION_ID})
                + "\n" + json.dumps({"type": "turn.completed"}) + "\n").encode("utf-8")

    def test_command_uses_exact_arguments_prompt_workspace_and_timeout(self) -> None:
        prompt = "Implement this literal text: $HOME && `echo hi`\nž"
        workspace = Path(__file__).resolve().parents[1]
        output_files = []

        def execute(command, **kwargs):
            output_files.append(kwargs["stdout_file"])
            kwargs["stdout_file"].write(self.jsonl())
            return WindowsProcessResult(0, False, "not_needed")

        with patch("auto_coding.integrations.codex.run_windows_process") as run:
            run.side_effect = execute
            outcome = run_implementation(prompt, "gpt-6-sol", "medium", workspace, 42)

        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], [
            "codex", "--ask-for-approval", "never", "exec", "--model", "gpt-6-sol",
            "-c", "model_reasoning_effort=medium",
            "--sandbox", "workspace-write", "--json", "-",
        ])
        self.assertEqual(kwargs["input_bytes"], prompt.encode("utf-8"))
        self.assertEqual(kwargs["cwd"], workspace)
        self.assertEqual(kwargs["timeout_seconds"], 42)
        self.assertTrue(output_files[0].closed)
        self.assertFalse(Path(output_files[0].name).exists())
        self.assertFalse(Path(workspace) in Path(output_files[0].name).resolve().parents)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.session_id, self.SESSION_ID)
        self.assertIsNone(outcome.error_code)
        self.assertEqual(outcome.exit_status, 0)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.cleanup_result, "not_needed")
        self.assertGreaterEqual(outcome.elapsed_seconds, 0)

    def test_missing_malformed_conflicting_and_oversized_evidence(self) -> None:
        bad_outputs = [
            (b'{"type":"turn.completed"}\n', "SESSION_ID_MISSING"),
            (b'{not json}\n', "SESSION_EVIDENCE_MALFORMED"),
            (self.jsonl("not-a-uuid"), "SESSION_ID_INVALID"),
            (self.jsonl() + self.jsonl(str(uuid.uuid4())), "SESSION_ID_AMBIGUOUS"),
            (b"x" * (8 * 1024 * 1024 + 1), "OUTPUT_TOO_LARGE"),
        ]
        for output, error in bad_outputs:
            with self.subTest(error=error):
                def execute(command, **kwargs):
                    kwargs["stdout_file"].write(output)
                    return WindowsProcessResult(0, False, "not_needed")

                with patch("auto_coding.integrations.codex.run_windows_process",
                           side_effect=execute) as run:
                    outcome = run_implementation("private prompt", "gpt-6-sol", "low", ".", 5)
                run.assert_called_once()
                self.assertFalse(outcome.succeeded)
                self.assertIsNone(outcome.session_id)
                self.assertEqual(outcome.error_code, error)
                self.assertNotIn("private prompt", repr(outcome))
                self.assertNotIn("thread.started", repr(outcome))

    def test_resume_uses_exact_session_model_effort_workspace_and_prompt(self) -> None:
        prompt = "Continue literally: $HOME && private model output"
        workspace = Path(__file__).resolve().parents[1]

        def execute(command, **kwargs):
            kwargs["stdout_file"].write(self.jsonl())
            return WindowsProcessResult(0, False, "not_needed")

        with patch("auto_coding.integrations.codex.run_windows_process",
                   side_effect=execute) as run:
            outcome = run_implementation(
                prompt, "gpt-6-sol", "high", workspace, 21, session_id=self.SESSION_ID,
            )

        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], [
            "codex", "--ask-for-approval", "never", "exec", "resume",
            "--model", "gpt-6-sol", "-c", "model_reasoning_effort=high",
            "-c", 'sandbox_mode="workspace-write"', "--json", self.SESSION_ID, "-",
        ])
        self.assertEqual(kwargs["input_bytes"], prompt.encode("utf-8"))
        self.assertEqual(kwargs["cwd"], workspace)
        self.assertEqual(kwargs["timeout_seconds"], 21)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.session_id, self.SESSION_ID)
        self.assertIsNone(outcome.error_code)
        self.assertNotIn(prompt, repr(outcome))
        self.assertNotIn("private model output", repr(outcome))

    def test_invalid_resume_id_is_rejected_before_process_launch(self) -> None:
        invalid = ("", "not-a-uuid", "00000000-0000-0000-0000-000000000000",
                   "a26947d0111144448888849398fad344", 42)
        with patch("auto_coding.integrations.codex.run_windows_process") as run:
            for value in invalid:
                with self.subTest(value=value):
                    outcome = run_implementation("private", "gpt-6-sol", "medium", ".", 5,
                                                 session_id=value)
                    self.assertFalse(outcome.succeeded)
                    self.assertIsNone(outcome.session_id)
                    self.assertEqual(outcome.error_code, "SESSION_ID_INVALID")
            run.assert_not_called()

    def test_resume_missing_or_different_returned_id_fails_closed(self) -> None:
        other_id = "ee39e95b-2222-4444-8888-2cb53be63093"
        for output, error in (
            (b'{"type":"turn.completed"}\n', "SESSION_ID_MISSING"),
            (self.jsonl(other_id), "SESSION_ID_MISMATCH"),
        ):
            with self.subTest(error=error):
                def execute(command, **kwargs):
                    kwargs["stdout_file"].write(output)
                    return WindowsProcessResult(0, False, "not_needed")

                with patch("auto_coding.integrations.codex.run_windows_process",
                           side_effect=execute) as run:
                    outcome = run_implementation("private", "gpt-6-sol", "medium", ".", 5,
                                                 session_id=self.SESSION_ID)
                run.assert_called_once()
                self.assertFalse(outcome.succeeded)
                self.assertIsNone(outcome.session_id)
                self.assertEqual(outcome.error_code, error)

    def test_resume_process_failures_do_not_retry_or_fallback(self) -> None:
        cases = [
            (WindowsProcessResult(7, False, "not_needed"), "PROCESS_FAILED"),
            (WindowsProcessResult(None, True, "taskkill_completed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(None, True, "taskkill_failed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(0, False, "parent_not_reaped"),
             "PROCESS_CLEANUP_UNCERTAIN"),
        ]
        for process, error in cases:
            with self.subTest(error=error, process=process):
                with patch("auto_coding.integrations.codex.run_windows_process",
                           return_value=process) as run:
                    outcome = run_implementation("private", "gpt-6-sol", "medium", ".", 5,
                                                 session_id=self.SESSION_ID)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][4], "resume")
                self.assertFalse(outcome.succeeded)
                self.assertEqual(outcome.error_code, error)
                self.assertIsNone(outcome.session_id)

    def test_nonzero_exit_is_failure_without_retry_or_output(self) -> None:
        prompt = "private prompt"
        with patch("auto_coding.integrations.codex.run_windows_process") as run:
            run.return_value = WindowsProcessResult(7, False, "not_needed")
            outcome = run_implementation(prompt, "gpt-6-sol", "low", ".", 10)

        run.assert_called_once()
        self.assertEqual(outcome.exit_status, 7)
        self.assertFalse(outcome.succeeded)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.error_code, "PROCESS_FAILED")
        self.assertNotIn(prompt, repr(outcome))

    def test_timeout_reports_cleanup_without_retry_or_output(self) -> None:
        prompt = "private timeout prompt"
        with patch("auto_coding.integrations.codex.run_windows_process") as run:
            run.return_value = WindowsProcessResult(None, True, "taskkill_completed")
            outcome = run_implementation(prompt, "gpt-6-sol", "high", ".", 5)

        run.assert_called_once()
        self.assertIsNone(outcome.exit_status)
        self.assertTrue(outcome.timed_out)
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.cleanup_result, "taskkill_completed")
        self.assertEqual(outcome.error_code, "PROCESS_TIMEOUT")
        self.assertNotIn(prompt, repr(outcome))

    def test_missing_cli_is_failure_without_retry_or_exception_text(self) -> None:
        prompt = "private launch prompt"
        with patch("auto_coding.integrations.codex.run_windows_process") as run:
            run.return_value = WindowsProcessResult(None, False, "launch_failed")
            outcome = run_implementation(prompt, "gpt-6-sol", "xhigh", ".", 5)

        run.assert_called_once()
        self.assertIsNone(outcome.exit_status)
        self.assertFalse(outcome.timed_out)
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.cleanup_result, "launch_failed")
        self.assertEqual(outcome.error_code, "PROCESS_LAUNCH_FAILED")
        self.assertNotIn(prompt, repr(outcome))
        self.assertNotIn("private output", repr(outcome))


if __name__ == "__main__":
    unittest.main()
