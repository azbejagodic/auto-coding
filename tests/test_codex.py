"""Mock-only tests for the Codex implementation adapter."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.codex import run_implementation


class CodexRunnerTests(unittest.TestCase):
    def test_command_uses_exact_arguments_prompt_workspace_and_timeout(self) -> None:
        prompt = "Implement this literal text: $HOME && `echo hi`\nž"
        workspace = Path(__file__).resolve().parents[1]
        with patch("auto_coding.integrations.codex.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            outcome = run_implementation(prompt, "gpt-6-sol", "medium", workspace, 42)

        run.assert_called_once_with(
            [
                "codex", "--ask-for-approval", "never", "exec", "--model", "gpt-6-sol",
                "-c", "model_reasoning_effort=medium",
                "--sandbox", "workspace-write", "-",
            ],
            input=prompt.encode("utf-8"),
            cwd=workspace,
            timeout=42,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.exit_status, 0)
        self.assertFalse(outcome.timed_out)
        self.assertGreaterEqual(outcome.elapsed_seconds, 0)

    def test_nonzero_exit_is_failure_without_retry_or_output(self) -> None:
        prompt = "private prompt"
        with patch("auto_coding.integrations.codex.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess(args=[], returncode=7)
            outcome = run_implementation(prompt, "gpt-6-sol", "low", ".", 10)

        run.assert_called_once()
        self.assertEqual(outcome.exit_status, 7)
        self.assertFalse(outcome.succeeded)
        self.assertFalse(outcome.timed_out)
        self.assertNotIn(prompt, repr(outcome))

    def test_timeout_is_failure_without_retry_or_output(self) -> None:
        prompt = "private timeout prompt"
        with patch("auto_coding.integrations.codex.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired("codex", 5, output=b"private output")
            outcome = run_implementation(prompt, "gpt-6-sol", "high", ".", 5)

        run.assert_called_once()
        self.assertIsNone(outcome.exit_status)
        self.assertTrue(outcome.timed_out)
        self.assertFalse(outcome.succeeded)
        self.assertNotIn(prompt, repr(outcome))
        self.assertNotIn("private output", repr(outcome))

    def test_missing_cli_is_failure_without_retry_or_exception_text(self) -> None:
        prompt = "private launch prompt"
        with patch("auto_coding.integrations.codex.subprocess.run") as run:
            run.side_effect = FileNotFoundError("private output")
            outcome = run_implementation(prompt, "gpt-6-sol", "xhigh", ".", 5)

        run.assert_called_once()
        self.assertIsNone(outcome.exit_status)
        self.assertFalse(outcome.timed_out)
        self.assertFalse(outcome.succeeded)
        self.assertNotIn(prompt, repr(outcome))
        self.assertNotIn("private output", repr(outcome))


if __name__ == "__main__":
    unittest.main()
