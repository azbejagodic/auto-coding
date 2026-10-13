"""Tests for the independent, read-only Codex reviewer."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import WindowsProcessResult
from auto_coding.review import run_review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.repository = tempfile.TemporaryDirectory()
        self.addCleanup(self.repository.cleanup)
        self.prompt = "Review this exact change; private prompt marker"
        self.criteria = ["First exact criterion", "Second exact criterion"]
        self.calls = []
        self.artifact_paths = []

    def verdict(self, **changes):
        value = {
            "decision": "APPROVED",
            "blocking_issues": [],
            "nonblocking_notes": [],
            "checked_requirements": self.criteria.copy(),
            "confidence": "HIGH",
        }
        value.update(changes)
        return value

    def run_with_verdict(self, verdict=None, *, raw=None, process=None):
        if process is None:
            process = WindowsProcessResult(0, False, "not_needed")

        def execute(command, **kwargs):
            self.calls.append((command, kwargs))
            schema_path = Path(command[command.index("--output-schema") + 1])
            verdict_path = Path(command[command.index("--output-last-message") + 1])
            self.artifact_paths.extend((schema_path, verdict_path))
            if process.exit_status == 0 and raw is not None:
                verdict_path.write_bytes(raw)
            elif process.exit_status == 0 and verdict is not None:
                verdict_path.write_text(json.dumps(verdict), encoding="utf-8")
            return process

        with patch("auto_coding.review.run_windows_process", side_effect=execute) as runner:
            result = run_review(
                self.prompt, "gpt-6-sol", "medium", self.repository.name, 12,
                self.criteria,
            )
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(all(not path.exists() for path in self.artifact_paths))
        return result

    def test_approved_verdict_and_exact_fresh_command(self):
        result = self.run_with_verdict(self.verdict())
        self.assertTrue(result.approved)
        self.assertEqual(result.verdict.checked_requirements, tuple(self.criteria))
        command, options = self.calls[0]
        schema_path, verdict_path = self.artifact_paths
        self.assertEqual(command, [
            "codex", "--ask-for-approval", "never", "exec",
            "--model", "gpt-6-sol", "-c", "model_reasoning_effort=medium",
            "--sandbox", "read-only", "--output-schema", str(schema_path),
            "--output-last-message", str(verdict_path), "-",
        ])
        self.assertEqual(options, {
            "input_bytes": self.prompt.encode("utf-8"),
            "cwd": self.repository.name,
            "timeout_seconds": 12,
        })
        self.assertNotIn("resume", command)
        self.assertFalse(Path(self.repository.name) in schema_path.parents)
        self.assertFalse(Path(self.repository.name) in verdict_path.parents)

    def test_schema_requires_exact_verdict_fields(self):
        def inspect(command, **kwargs):
            schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), {
                "decision", "blocking_issues", "nonblocking_notes",
                "checked_requirements", "confidence",
            })
            Path(command[command.index("--output-last-message") + 1]).write_text(
                json.dumps(self.verdict()), encoding="utf-8",
            )
            return WindowsProcessResult(0, False, "not_needed")

        with patch("auto_coding.review.run_windows_process", side_effect=inspect):
            result = run_review(self.prompt, "gpt-6-sol", "medium", self.repository.name,
                                12, self.criteria)
        self.assertTrue(result.approved)

    def test_rejected_and_uncertain_are_not_approved(self):
        for decision in ("REJECTED", "UNCERTAIN"):
            with self.subTest(decision=decision):
                result = self.run_with_verdict(self.verdict(decision=decision))
                self.assertFalse(result.approved)
                self.assertEqual(result.verdict.decision, decision)

    def test_blocking_issues_and_missing_criteria_stop_approval(self):
        blocked = self.run_with_verdict(self.verdict(blocking_issues=["issue"]))
        self.assertFalse(blocked.approved)
        self.assertEqual(blocked.error_code, "BLOCKING_ISSUES")
        uncovered = self.run_with_verdict(self.verdict(
            checked_requirements=[self.criteria[0]],
        ))
        self.assertFalse(uncovered.approved)
        self.assertEqual(uncovered.error_code, "CRITERIA_NOT_COVERED")

    def test_malformed_missing_and_oversized_verdicts(self):
        cases = [
            (b"{raw private response", "VERDICT_INVALID_JSON"),
            (b'\xff', "VERDICT_INVALID_JSON"),
            (b"x" * (64 * 1024 + 1), "VERDICT_TOO_LARGE"),
        ]
        for raw, code in cases:
            with self.subTest(code=code):
                result = self.run_with_verdict(raw=raw)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertNotIn(self.prompt, repr(result))
                self.assertNotIn("raw private response", repr(result))
        missing = self.run_with_verdict()
        self.assertEqual(missing.error_code, "VERDICT_MISSING")

    def test_invalid_verdict_fields_fail_closed(self):
        invalid = [
            {key: value for key, value in self.verdict().items() if key != "confidence"},
            self.verdict(extra="unexpected"),
            self.verdict(decision="MAYBE"),
            self.verdict(confidence="CERTAIN"),
            self.verdict(blocking_issues="none"),
            self.verdict(checked_requirements=[42]),
        ]
        for verdict in invalid:
            with self.subTest(verdict=verdict):
                result = self.run_with_verdict(verdict)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, "VERDICT_INVALID")
        duplicate = b'{"decision":"APPROVED","decision":"APPROVED"}'
        self.assertEqual(self.run_with_verdict(raw=duplicate).error_code,
                         "VERDICT_INVALID_JSON")

    def test_process_failures_stop_before_verdict(self):
        cases = [
            (WindowsProcessResult(1, False, "not_needed"), "PROCESS_FAILED"),
            (WindowsProcessResult(None, False, "launch_failed"), "PROCESS_LAUNCH_FAILED"),
            (WindowsProcessResult(None, True, "taskkill_completed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(None, True, "taskkill_failed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(0, False, "parent_not_reaped"), "PROCESS_CLEANUP_UNCERTAIN"),
        ]
        for process, code in cases:
            with self.subTest(process=process):
                result = self.run_with_verdict(self.verdict(), process=process)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertEqual(result.cleanup_result, process.cleanup_result)

    def test_invalid_inputs_do_not_launch(self):
        with patch("auto_coding.review.run_windows_process") as runner:
            for timeout in (0, -1, float("nan"), True):
                with self.subTest(timeout=timeout):
                    with self.assertRaises(ValueError):
                        run_review(self.prompt, "gpt-6-sol", "medium",
                                   self.repository.name, timeout, self.criteria)
            for criteria in ([], [""], [1], "one"):
                with self.subTest(criteria=criteria):
                    with self.assertRaises(ValueError):
                        run_review(self.prompt, "gpt-6-sol", "medium",
                                   self.repository.name, 12, criteria)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
