"""Tests for the operational event log."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stderr


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.operations import OperationsLog, TimeoutDiagnostic, show_timeout


class TimeoutDiagnosticTests(unittest.TestCase):
    def test_timeout_record_and_sanitized_terminal_notice(self):
        diagnostic = TimeoutDiagnostic("implementation", 1800, 1800.5)
        output = io.StringIO()
        with redirect_stderr(output):
            show_timeout(diagnostic, "batch-1", "task-2", ("task-1",))
        self.assertEqual(diagnostic.record()["classification"], "TIMEOUT")
        for text in ("TIMEOUT", "batch-1", "task-2", "1800 seconds", "1800.5", "Published task: task-1", "NOT STARTED"):
            self.assertIn(text, output.getvalue())
        output = io.StringIO()
        with redirect_stderr(output):
            show_timeout(diagnostic, "PRIVATE\nESCAPE", "bad\x1b", ("bad\n",))
        self.assertNotIn("PRIVATE", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())
        output = io.StringIO()
        with redirect_stderr(output):
            show_timeout(TimeoutDiagnostic("git_preflight", 0.123456789))
        self.assertIn("Configured limit: 0.123456789 seconds", output.getvalue())
        output.close()
        with redirect_stderr(output):
            show_timeout(diagnostic)  # Closed terminal does not prevent stopping.

    def test_invalid_timeout_records_are_rejected(self):
        for changes in ({"operation": "PRIVATE"}, {"classification": "SUCCESS"},
                        {"configured_seconds": True}, {"configured_seconds": 0},
                        {"configured_seconds": float("inf")}, {"configured_seconds": 10**1000},
                        {"elapsed_seconds": -1}, {"elapsed_seconds": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TimeoutDiagnostic(**({"operation": "review", "configured_seconds": 600} | changes))


class OperationsLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "runtime" / "logs" / "operations.jsonl"

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def test_valid_event_has_utc_timestamp_and_structured_fields(self) -> None:
        log = OperationsLog(self.path)
        log.append(
            run_id="run-1",
            stage="planning",
            status="started",
            batch_id="batch-1",
            task_id="task-1",
            reason_code="READY",
        )

        event, = self.events()
        timestamp = datetime.fromisoformat(event["timestamp"])
        self.assertEqual(timestamp.utcoffset(), timezone.utc.utcoffset(timestamp))
        self.assertEqual(event["run_id"], "run-1")
        self.assertEqual(event["stage"], "planning")
        self.assertEqual(event["status"], "started")
        self.assertEqual(event["batch_id"], "batch-1")
        self.assertEqual(event["task_id"], "task-1")
        self.assertEqual(event["reason_code"], "READY")
        self.assertEqual(len(event), 7)

    def test_multiple_events_append(self) -> None:
        log = OperationsLog(self.path)
        log.append(run_id="run-1", stage="planning", status="started")
        log.append(run_id="run-1", stage="planning", status="succeeded")

        self.assertEqual([event["status"] for event in self.events()], ["started", "succeeded"])

    def test_reinitialization_preserves_history(self) -> None:
        OperationsLog(self.path).append(run_id="run-1", stage="planning", status="started")
        OperationsLog(self.path).append(run_id="run-2", stage="planning", status="stopped")

        self.assertEqual([event["run_id"] for event in self.events()], ["run-1", "run-2"])

    def test_invalid_status_is_rejected(self) -> None:
        log = OperationsLog(self.path)
        with self.assertRaises(ValueError):
            log.append(run_id="run-1", stage="planning", status="unknown")
        self.assertEqual(self.events(), [])

    def test_initialization_failure_raises(self) -> None:
        blocking_file = Path(self.temp_dir.name) / "blocking-file"
        blocking_file.write_text("occupied", encoding="utf-8")
        with self.assertRaises(OSError):
            OperationsLog(blocking_file / "operations.jsonl")

    def test_arbitrary_message_payload_is_rejected(self) -> None:
        log = OperationsLog(self.path)
        with self.assertRaises(TypeError):
            log.append(run_id="run-1", stage="planning", status="failed", message="raw error")
        with self.assertRaises(ValueError):
            log.append(run_id="run-1", stage="planning", status="failed", reason_code="raw error")
        self.assertEqual(self.events(), [])


if __name__ == "__main__":
    unittest.main()
