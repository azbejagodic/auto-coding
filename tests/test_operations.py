"""Tests for the operational event log."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.operations import OperationsLog


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
