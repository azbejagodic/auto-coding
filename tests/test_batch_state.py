"""Strict durable checkpoint storage tests using disposable metadata files."""

from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.batch_state import (
    CheckpointError, create_checkpoint, load_checkpoint, update_checkpoint, validate_checkpoint_capacity,
)
from auto_coding.operations import TimeoutDiagnostic


_A = "11111111-1111-4111-8111-111111111111"
_B = "22222222-2222-4222-8222-222222222222"


def initial_state() -> dict:
    return {
        "schema_version": 1, "batch_id": "batch-1",
        "frozen_plan_sha256": "a" * 64, "expected_branch": "feature/example",
        "base_head": "b" * 40, "status": "pending", "current_task_id": None,
        "stage": "pending", "implementer_session_id": None,
        "reviewer_session_id": None, "stop_reason_code": None,
    }


class BatchStateTests(unittest.TestCase):
    def test_optional_timeout_is_durable_without_changing_legacy_checkpoint(self):
        self.create()
        self.assertNotIn("timeout", load_checkpoint(self.path))
        stopped = dict(self.running(), status="stopped", stop_reason_code="PROCESS_TIMEOUT",
                       timeout=TimeoutDiagnostic("implementation", 1800, 1801).record())
        update_checkpoint(self.path, stopped)
        self.assertEqual(load_checkpoint(self.path), stopped)
        changed = dict(stopped)
        del changed["timeout"]
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, changed)
        self.assertEqual(load_checkpoint(self.path), stopped)

    def test_invalid_or_nonstopped_timeout_checkpoint_is_rejected(self):
        detail = TimeoutDiagnostic("implementation", 1800).record()
        for state in (dict(self.running(), timeout=detail),
                      dict(self.running(), status="stopped", stop_reason_code="PROCESS_TIMEOUT", timeout=None),
                      dict(self.running(), status="stopped", stop_reason_code="PROCESS_TIMEOUT",
                           timeout=dict(detail, private="PRIVATE")),
                      dict(self.running(), status="stopped", stop_reason_code="PROCESS_TIMEOUT",
                           timeout=dict(detail, configured_seconds=0))):
            with self.subTest(state=state), self.assertRaises(CheckpointError):
                create_checkpoint(self.path, state)
        self.assertFalse(self.path.exists())

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "checkpoint.json"

    def create(self) -> dict:
        return create_checkpoint(self.path, initial_state())

    def running(self) -> dict:
        state = initial_state()
        state.update(status="running", stage="implementation", current_task_id="task-1")
        return state

    def assert_unchanged(self, original: bytes) -> None:
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_create_and_reload_exact_metadata(self) -> None:
        state = initial_state()
        with patch("auto_coding.batch_state.os.fsync", wraps=os.fsync) as sync:
            self.assertEqual(create_checkpoint(self.path, state), state)
            sync.assert_called_once()
        loaded = load_checkpoint(self.path)
        self.assertEqual(loaded, state)
        self.assertIsNot(loaded, state)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_reload_in_a_separate_python_process(self) -> None:
        self.create()
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from auto_coding.batch_state import load_checkpoint; "
            "s=load_checkpoint(sys.argv[2]); "
            "assert s['batch_id']=='batch-1' and s['status']=='pending'"
        )
        result = subprocess.run(
            [sys.executable, "-c", program, str(Path(__file__).resolve().parents[1] / "src"),
             str(self.path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_updates_persist_across_separate_loads(self) -> None:
        self.create()
        state = self.running()
        state["implementer_session_id"] = _A
        update_checkpoint(self.path, state)
        reloaded = load_checkpoint(self.path)
        self.assertEqual(reloaded, state)
        reloaded.update(stage="review", reviewer_session_id=_B)
        update_checkpoint(self.path, reloaded)
        self.assertEqual(load_checkpoint(self.path), reloaded)
        reloaded["stage"] = "testing"
        self.assertEqual(load_checkpoint(self.path)["stage"], "review")

    def test_existing_checkpoint_cannot_be_overwritten(self) -> None:
        self.create()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            create_checkpoint(self.path, initial_state())
        self.assert_unchanged(original)

    def test_existing_invalid_or_empty_file_is_not_overwritten(self) -> None:
        for raw in (b"", b"not JSON"):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with self.assertRaises(FileExistsError):
                    create_checkpoint(self.path, initial_state())
                self.assert_unchanged(raw)

    def test_immutable_identity_cannot_change(self) -> None:
        self.create()
        original = self.path.read_bytes()
        for field, value in (
            ("batch_id", "batch-2"), ("frozen_plan_sha256", "c" * 64),
            ("expected_branch", "feature/other"), ("base_head", "d" * 40),
        ):
            with self.subTest(field=field):
                state = self.running()
                state[field] = value
                with self.assertRaises(CheckpointError):
                    update_checkpoint(self.path, state)
                self.assert_unchanged(original)

    def test_established_sessions_cannot_be_replaced_or_removed(self) -> None:
        self.create()
        state = self.running()
        state.update(implementer_session_id=_A, reviewer_session_id=_B)
        update_checkpoint(self.path, state)
        original = self.path.read_bytes()
        for field in ("implementer_session_id", "reviewer_session_id"):
            for value in (None, "33333333-3333-4333-8333-333333333333"):
                with self.subTest(field=field, value=value):
                    changed = dict(state, **{field: value})
                    with self.assertRaises(CheckpointError):
                        update_checkpoint(self.path, changed)
                    self.assert_unchanged(original)

    def test_invalid_fields_types_and_formats_fail_closed(self) -> None:
        invalid = (
            ("schema_version", True), ("schema_version", 2), ("batch_id", "raw prompt text"),
            ("frozen_plan_sha256", "a" * 63), ("frozen_plan_sha256", "A" * 64),
            ("base_head", "abc"), ("expected_branch", "bad..branch"),
            ("expected_branch", 42), ("expected_branch", "x" * 513),
            ("status", "succeeded"), ("status", []), ("stage", "unknown"),
            ("stage", {}), ("current_task_id", "private transcript"),
            ("implementer_session_id", "0" * 32), ("reviewer_session_id", 123),
            ("reviewer_session_id", "00000000-0000-0000-0000-000000000000"),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                state = initial_state()
                state[field] = value
                with self.assertRaises(CheckpointError):
                    create_checkpoint(self.path, state)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_unknown_missing_and_sensitive_metadata_are_rejected(self) -> None:
        for field in ("prompt", "transcript", "credentials", "environment", "diff"):
            with self.subTest(field=field):
                state = dict(initial_state(), **{field: "SECRET"})
                with self.assertRaises(CheckpointError) as error:
                    create_checkpoint(self.path, state)
                self.assertNotIn("SECRET", str(error.exception))
        for field in initial_state():
            state = initial_state()
            del state[field]
            with self.assertRaises(CheckpointError):
                create_checkpoint(self.path, state)

    def test_inconsistent_status_task_stage_and_reason_are_rejected(self) -> None:
        invalid = (
            {"current_task_id": "task-1"}, {"implementer_session_id": _A},
            {"status": "running"}, {"status": "completed"},
            {"status": "stopped"}, {"stop_reason_code": "TEST_FAILED"},
            {"status": "stopped", "stop_reason_code": "raw text"},
            {"status": "stopped", "stage": "finished", "stop_reason_code": "FAILED"},
            {"status": "stopped", "stage": "review", "stop_reason_code": "FAILED"},
            {"status": "stopped", "current_task_id": "task-1", "stop_reason_code": "FAILED"},
            {"status": "completed", "stage": "finished", "current_task_id": "task-1"},
            {"status": "running", "stage": "finished", "current_task_id": "task-1"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                self.path.write_text(json.dumps(dict(initial_state(), **changes)), encoding="utf-8")
                with self.assertRaises(CheckpointError):
                    load_checkpoint(self.path)

    def test_implementer_and_reviewer_cannot_share_session(self) -> None:
        self.create()
        state = self.running()
        state.update(implementer_session_id=_A, reviewer_session_id=_A)
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, state)

    def test_interrupted_running_state_is_not_inferred_as_completed(self) -> None:
        self.create()
        update_checkpoint(self.path, self.running())
        self.assertEqual(load_checkpoint(self.path), self.running())
        complete = dict(self.running(), status="completed", stage="finished", current_task_id=None)
        original = self.path.read_bytes()
        for flag in (False, 1, "verified"):
            with self.assertRaises(CheckpointError):
                update_checkpoint(self.path, complete, completion_verified=flag)
            self.assert_unchanged(original)
        update_checkpoint(self.path, complete, completion_verified=True)
        self.assertEqual(load_checkpoint(self.path), complete)
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, self.running())

    def test_stopped_state_preserves_stage_task_sessions_and_reason(self) -> None:
        self.create()
        state = dict(self.running(), implementer_session_id=_A)
        update_checkpoint(self.path, state)
        stopped = dict(state, status="stopped", stop_reason_code="TEST_FAILED")
        update_checkpoint(self.path, stopped)
        self.assertEqual(load_checkpoint(self.path), stopped)
        update_checkpoint(self.path, state)
        self.assertEqual(load_checkpoint(self.path), state)
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, initial_state())

    def test_stop_before_task_is_valid_but_nonpending_creation_is_not(self) -> None:
        self.create()
        stopped = dict(initial_state(), status="stopped", stop_reason_code="USER_STOP")
        update_checkpoint(self.path, stopped)
        self.assertEqual(load_checkpoint(self.path), stopped)
        with self.assertRaises(CheckpointError):
            create_checkpoint(self.directory / "other.json", self.running())

    def test_corrupt_truncated_duplicate_and_unsupported_json_are_rejected(self) -> None:
        invalid = (
            b"", b"{", b'{"schema_version":1', b"\xff", b"[]", b"null", b"1",
            b'{"schema_version":1,"schema_version":1}',
            json.dumps(dict(initial_state(), schema_version=2)).encode(),
            json.dumps(dict(initial_state(), extra=True)).encode(),
            b" " * (16 * 1024 + 1), b"[" * 2000 + b"]" * 2000,
        )
        for raw in invalid:
            with self.subTest(raw_length=len(raw)):
                self.path.write_bytes(raw)
                with self.assertRaises(CheckpointError):
                    load_checkpoint(self.path)
                self.assertEqual(self.path.read_bytes(), raw)

    def test_invalid_existing_checkpoint_cannot_be_updated(self) -> None:
        self.path.write_bytes(b"corrupt")
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, self.running())
        self.assert_unchanged(b"corrupt")

    def test_missing_checkpoint_is_not_automatically_created_by_update(self) -> None:
        with self.assertRaises(FileNotFoundError):
            update_checkpoint(self.path, self.running())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_replacement_is_same_directory_closed_and_complete(self) -> None:
        self.create()
        original = self.path.read_bytes()
        replace = os.replace

        def inspect(source, destination):
            self.assertEqual(Path(source).parent, self.path.parent)
            self.assertEqual(Path(destination), self.path)
            self.assertEqual(load_checkpoint(source), self.running())
            self.assertEqual(self.path.read_bytes(), original)
            replace(source, destination)

        with patch("auto_coding.batch_state.os.replace", side_effect=inspect) as replacement:
            update_checkpoint(self.path, self.running())
            replacement.assert_called_once()
        self.assertEqual(load_checkpoint(self.path), self.running())
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_failed_replacement_preserves_previous_and_removes_temp(self) -> None:
        self.create()
        original = self.path.read_bytes()
        with patch("auto_coding.batch_state.os.replace", side_effect=OSError(errno.EACCES, "denied")):
            with self.assertRaises(OSError):
                update_checkpoint(self.path, self.running())
        self.assert_unchanged(original)

    def test_interruption_before_replacement_preserves_previous_and_removes_temp(self) -> None:
        self.create()
        original = self.path.read_bytes()
        with patch("auto_coding.batch_state.os.replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                update_checkpoint(self.path, self.running())
        self.assert_unchanged(original)

    def test_sync_failure_preserves_previous_and_cleans_temp(self) -> None:
        self.create()
        original = self.path.read_bytes()
        with patch("auto_coding.batch_state.os.fsync", side_effect=OSError(errno.EIO, "failure")):
            with self.assertRaises(OSError):
                update_checkpoint(self.path, self.running())
        self.assert_unchanged(original)

    def test_failed_initial_publication_cleans_temp_and_creates_no_checkpoint(self) -> None:
        with patch("auto_coding.batch_state.os.link", side_effect=OSError(errno.EIO, "failure")):
            with self.assertRaises(OSError):
                self.create()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_initial_sync_creates_no_checkpoint(self) -> None:
        with patch("auto_coding.batch_state.os.fsync", side_effect=OSError(errno.EIO, "failure")):
            with self.assertRaises(OSError):
                self.create()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_caller_owned_input_is_not_changed(self) -> None:
        state = initial_state()
        expected = deepcopy(state)
        self.create()
        update = self.running()
        update_expected = deepcopy(update)
        update_checkpoint(self.path, update)
        self.assertEqual(state, expected)
        self.assertEqual(update, update_expected)

    def test_external_checkpoint_does_not_modify_target_repository(self) -> None:
        target = self.directory / "target"
        target.mkdir()
        subprocess.run(
            ["git", "init", "--quiet", str(target)], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
        (target / "source.txt").write_bytes(b"target content")

        def snapshot():
            return {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}

        before = snapshot()
        self.create()
        update_checkpoint(self.path, self.running())
        load_checkpoint(self.path)
        self.assertEqual(snapshot(), before)


class SequentialCheckpointTests(unittest.TestCase):
    def test_timeout_preserves_prior_verified_publication_and_original_base(self):
        create_checkpoint(self.path, self.initial)
        state = dict(self.first_completed(), status="stopped", stop_reason_code="TEST_TIMEOUT", stage="testing",
                     timeout=TimeoutDiagnostic("tests", 1800, 1800.5).record())
        update_checkpoint(self.path, state)
        self.assertEqual(load_checkpoint(self.path), state)
        self.assertEqual(state["base_head"], self.initial["base_head"])
        self.assertEqual(state["publications"][0]["task_id"], "task-1")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "checkpoint.json"
        self.initial = dict(initial_state(), schema_version=2, task_ids=["task-1", "task-2"],
                            publications=[], current_head="b" * 40)

    def first_completed(self):
        entry = dict(task_id="task-1", starting_head="b" * 40, commit_sha="c" * 40,
                     remote_sha="c" * 40, finalized=True)
        return dict(self.initial, status="running", stage="implementation", current_task_id="task-2",
                    publications=[entry], current_head="c" * 40)

    def test_verified_chain_is_persisted_without_changing_original_base(self):
        validate_checkpoint_capacity(self.initial)
        create_checkpoint(self.path, self.initial)
        first = self.first_completed()
        update_checkpoint(self.path, first)
        final = dict(first, status="completed", stage="finished", current_task_id=None,
                     current_head="d" * 40, publications=[*first["publications"],
                     dict(task_id="task-2", starting_head="c" * 40, commit_sha="d" * 40,
                          remote_sha="d" * 40, finalized=True)])
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, final)
        update_checkpoint(self.path, final, completion_verified=True)
        self.assertEqual(load_checkpoint(self.path), final)
        self.assertEqual(final["base_head"], self.initial["base_head"])

    def test_malformed_chain_or_scope_and_history_rewrites_fail_closed(self):
        create_checkpoint(self.path, self.initial)
        first = self.first_completed()
        update_checkpoint(self.path, first)
        before = self.path.read_bytes()
        invalid = [dict(first, publications=[], current_head="b" * 40, current_task_id="task-1"),
                   dict(first, base_head="a" * 40), dict(first, current_head="a" * 40),
                   dict(first, task_ids=["task-1", "other"]), dict(first, current_task_id="task-1"),
                   dict(first, implementer_session_id=_A), dict(first, schema_version=1)]
        for changes in ({"starting_head": "0" * 40}, {"remote_sha": "0" * 40},
                        {"task_id": "task-2"}, {"finalized": 1}, {"extra": "field"}):
            invalid.append(dict(first, publications=[dict(first["publications"][0], **changes)]))
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(CheckpointError):
                update_checkpoint(self.path, candidate)
            self.assertEqual(self.path.read_bytes(), before)

    def test_verified_push_with_failed_finalization_is_durable_and_cannot_resume(self):
        create_checkpoint(self.path, self.initial)
        first = self.first_completed()
        first.update(status="stopped", current_task_id="task-1", stage="publishing",
                     stop_reason_code="REPORT_UNAVAILABLE",
                     publications=[dict(first["publications"][0], finalized=False)])
        update_checkpoint(self.path, first)
        self.assertEqual(load_checkpoint(self.path), first)
        with self.assertRaises(CheckpointError):
            update_checkpoint(self.path, dict(first, status="running", stop_reason_code=None))

    def test_capacity_reservation_rejects_large_ledger_without_writes(self):
        large = dict(self.initial, task_ids=[f"task-{index:04d}-" + "a" * 50 for index in range(100)])
        with self.assertRaises(CheckpointError):
            validate_checkpoint_capacity(large)
        self.assertFalse(self.path.exists())
        for task_ids in ([], ["task-1", "task-1"], ["bad/id"], ["task-1"] * 101):
            with self.subTest(task_ids=task_ids), self.assertRaises(CheckpointError):
                create_checkpoint(self.path, dict(self.initial, task_ids=task_ids))
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
