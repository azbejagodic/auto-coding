"""Tests for approved v1 plan fingerprints and storage."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.plan_freeze import (
    PlanIntegrityError, freeze_approved_plan, load_frozen_plan, plan_fingerprint,
)
from auto_coding.plans import PlanValidationError


def make_plan() -> dict:
    return {
        "schema_version": 1,
        "batch_id": "batch-1",
        "expected_branch": "feature/002-v1-clean",
        "expected_head": "a" * 40,
        "tasks": [{
            "task_id": "task-1",
            "order": 1,
            "depends_on": [],
            "title": "Create module",
            "implementation": {
                "prompt": "Create the module.",
                "model": "gpt-6-sol",
                "reasoning_effort": "medium",
            },
            "review": {
                "prompt": "Review the module.",
                "model": "gpt-6.1-sol",
                "reasoning_effort": "high",
            },
            "permitted_files": ["src/auto_coding/example.py"],
            "test_commands": [["py", "-m", "unittest"]],
            "acceptance_criteria": ["The module imports successfully."],
            "limits": {"implementation_seconds": 60, "review_seconds": 30},
        }],
    }


class PlanFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.destination = Path(self.temp_dir.name) / "approved.json"

    def test_identical_plans_have_identical_fingerprints(self) -> None:
        plan = make_plan()
        self.assertEqual(plan_fingerprint(plan), plan_fingerprint(deepcopy(plan)))
        self.assertEqual(len(plan_fingerprint(plan)), 64)

    def test_object_key_order_does_not_change_fingerprint(self) -> None:
        plan = make_plan()
        reordered = dict(reversed(list(plan.items())))
        reordered["tasks"] = [dict(reversed(list(plan["tasks"][0].items())))]
        reordered["tasks"][0]["implementation"] = dict(
            reversed(list(plan["tasks"][0]["implementation"].items()))
        )
        self.assertEqual(plan_fingerprint(plan), plan_fingerprint(reordered))

    def test_meaningful_changes_change_fingerprint(self) -> None:
        original = make_plan()
        fingerprint = plan_fingerprint(original)
        changes = (
            ("implementation", "prompt", "Use a different prompt."),
            ("review", "model", "gpt-6-sol"),
            ("permitted_files", None, ["tests/test_example.py"]),
            ("test_commands", None, [["py", "-m", "compileall", "src"]]),
        )
        for field, nested, value in changes:
            with self.subTest(field=field):
                changed = deepcopy(original)
                if nested is None:
                    changed["tasks"][0][field] = value
                else:
                    changed["tasks"][0][field][nested] = value
                self.assertNotEqual(plan_fingerprint(changed), fingerprint)

    def test_correct_explicit_approval_freezes_and_loads_plan(self) -> None:
        plan = make_plan()
        approval = plan_fingerprint(plan)
        with patch("auto_coding.plan_freeze.os.fsync", wraps=os.fsync) as sync:
            freeze_approved_plan(plan, approval, self.destination)
            sync.assert_called_once()
        expected = json.dumps(
            plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        self.assertEqual(self.destination.read_bytes(), expected)
        self.assertEqual(load_frozen_plan(self.destination, approval), plan)

    def test_incorrect_approval_cannot_create_file(self) -> None:
        with self.assertRaises(PlanIntegrityError):
            freeze_approved_plan(make_plan(), "0" * 64, self.destination)
        self.assertFalse(self.destination.exists())

    def test_existing_file_cannot_be_overwritten(self) -> None:
        self.destination.write_bytes(b"existing contents")
        with self.assertRaises(PlanIntegrityError):
            freeze_approved_plan(make_plan(), plan_fingerprint(make_plan()), self.destination)
        self.assertEqual(self.destination.read_bytes(), b"existing contents")

    def test_changed_or_corrupted_file_is_rejected(self) -> None:
        plan = make_plan()
        approval = plan_fingerprint(plan)
        freeze_approved_plan(plan, approval, self.destination)
        self.destination.write_bytes(self.destination.read_bytes() + b" ")
        with self.assertRaises(PlanIntegrityError):
            load_frozen_plan(self.destination, approval)
        self.destination.write_bytes(b"not json")
        with self.assertRaises(PlanIntegrityError):
            load_frozen_plan(self.destination, approval)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(PlanIntegrityError):
            load_frozen_plan(self.destination, plan_fingerprint(make_plan()))

    def test_invalid_plan_cannot_be_frozen(self) -> None:
        plan = make_plan()
        plan["tasks"][0]["implementation"]["prompt"] = ""
        with self.assertRaises(PlanValidationError):
            freeze_approved_plan(plan, "0" * 64, self.destination)
        self.assertFalse(self.destination.exists())

    def test_loaded_plan_is_independent_of_original(self) -> None:
        plan = make_plan()
        original = deepcopy(plan)
        approval = plan_fingerprint(plan)
        freeze_approved_plan(plan, approval, self.destination)
        loaded = load_frozen_plan(self.destination, approval)
        loaded["tasks"][0]["implementation"]["prompt"] = "Changed after loading."
        self.assertEqual(plan, original)
        self.assertEqual(load_frozen_plan(self.destination, approval), original)


if __name__ == "__main__":
    unittest.main()
