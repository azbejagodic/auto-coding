"""Tests for strict v1 task-plan validation."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.plans import PlanValidationError, validate_plan


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


class PlanValidationTests(unittest.TestCase):
    def test_valid_one_task_plan_returns_independent_copy(self) -> None:
        plan = make_plan()
        original = deepcopy(plan)
        validated = validate_plan(plan)
        self.assertEqual(validated, original)
        self.assertEqual(plan, original)
        self.assertIsNot(validated, plan)
        self.assertIsNot(validated["tasks"][0], plan["tasks"][0])
        validated["tasks"][0]["implementation"]["prompt"] = "Changed"
        self.assertEqual(plan, original)

    def test_valid_two_task_plan(self) -> None:
        plan = make_plan()
        second = deepcopy(plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        second["permitted_files"] = ["tests/test_example.py"]
        plan["tasks"].append(second)
        self.assertEqual(len(validate_plan(plan)["tasks"]), 2)

    def test_missing_and_unknown_fields(self) -> None:
        for target, key, value in (
            ("plan", "batch_id", "x"),
            ("task", "title", "x"),
            ("role", "model", "x"),
            ("limits", "review_seconds", 1),
        ):
            with self.subTest(target=target, key=key):
                for unknown in (False, True):
                    plan = make_plan()
                    obj = {"plan": plan, "task": plan["tasks"][0],
                           "role": plan["tasks"][0]["review"],
                           "limits": plan["tasks"][0]["limits"]}[target]
                    if unknown:
                        obj["unexpected"] = value
                    else:
                        del obj[key]
                    with self.assertRaisesRegex(PlanValidationError, "unexpected|" + key):
                        validate_plan(plan)

    def test_invalid_types_and_empty_prompts(self) -> None:
        changes = (
            ("schema_version", True),
            ("tasks", {}),
            ("batch_id", "bad id"),
            ("expected_branch", "bad branch"),
            ("expected_head", "abc"),
        )
        for key, value in changes:
            with self.subTest(field=key):
                plan = make_plan()
                plan[key] = value
                with self.assertRaisesRegex(PlanValidationError, key):
                    validate_plan(plan)
        for role in ("implementation", "review"):
            with self.subTest(role=role):
                plan = make_plan()
                plan["tasks"][0][role]["prompt"] = "  "
                with self.assertRaisesRegex(PlanValidationError, role + r"\.prompt"):
                    validate_plan(plan)

    def test_duplicate_ids_orders_and_dependencies(self) -> None:
        plan = make_plan()
        second = deepcopy(plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        plan["tasks"].append(second)
        for change, field in (
            ({"task_id": "task-1"}, "task_id"),
            ({"order": 3}, "order"),
            ({"depends_on": ["task-2"]}, "depends_on"),
            ({"depends_on": ["missing"]}, "depends_on"),
            ({"depends_on": ["task-1", "task-1"]}, "depends_on"),
        ):
            with self.subTest(change=change):
                candidate = deepcopy(plan)
                candidate["tasks"][1].update(change)
                with self.assertRaisesRegex(PlanValidationError, field):
                    validate_plan(candidate)
        plan["tasks"][0]["depends_on"] = ["task-2"]
        with self.assertRaisesRegex(PlanValidationError, "depends_on"):
            validate_plan(plan)

    def test_unsafe_and_duplicate_paths(self) -> None:
        for path in ("/absolute.py", "../escape.py", "a/../b.py", "a\\b.py",
                     "C:/drive.py", ".git/config", "a/.git/config", "a/*.py",
                     "a//b.py", "a/./b.py"):
            with self.subTest(path=path):
                plan = make_plan()
                plan["tasks"][0]["permitted_files"] = [path]
                with self.assertRaisesRegex(PlanValidationError, "permitted_files"):
                    validate_plan(plan)
        plan = make_plan()
        plan["tasks"][0]["permitted_files"] *= 2
        with self.assertRaisesRegex(PlanValidationError, "permitted_files"):
            validate_plan(plan)

    def test_windows_device_names_are_rejected(self) -> None:
        devices = ("CON", "PRN", "AUX", "NUL") + tuple(
            f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
        )
        for device in devices:
            for component in (device, device.lower() + ".txt", device.title() + ".log"):
                with self.subTest(component=component):
                    plan = make_plan()
                    plan["tasks"][0]["permitted_files"] = [f"src/{component}"]
                    with self.assertRaisesRegex(PlanValidationError, "permitted_files"):
                        validate_plan(plan)

    def test_windows_invalid_component_endings_and_characters(self) -> None:
        for path in ("folder./file.py", "folder /file.py", "folder/file.",
                     "folder/file ", "folder/a<b.py", "folder/a>b.py",
                     'folder/a"b.py', "folder/a|b.py"):
            with self.subTest(path=path):
                plan = make_plan()
                plan["tasks"][0]["permitted_files"] = [path]
                with self.assertRaisesRegex(PlanValidationError, "permitted_files"):
                    validate_plan(plan)

    def test_case_only_duplicate_paths_are_rejected(self) -> None:
        plan = make_plan()
        plan["tasks"][0]["permitted_files"] = ["src/Example.py", "SRC/example.PY"]
        with self.assertRaisesRegex(PlanValidationError, "permitted_files"):
            validate_plan(plan)

    def test_normal_windows_paths_remain_valid(self) -> None:
        plan = make_plan()
        paths = ["src/auto_coding/example.py", "docs/conductor.md", "ports/COM10.txt"]
        plan["tasks"][0]["permitted_files"] = paths
        self.assertEqual(validate_plan(plan)["tasks"][0]["permitted_files"], paths)

    def test_invalid_test_commands(self) -> None:
        for commands in ([], ["py"], [[]], [[""]], [[" "]], [["py", 1]]):
            with self.subTest(commands=commands):
                plan = make_plan()
                plan["tasks"][0]["test_commands"] = commands
                with self.assertRaisesRegex(PlanValidationError, "test_commands"):
                    validate_plan(plan)

    def test_invalid_model_and_effort(self) -> None:
        for role in ("implementation", "review"):
            for key, value in (("model", "gpt-6-astra"), ("model", ""),
                               ("reasoning_effort", "ultra")):
                with self.subTest(role=role, field=key, value=value):
                    plan = make_plan()
                    plan["tasks"][0][role][key] = value
                    with self.assertRaisesRegex(PlanValidationError, role + r"\." + key):
                        validate_plan(plan)

    def test_invalid_timeouts(self) -> None:
        for key in ("implementation_seconds", "review_seconds"):
            for value in (0, -1, True, 1.5, "60"):
                with self.subTest(field=key, value=value):
                    plan = make_plan()
                    plan["tasks"][0]["limits"][key] = value
                    with self.assertRaisesRegex(PlanValidationError, key):
                        validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
