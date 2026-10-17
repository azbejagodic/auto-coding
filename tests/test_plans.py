"""Tests for strict v1 task-plan validation."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.plans import PlanValidationError, validate_plan
from auto_coding.plans import MAX_TEST_SECONDS
from auto_coding.plan_freeze import plan_fingerprint


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


def make_v2_plan(origin="https://example.invalid/project.git") -> dict:
    plan = make_plan()
    plan.update(schema_version=2, expected_origin_url=origin)
    plan["tasks"][0]["limits"]["test_seconds"] = 120
    return plan


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


class PlanVersionTwoTests(unittest.TestCase):
    def test_recommended_execution_timeouts_are_explicit_approved_values(self):
        plan = make_v2_plan()
        plan["tasks"][0]["limits"].update(implementation_seconds=1800, review_seconds=600, test_seconds=1800)
        self.assertEqual(validate_plan(plan), plan)

    def test_valid_v2_keeps_exact_commands_and_explicit_timeout(self):
        plan = make_v2_plan()
        plan["tasks"][0]["test_commands"].append(["py", "-m", "compileall", "src"])
        self.assertEqual(validate_plan(plan), plan)
        self.assertEqual(plan["tasks"][0]["limits"]["test_seconds"], 120)

    def test_every_task_requires_explicit_test_timeout(self):
        plan = make_v2_plan()
        second = deepcopy(plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        del second["limits"]["test_seconds"]
        plan["tasks"].append(second)
        with self.assertRaisesRegex(PlanValidationError, "test_seconds"):
            validate_plan(plan)

    def test_missing_and_invalid_terminal_timeouts_rejected(self):
        for value in (0, -1, True, False, 1.5, "120", None, MAX_TEST_SECONDS + 1, 10**100):
            with self.subTest(value=value):
                plan = make_v2_plan()
                plan["tasks"][0]["limits"]["test_seconds"] = value
                with self.assertRaisesRegex(PlanValidationError, "test_seconds"):
                    validate_plan(plan)
        plan = make_v2_plan()
        del plan["tasks"][0]["limits"]["test_seconds"]
        with self.assertRaisesRegex(PlanValidationError, "test_seconds"):
            validate_plan(plan)
        for seconds in (1, MAX_TEST_SECONDS):
            plan = make_v2_plan()
            plan["tasks"][0]["limits"]["test_seconds"] = seconds
            self.assertEqual(validate_plan(plan), plan)

    def test_timeout_and_remote_identity_change_canonical_fingerprint(self):
        plan = make_v2_plan()
        original = plan_fingerprint(plan)
        changed_timeout = deepcopy(plan)
        changed_timeout["tasks"][0]["limits"]["test_seconds"] += 1
        self.assertNotEqual(plan_fingerprint(changed_timeout), original)
        changed_origin = dict(plan, expected_origin_url="https://example.invalid/other.git")
        self.assertNotEqual(plan_fingerprint(changed_origin), original)
        self.assertEqual(plan_fingerprint(deepcopy(plan)), original)

    def test_legitimate_network_and_absolute_local_remotes(self):
        for origin in (
            "https://github.com/example/project.git", "http://localhost:8080/project.git",
            "ssh://git@example.invalid:2222/project.git", "git@example.invalid:project.git",
            "git://example.invalid/project.git", "ssh://git@[::1]/project.git",
            "file:///C:/tmp/remote.git", "file:///tmp/remote.git", "file://localhost/tmp/remote.git",
            "C:\\Users\\Test User\\remote.git", "C:/tmp/remote.git", "/tmp/local remote.git",
            "\\\\server\\share\\remote.git",
        ):
            with self.subTest(origin=origin):
                self.assertEqual(validate_plan(make_v2_plan(origin))["expected_origin_url"], origin)

    def test_credential_bearing_and_ambiguous_remotes_rejected(self):
        for origin in (
            None, 42, "", " ", "-remote", "relative/remote.git", "../remote.git",
            "https://user:password@example.invalid/repo", "https://token@example.invalid/repo",
            "ssh://git:password@example.invalid/repo", "https://example.invalid/repo?token=secret",
            "https://example.invalid/repo?", "https://example.invalid/repo#",
            "//user:password@example.invalid/repo", "\\\\user:password@host\\share\\repo",
            "https://example.invalid/repo#secret", "file://user:secret@host/repo",
            "ext::some-command", "unknown://example.invalid/repo", "https://example.invalid",
            "https://example.invalid:99999/repo", "https://example.invalid:0/repo",
            "https://example.invalid/repo\n", "https://example.invalid/repo\0",
            "https://example.invalid/repo space", "https://bad%40host/repo",
            "git:password@example.invalid:repo", "https://example.invalid\\repo",
            "x" * 2049,
        ):
            with self.subTest(origin=origin):
                with self.assertRaisesRegex(PlanValidationError, "expected_origin_url") as error:
                    validate_plan(make_v2_plan(origin))
                self.assertNotIn("password", str(error.exception))

    def test_v1_is_not_silently_upgraded(self):
        plan = make_plan()
        self.assertEqual(validate_plan(plan), plan)
        self.assertNotIn("test_seconds", validate_plan(plan)["tasks"][0]["limits"])
        for modified in (
            dict(plan, expected_origin_url="https://example.invalid/project.git"),
            dict(plan, schema_version=3), dict(plan, schema_version=True),
        ):
            with self.assertRaises(PlanValidationError):
                validate_plan(modified)
        plan["tasks"][0]["limits"]["test_seconds"] = 120
        with self.assertRaises(PlanValidationError):
            validate_plan(plan)

    def test_v2_required_origin_and_unknown_fields_rejected(self):
        plan = make_v2_plan()
        del plan["expected_origin_url"]
        with self.assertRaisesRegex(PlanValidationError, "expected_origin_url"):
            validate_plan(plan)
        for location in ("plan", "task", "limits", "role"):
            plan = make_v2_plan()
            targets = {"plan": plan, "task": plan["tasks"][0], "limits": plan["tasks"][0]["limits"],
                       "role": plan["tasks"][0]["review"]}
            targets[location]["unexpected"] = "value"
            with self.assertRaisesRegex(PlanValidationError, "unexpected"):
                validate_plan(plan)

    def test_v2_retains_task_structure_and_safety_validation(self):
        for field, value in (
            ("task_id", "bad id"), ("order", 2), ("depends_on", ["missing"]),
            ("test_commands", [["py", 5]]), ("permitted_files", ["../unsafe"]),
            ("acceptance_criteria", []),
        ):
            with self.subTest(field=field):
                plan = make_v2_plan()
                plan["tasks"][0][field] = value
                with self.assertRaises(PlanValidationError):
                    validate_plan(plan)
        for field, value in (("model", "gpt-6-astra"), ("prompt", ""), ("reasoning_effort", "ultra")):
            plan = make_v2_plan()
            plan["tasks"][0]["review"][field] = value
            with self.assertRaises(PlanValidationError):
                validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
