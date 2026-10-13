"""Strict validation for Auto-Coding v1 task plans."""

from copy import deepcopy
import re


class PlanValidationError(ValueError):
    """A task plan does not satisfy the v1 contract."""


_PLAN_FIELDS = {"schema_version", "batch_id", "expected_branch", "expected_head", "tasks"}
_TASK_FIELDS = {
    "task_id", "order", "depends_on", "title", "implementation", "review",
    "permitted_files", "test_commands", "acceptance_criteria", "limits",
}
_ROLE_FIELDS = {"prompt", "model", "reasoning_effort"}
_LIMIT_FIELDS = {"implementation_seconds", "review_seconds"}
_EFFORTS = {"low", "medium", "high", "xhigh"}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def _fields(value: object, required: set[str], field: str) -> dict:
    if not isinstance(value, dict):
        raise PlanValidationError(f"{field} must be an object")
    missing = required - value.keys()
    if missing:
        raise PlanValidationError(f"{field}.{sorted(missing)[0]} is required")
    unknown = value.keys() - required
    if unknown:
        raise PlanValidationError(f"{field} has unknown field {sorted(unknown, key=str)[0]}")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{field} must be a nonempty string")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise PlanValidationError(f"{field} must be an identifier")
    return value


def _branch(value: object) -> None:
    field = "expected_branch"
    name = _text(value, field)
    if (
        name == "@" or name.startswith("-") or name.endswith(".")
        or ".." in name or "@{" in name
        or any(ord(char) < 33 or ord(char) == 127 or char in "~^:?*[\\" for char in name)
        or any(not part or part.startswith(".") or part.endswith(".lock") for part in name.split("/"))
    ):
        raise PlanValidationError(f"{field} must be a valid Git branch name")


def _path(value: object, field: str) -> None:
    path = _text(value, field)
    parts = path.split("/")
    if (
        path.startswith("/") or "\\" in path or ":" in path
        or any(char in path for char in "*?[]{}<>\"|")
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or any(
            part in {"", ".", ".."}
            or part.lower() == ".git"
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICES
            for part in parts
        )
    ):
        raise PlanValidationError(f"{field} must be a normalized repository-relative path")


def _role(value: object, field: str) -> None:
    role = _fields(value, _ROLE_FIELDS, field)
    _text(role["prompt"], f"{field}.prompt")
    model = _text(role["model"], f"{field}.model")
    if not _MODEL.fullmatch(model) or "astra" in model.lower():
        raise PlanValidationError(f"{field}.model must be a non-Astra model ID")
    if (
        not isinstance(role["reasoning_effort"], str)
        or role["reasoning_effort"] not in _EFFORTS
    ):
        raise PlanValidationError(f"{field}.reasoning_effort is invalid")


def validate_plan(data: object) -> dict:
    """Return an independent validated copy, or raise PlanValidationError."""
    plan = _fields(data, _PLAN_FIELDS, "plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise PlanValidationError("schema_version must be integer 1")
    _identifier(plan["batch_id"], "batch_id")
    _branch(plan["expected_branch"])
    if not isinstance(plan["expected_head"], str) or not _SHA.fullmatch(plan["expected_head"]):
        raise PlanValidationError("expected_head must be a 40-character Git SHA")
    tasks = plan["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise PlanValidationError("tasks must be a nonempty array")

    seen_ids: set[str] = set()
    for position, value in enumerate(tasks, start=1):
        field = f"tasks[{position - 1}]"
        task = _fields(value, _TASK_FIELDS, field)
        task_id = _identifier(task["task_id"], f"{field}.task_id")
        if task_id in seen_ids:
            raise PlanValidationError(f"{field}.task_id is duplicated")
        if type(task["order"]) is not int or task["order"] != position:
            raise PlanValidationError(f"{field}.order must be {position}")
        dependencies = task["depends_on"]
        if not isinstance(dependencies, list):
            raise PlanValidationError(f"{field}.depends_on must be an array")
        used_dependencies: set[str] = set()
        for dependency in dependencies:
            _identifier(dependency, f"{field}.depends_on")
            if dependency not in seen_ids or dependency in used_dependencies:
                raise PlanValidationError(f"{field}.depends_on must contain unique earlier task IDs")
            used_dependencies.add(dependency)
        _text(task["title"], f"{field}.title")
        _role(task["implementation"], f"{field}.implementation")
        _role(task["review"], f"{field}.review")

        paths = task["permitted_files"]
        if not isinstance(paths, list) or not paths:
            raise PlanValidationError(f"{field}.permitted_files must be a nonempty array")
        seen_paths: set[str] = set()
        for path in paths:
            _path(path, f"{field}.permitted_files")
            normalized_case = path.casefold()
            if normalized_case in seen_paths:
                raise PlanValidationError(f"{field}.permitted_files contains a duplicate path")
            seen_paths.add(normalized_case)

        commands = task["test_commands"]
        if not isinstance(commands, list) or not commands:
            raise PlanValidationError(f"{field}.test_commands must be a nonempty array")
        for index, command in enumerate(commands):
            command_field = f"{field}.test_commands[{index}]"
            if not isinstance(command, list) or not command:
                raise PlanValidationError(f"{command_field} must be a nonempty argument array")
            for argument in command:
                _text(argument, command_field)

        criteria = task["acceptance_criteria"]
        if not isinstance(criteria, list) or not criteria:
            raise PlanValidationError(f"{field}.acceptance_criteria must be a nonempty array")
        for criterion in criteria:
            _text(criterion, f"{field}.acceptance_criteria")

        limits = _fields(task["limits"], _LIMIT_FIELDS, f"{field}.limits")
        for name in _LIMIT_FIELDS:
            seconds = limits[name]
            if type(seconds) is not int or seconds <= 0:
                raise PlanValidationError(f"{field}.limits.{name} must be a positive integer")
        seen_ids.add(task_id)

    return deepcopy(plan)
