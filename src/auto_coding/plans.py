"""Strict validation for task-plan schema versions 1 and 2."""

from copy import deepcopy
import ipaddress
from pathlib import PurePosixPath, PureWindowsPath
import re
from urllib.parse import urlsplit


class PlanValidationError(ValueError):
    """A task plan does not satisfy the v1 contract."""


_PLAN_FIELDS = {"schema_version", "batch_id", "expected_branch", "expected_head", "tasks"}
_TASK_FIELDS = {
    "task_id", "order", "depends_on", "title", "implementation", "review",
    "permitted_files", "test_commands", "acceptance_criteria", "limits",
}
_ROLE_FIELDS = {"prompt", "model", "reasoning_effort"}
_LIMIT_FIELDS = {"implementation_seconds", "review_seconds"}
# An explicit v2 approval can allow at most 24 hours for each terminal command.
MAX_TEST_SECONDS = 24 * 60 * 60
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


def _origin_url(value: object) -> None:
    """Accept credential-free Git URLs or absolute local remote paths.

    SSH account names are identities, not passwords. HTTPS userinfo, passwords,
    query/fragment tokens, helper transports and relative paths are unsupported.
    This checks URL structure; it cannot identify secrets hidden in path names.
    """
    def invalid():
        raise PlanValidationError("expected_origin_url must be a credential-free Git remote")

    if (not isinstance(value, str) or not value or len(value) > 2048
            or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        invalid()
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        if (any(c in value for c in '<>"|?*')
                or value.startswith(("//", "\\\\")) and (":" in value or "@" in PureWindowsPath(value).drive)):
            invalid()
        return
    # The common git@host:path form is SSH, never a password-bearing URL.
    if "://" not in value:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*:[^\s\\?#]+", value):
            invalid()
        return
    if any(c.isspace() for c in value) or "\\" in value:
        invalid()
    try:
        remote = urlsplit(value)
        port = remote.port  # Reject malformed or out-of-range ports.
        if "?" in value or "#" in value or remote.password is not None:
            invalid()
        if remote.scheme == "file":
            if (remote.netloc not in ("", "localhost") or port is not None
                    or not remote.path.startswith("/") or remote.path == "/"):
                invalid()
            return
        if remote.scheme not in {"https", "http", "ssh", "git"}:
            invalid()
        if not remote.hostname or not remote.path.startswith("/") or remote.path == "/" or port == 0:
            invalid()
        host = remote.hostname
        if ":" in host:
            ipaddress.IPv6Address(host)
        elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
            invalid()
        if remote.username is not None and (
            remote.scheme != "ssh" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9._-]*", remote.username)
        ):
            invalid()
    except ValueError:
        invalid()


def validate_plan(data: object) -> dict:
    """Return an independent validated copy, or raise PlanValidationError."""
    version = data.get("schema_version") if isinstance(data, dict) else None
    fields = _PLAN_FIELDS | {"expected_origin_url"} if type(version) is int and version == 2 else _PLAN_FIELDS
    plan = _fields(data, fields, "plan")
    if type(version) is not int or version not in (1, 2):
        raise PlanValidationError("schema_version must be integer 1 or 2")
    if version == 2:
        _origin_url(plan["expected_origin_url"])
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

        limit_fields = _LIMIT_FIELDS | {"test_seconds"} if version == 2 else _LIMIT_FIELDS
        limits = _fields(task["limits"], limit_fields, f"{field}.limits")
        for name in limit_fields:
            seconds = limits[name]
            if type(seconds) is not int or seconds <= 0:
                raise PlanValidationError(f"{field}.limits.{name} must be a positive integer")
        if version == 2 and limits["test_seconds"] > MAX_TEST_SECONDS:
            raise PlanValidationError(f"{field}.limits.test_seconds exceeds {MAX_TEST_SECONDS}")
        seen_ids.add(task_id)

    return deepcopy(plan)
