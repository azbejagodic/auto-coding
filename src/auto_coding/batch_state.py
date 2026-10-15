"""Single-process checkpoint storage, separate from operational event logging.

Callers choose a metadata destination outside the target repository. Writes are
flushed and fsynced before publication; atomic replacement does not guarantee
survival of a filesystem/power failure. No recovery or concurrent-writer locking
is provided. Completion verification and frozen-plan authentication belong to
the caller; this module only records explicitly supplied state.
"""

import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from auto_coding.plans import PlanValidationError, _branch


class CheckpointError(ValueError):
    """Checkpoint metadata or its requested update is invalid."""


_FIELDS = {
    "schema_version", "batch_id", "frozen_plan_sha256", "expected_branch",
    "base_head", "status", "current_task_id", "stage", "implementer_session_id",
    "reviewer_session_id", "stop_reason_code",
}
_IMMUTABLE = ("batch_id", "frozen_plan_sha256", "expected_branch", "base_head")
_SESSIONS = ("implementer_session_id", "reviewer_session_id")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_REASON = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HEAD = re.compile(r"[0-9a-fA-F]{40}\Z")
_ACTIVE_STAGES = {"implementation", "testing", "review", "staging", "publishing"}
_MAX_BYTES = 16 * 1024


def _matches(value: object, pattern: re.Pattern) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _validate(data: object) -> dict:
    if not isinstance(data, dict) or data.keys() != _FIELDS:
        raise CheckpointError("checkpoint fields are invalid")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise CheckpointError("unsupported checkpoint version")
    if not _matches(data["batch_id"], _IDENTIFIER):
        raise CheckpointError("invalid batch identifier")
    if not _matches(data["frozen_plan_sha256"], _SHA256):
        raise CheckpointError("invalid plan fingerprint")
    if not _matches(data["base_head"], _HEAD):
        raise CheckpointError("invalid base HEAD")
    branch = data["expected_branch"]
    if not isinstance(branch, str) or len(branch) > 512:
        raise CheckpointError("invalid expected branch")
    try:
        _branch(branch)
    except PlanValidationError:
        raise CheckpointError("invalid expected branch") from None
    task = data["current_task_id"]
    if task is not None and not _matches(task, _IDENTIFIER):
        raise CheckpointError("invalid task identifier")
    for field in _SESSIONS:
        value = data[field]
        if value is None:
            continue
        if not isinstance(value, str) or len(value) != 36:
            raise CheckpointError("invalid session UUID")
        try:
            parsed = uuid.UUID(value)
        except ValueError:
            raise CheckpointError("invalid session UUID") from None
        if parsed.int == 0 or str(parsed) != value.lower():
            raise CheckpointError("invalid session UUID")
    if (data[_SESSIONS[0]] is not None and data[_SESSIONS[1]] is not None
            and data[_SESSIONS[0]].lower() == data[_SESSIONS[1]].lower()):
        raise CheckpointError("implementer and reviewer sessions must be separate")
    status, stage = data["status"], data["stage"]
    if not isinstance(status, str) or status not in {"pending", "running", "stopped", "completed"}:
        raise CheckpointError("invalid batch status")
    if not isinstance(stage, str) or stage not in _ACTIVE_STAGES | {"pending", "finished"}:
        raise CheckpointError("invalid execution stage")
    reason = data["stop_reason_code"]
    if status == "stopped":
        if not _matches(reason, _REASON):
            raise CheckpointError("stopped batch requires a safe reason code")
    elif reason is not None:
        raise CheckpointError("stop reason requires stopped status")
    if status == "pending":
        if stage != "pending" or task is not None or any(data[f] is not None for f in _SESSIONS):
            raise CheckpointError("inconsistent pending checkpoint")
    elif status == "completed":
        if stage != "finished" or task is not None:
            raise CheckpointError("inconsistent completed checkpoint")
    elif status == "running":
        if stage not in _ACTIVE_STAGES or task is None:
            raise CheckpointError("running batch requires an active task and stage")
    elif stage == "finished" or (stage == "pending") != (task is None):
        raise CheckpointError("inconsistent stopped checkpoint")
    return dict(data)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("duplicate checkpoint field")
        result[key] = value
    return result


def load_checkpoint(destination: str | Path) -> dict:
    """Load validated metadata without inferring progress or recovering state.

Invalid content raises CheckpointError; filesystem failures remain OSError.
"""
    with Path(destination).open("rb") as source:
        raw = source.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise CheckpointError("checkpoint exceeds metadata size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise CheckpointError("invalid checkpoint JSON") from None
    return _validate(data)


def _write(destination: Path, data: dict, *, creating: bool) -> None:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(raw) > _MAX_BYTES:
        raise CheckpointError("checkpoint exceeds metadata size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".batch-checkpoint-", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        if creating:
            # Publish complete bytes without overwriting even an existing empty
            # file or dangling symlink. Unsupported hard links fail closed.
            os.link(temporary, destination)
        else:
            os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_checkpoint(destination: str | Path, state: object) -> dict:
    """Create an initial pending checkpoint; never overwrite an existing path.

The parent directory must already exist. Initial atomic publication requires
hard-link support. Existing paths raise FileExistsError.
"""
    data = _validate(state)
    if data["status"] != "pending":
        raise CheckpointError("initial checkpoint must be pending")
    _write(Path(destination), data, creating=True)
    return data


def update_checkpoint(
    destination: str | Path, state: object, *, completion_verified: bool = False,
) -> dict:
    """Explicitly replace valid state, preserving identity and established UUIDs.

completion_verified=True is the caller's assertion of verified batch completion,
not proof produced by storage. It is required to enter completed status.
"""
    if type(completion_verified) is not bool:
        raise CheckpointError("completion verification must be boolean")
    data = _validate(state)
    previous = load_checkpoint(destination)
    if any(data[f] != previous[f] for f in _IMMUTABLE):
        raise CheckpointError("batch identity cannot change")
    if any(previous[f] is not None and data[f] != previous[f] for f in _SESSIONS):
        raise CheckpointError("established session cannot change")
    if previous["status"] == "completed" and data != previous:
        raise CheckpointError("completed checkpoint cannot resume")
    if previous["status"] != "pending" and data["status"] == "pending":
        raise CheckpointError("checkpoint cannot return to initial state")
    if (data["status"] == "completed" and previous["status"] != "completed"
            and not completion_verified):
        raise CheckpointError("completion requires explicit verified input")
    _write(Path(destination), data, creating=False)
    return data
