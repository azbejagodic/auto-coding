"""Read-only Git state checks before a later publishing gate."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from auto_coding.change_evidence import _external_temporary_directory
from auto_coding.plans import PlanValidationError, _branch


_MAX_OUTPUT_BYTES = 64 * 1024
_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_OPERATION_MARKERS = (
    "MERGE_HEAD", "MERGE_AUTOSTASH", "CHERRY_PICK_HEAD", "REVERT_HEAD",
    "REBASE_HEAD", "rebase-merge", "rebase-apply", "sequencer",
    "BISECT_LOG", "BISECT_START",
)
_REPOSITORY_ENVIRONMENT = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE",
)


@dataclass(frozen=True)
class GitPreflightResult:
    passed: bool
    error_code: str | None = None


def _git_query(repository, arguments, timeout_seconds, temporary_directory, environment):
    try:
        with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory) as output:
            result = subprocess.run(
                ["git", "--no-optional-locks", *arguments],
                cwd=repository, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.DEVNULL, shell=False, check=False,
                timeout=timeout_seconds, env=environment,
            )
            output.seek(0)
            captured = output.read(_MAX_OUTPUT_BYTES + 1)
            if len(captured) > _MAX_OUTPUT_BYTES:
                return None, None, "GIT_OUTPUT_TOO_LARGE"
            return result.returncode, captured, None
    except subprocess.TimeoutExpired:
        return None, None, "GIT_TIMEOUT"
    except OSError:
        return None, None, "GIT_IO_FAILED"


def _line(output: bytes) -> str:
    if not output.endswith(b"\n"):
        raise ValueError("incomplete Git output")
    value = output[:-1]
    if value.endswith(b"\r"):
        value = value[:-1]
    text = value.decode("utf-8")
    if not text or any(character in text for character in "\r\n\0"):
        raise ValueError("ambiguous Git output")
    return text


def _present(path: Path) -> bool:
    try:
        path.lstat()  # Broken symlinks count as existing markers/locks.
        return True
    except FileNotFoundError:
        return False


def check_publish_preflight(
    repository_directory: str | Path,
    expected_branch: str,
    expected_head: str,
    expected_origin_url: str,
    timeout_seconds: float,
) -> GitPreflightResult:
    """Check expected Git state without staging or modifying the repository.

    Unstaged changes are allowed. Success is a non-atomic observation, not a
    reservation: callers must revalidate immediately before later mutations.
    Both effective fetch and push origin URLs must match the supplied URL.
    """
    try:
        valid_timeout = (
            not isinstance(timeout_seconds, bool)
            and isinstance(timeout_seconds, (int, float))
            and math.isfinite(timeout_seconds) and timeout_seconds > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        return GitPreflightResult(False, "INVALID_TIMEOUT")
    try:
        _branch(expected_branch)
        if (not isinstance(expected_head, str) or not _SHA.fullmatch(expected_head)
                or not isinstance(expected_origin_url, str) or not expected_origin_url.strip()
                or any(character in expected_origin_url for character in "\r\n\0")):
            return GitPreflightResult(False, "INVALID_EXPECTED_STATE")
    except PlanValidationError:
        return GitPreflightResult(False, "INVALID_EXPECTED_STATE")
    try:
        repository = Path(repository_directory).resolve(strict=True)
        if not repository.is_dir():
            return GitPreflightResult(False, "INVALID_REPOSITORY")
        temporary_directory = _external_temporary_directory(repository)
        if temporary_directory is None:
            return GitPreflightResult(False, "UNSAFE_TEMP_LOCATION")
    except (OSError, ValueError, TypeError):
        return GitPreflightResult(False, "INVALID_REPOSITORY")
    environment = os.environ.copy()
    if any(name in environment for name in _REPOSITORY_ENVIRONMENT):
        return GitPreflightResult(False, "UNSAFE_GIT_ENVIRONMENT")
    environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    expected_head = expected_head.lower()

    def query(arguments):
        code, output, error = _git_query(
            repository, arguments, timeout_seconds, temporary_directory, environment,
        )
        if error:
            return None, error
        if code != 0:
            return None, "GIT_FAILED"
        return _line(output), None

    try:
        root, error = query(["rev-parse", "--show-toplevel"])
        if error:
            return GitPreflightResult(False, error)
        if not Path(root).is_absolute() or Path(root).resolve(strict=True) != repository:
            return GitPreflightResult(False, "REPOSITORY_ROOT_MISMATCH")
        git_directory, error = query(["rev-parse", "--absolute-git-dir"])
        if error:
            return GitPreflightResult(False, error)
        if not Path(git_directory).is_absolute():
            return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
        git_directory = Path(git_directory).resolve(strict=True)
        if not git_directory.is_dir():
            return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
        for marker in _OPERATION_MARKERS:
            if _present(git_directory / marker):
                return GitPreflightResult(False, "GIT_OPERATION_IN_PROGRESS")
        if _present(git_directory / "index.lock"):
            return GitPreflightResult(False, "INDEX_LOCK_PRESENT")
        branch, error = query(["symbolic-ref", "--quiet", "HEAD"])
        if error:
            return GitPreflightResult(False, error)
        if branch != f"refs/heads/{expected_branch}":
            return GitPreflightResult(False, "BRANCH_MISMATCH")
        head, error = query(["rev-parse", "--verify", "HEAD^{commit}"])
        if error:
            return GitPreflightResult(False, error)
        if not _SHA.fullmatch(head):
            return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
        if head.lower() != expected_head:
            return GitPreflightResult(False, "HEAD_MISMATCH")
        for arguments in (
            ["remote", "get-url", "--all", "origin"],
            ["remote", "get-url", "--push", "--all", "origin"],
        ):
            origin, error = query(arguments)
            if error:
                return GitPreflightResult(False, error)
            if origin != expected_origin_url:
                return GitPreflightResult(False, "ORIGIN_MISMATCH")
        code, output, error = _git_query(
            repository,
            ["diff", "--cached", "--quiet", "--exit-code", "--no-ext-diff",
             "--no-textconv", "HEAD", "--"],
            timeout_seconds, temporary_directory, environment,
        )
        if error:
            return GitPreflightResult(False, error)
        if code not in (0, 1):
            return GitPreflightResult(False, "GIT_FAILED")
        if output:
            return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
        if code == 1:
            return GitPreflightResult(False, "STAGED_CHANGES")
        remote, error = query(
            ["ls-remote", "--exit-code", "--refs", "origin", f"refs/heads/{expected_branch}"],
        )
        if error:
            return GitPreflightResult(False, error)
        fields = remote.split("\t")
        if (len(fields) != 2 or not _SHA.fullmatch(fields[0])
                or fields[1] != f"refs/heads/{expected_branch}"):
            return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
        if fields[0].lower() != expected_head:
            return GitPreflightResult(False, "REMOTE_HEAD_MISMATCH")
        return GitPreflightResult(True)
    except (UnicodeError, ValueError):
        return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
    except OSError:
        return GitPreflightResult(False, "GIT_IO_FAILED")
