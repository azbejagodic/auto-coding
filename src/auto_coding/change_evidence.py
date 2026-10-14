"""Collect bounded, read-only evidence of pending Git changes."""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat
import subprocess

from auto_coding.plans import PlanValidationError, _path
from auto_coding.scope_check import check_file_scope


_DIFF_OPTIONS = ["--no-ext-diff", "--no-textconv", "--no-renames"]


@dataclass(frozen=True)
class UntrackedFile:
    path: str
    contents: bytes = field(repr=False)


@dataclass(frozen=True)
class ChangeEvidence:
    passed: bool
    changed_paths: tuple[str, ...] = ()
    tracked_diff: bytes = field(default=b"", repr=False)
    untracked_files: tuple[UntrackedFile, ...] = field(default=(), repr=False)
    fingerprint: str | None = None
    error_code: str | None = None


def _git(repository: Path, arguments: list[str]) -> tuple[bytes | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=repository, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False,
            check=False, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return None, "GIT_TIMEOUT"
    except OSError:
        return None, "GIT_LAUNCH_FAILED"
    if result.returncode != 0:
        return None, "GIT_FAILED"
    return result.stdout, None


def _nul_paths(output: bytes) -> tuple[str, ...]:
    if output and not output.endswith(b"\0"):
        raise ValueError("malformed Git paths")
    try:
        paths = tuple(value.decode("utf-8") for value in output.split(b"\0")[:-1])
    except UnicodeDecodeError as exc:
        raise ValueError("malformed Git paths") from exc
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("malformed Git paths")
    return paths


def _tracked_paths(output: bytes) -> tuple[str, ...]:
    if output and not output.endswith(b"\0"):
        raise ValueError("malformed Git raw diff")
    records = output.split(b"\0")[:-1] if output else []
    paths = []
    index = 0
    while index < len(records):
        metadata = records[index]
        fields = metadata[1:].split(b" ") if metadata.startswith(b":") else []
        if len(fields) != 5 or any(len(mode) != 6 or not mode.isdigit()
                                   for mode in fields[:2]):
            raise ValueError("malformed Git raw diff")
        if b"120000" in fields[:2]:
            raise ValueError("symlink in Git raw diff")
        status = fields[4][:1]
        if status not in (b"A", b"C", b"D", b"M", b"R", b"T", b"U", b"X", b"B"):
            raise ValueError("malformed Git raw diff")
        count = 2 if status in (b"R", b"C") else 1
        if index + count >= len(records):
            raise ValueError("malformed Git raw diff")
        try:
            paths.extend(record.decode("utf-8") for record in records[index + 1:
                                                                        index + count + 1])
        except UnicodeDecodeError as exc:
            raise ValueError("malformed Git raw diff") from exc
        index += count + 1
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("malformed Git raw diff")
    return tuple(paths)


def _safe_location(repository: Path, relative: str) -> Path:
    _path(relative, "changed path")
    target = repository.joinpath(*relative.split("/"))
    current = repository
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("symlink path")
    if not target.resolve(strict=False).is_relative_to(repository):
        raise ValueError("escaping path")
    return target


def _digest_part(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def collect_change_evidence(
    repository_directory: str | Path, permitted_files: list[str], max_bytes: int,
) -> ChangeEvidence:
    """Capture complete pending changes; callers must recheck before publishing."""
    if type(max_bytes) is not int or max_bytes <= 0:
        return ChangeEvidence(False, error_code="INVALID_MAX_BYTES")
    scope = check_file_scope(repository_directory, permitted_files)
    if not scope.passed:
        return ChangeEvidence(False, error_code=scope.error_code or "OUT_OF_SCOPE")

    try:
        repository = Path(repository_directory).resolve(strict=True)
        if not repository.is_dir():
            return ChangeEvidence(False, error_code="INVALID_REPOSITORY")
        for path in scope.changed_files:
            _safe_location(repository, path)
    except (OSError, ValueError, PlanValidationError):
        return ChangeEvidence(False, error_code="UNSAFE_PATH")

    raw, error = _git(repository, ["diff", *_DIFF_OPTIONS, "--raw", "-z", "HEAD", "--"])
    if error:
        return ChangeEvidence(False, error_code=error)
    try:
        tracked_paths = _tracked_paths(raw)
    except ValueError:
        return ChangeEvidence(False, error_code="UNSAFE_OR_MALFORMED_DIFF")

    untracked_raw, error = _git(repository, ["ls-files", "--others", "--exclude-standard", "-z"])
    if error:
        return ChangeEvidence(False, error_code=error)
    try:
        untracked_paths = _nul_paths(untracked_raw)
        for path in (*tracked_paths, *untracked_paths):
            _safe_location(repository, path)
    except (ValueError, PlanValidationError, OSError):
        return ChangeEvidence(False, error_code="UNSAFE_PATH")
    changed = tuple(sorted(set(scope.changed_files)))
    if set(changed) != set(tracked_paths) | set(untracked_paths):
        return ChangeEvidence(False, error_code="INCONSISTENT_EVIDENCE")

    diff, error = _git(repository, ["diff", *_DIFF_OPTIONS, "--binary", "--full-index",
                                    "HEAD", "--"])
    if error:
        return ChangeEvidence(False, error_code=error)
    total = len(diff) + sum(len(path.encode("utf-8")) for path in changed)
    if total > max_bytes:
        return ChangeEvidence(False, error_code="EVIDENCE_TOO_LARGE")

    files = []
    for path in sorted(untracked_paths):
        target = repository.joinpath(*path.split("/"))
        try:
            _safe_location(repository, path)
            with target.open("rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    return ChangeEvidence(False, error_code="UNSAFE_PATH")
                contents = source.read(max_bytes - total + 1)
                after = os.fstat(source.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                return ChangeEvidence(False, error_code="INCONSISTENT_EVIDENCE")
        except (OSError, ValueError, PlanValidationError):
            return ChangeEvidence(False, error_code="UNTRACKED_READ_FAILED")
        total += len(contents)
        if total > max_bytes or len(contents) != before.st_size:
            return ChangeEvidence(False, error_code="EVIDENCE_TOO_LARGE")
        files.append(UntrackedFile(path, contents))

    digest = hashlib.sha256()
    digest.update(len(changed).to_bytes(8, "big"))
    for path in changed:
        _digest_part(digest, path.encode("utf-8"))
    _digest_part(digest, diff)
    digest.update(len(files).to_bytes(8, "big"))
    for item in files:
        _digest_part(digest, item.path.encode("utf-8"))
        _digest_part(digest, item.contents)
    return ChangeEvidence(True, changed, diff, tuple(files), digest.hexdigest())
