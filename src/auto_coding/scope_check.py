"""Read-only Git changed-file scope checks.

Git-ignored files are not visible to this check.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class FileScopeResult:
    passed: bool
    changed_files: tuple[str, ...]
    out_of_scope_paths: tuple[str, ...]
    error_code: str | None = None


def _changed_paths(output: bytes) -> tuple[str, ...]:
    if not isinstance(output, bytes) or (output and not output.endswith(b"\0")):
        raise ValueError("malformed Git status")
    records = output.split(b"\0")[:-1] if output else []
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("malformed Git status")
        status = record[:2]
        if status != b"??" and (
            status == b"  " or any(char not in b" MADRCUTXB" for char in status)
        ):
            raise ValueError("malformed Git status")
        try:
            paths.append(record[3:].decode("utf-8"))
            if b"R" in status or b"C" in status:
                index += 1
                paths.append(records[index].decode("utf-8"))
        except (IndexError, UnicodeDecodeError) as exc:
            raise ValueError("malformed Git status") from exc
        if not paths[-1]:
            raise ValueError("malformed Git status")
        index += 1
    return tuple(paths)


def check_file_scope(
    repository_directory: str | Path, permitted_files: list[str]
) -> FileScopeResult:
    """Check pending Git paths against an explicit Windows-insensitive file list."""
    if not isinstance(permitted_files, list) or any(
        not isinstance(path, str) or not path for path in permitted_files
    ):
        return FileScopeResult(False, (), (), "INVALID_PERMITTED_FILES")

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repository_directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return FileScopeResult(False, (), (), "GIT_TIMEOUT")
    except OSError:
        return FileScopeResult(False, (), (), "GIT_LAUNCH_FAILED")
    if result.returncode != 0:
        return FileScopeResult(False, (), (), "GIT_FAILED")
    try:
        changed = _changed_paths(result.stdout)
    except ValueError:
        return FileScopeResult(False, (), (), "MALFORMED_STATUS")

    permitted = {path.casefold() for path in permitted_files}
    outside = tuple(path for path in changed if path.casefold() not in permitted)
    return FileScopeResult(not outside, changed, outside)
