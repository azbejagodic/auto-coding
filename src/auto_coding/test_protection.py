"""Bounded controller observations of explicitly approved test artifacts.

These are integrity snapshots, not a filesystem sandbox or a lock. Protection
requires an honest, complete input declaration and a trusted host/interpreter.
Never infer dependencies from filenames, argv, model claims or test output.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

from auto_coding.change_evidence import _safe_location
from auto_coding.plans import PlanValidationError, _test_contract


@dataclass(frozen=True)
class TestProtectionResult:
    passed: bool
    error_code: str | None = None


def check_test_protection(repository_directory, contract, command_count) -> TestProtectionResult:
    """Check regular, unaliased artifact bytes against the approved manifest.

    At most 64 KiB is buffered. The approved size bounds each read sequence;
    mismatches, inaccessible files, symlinks/junctions and hard links fail closed.
    No file contents, exception messages or paths are returned.
    """
    if contract is None:
        return TestProtectionResult(False, "TEST_CONTRACT_UNAVAILABLE")
    try:
        _test_contract(contract, command_count)
    except (PlanValidationError, TypeError, ValueError):
        return TestProtectionResult(False, "TEST_CONTRACT_INVALID")
    try:
        repository = Path(repository_directory).resolve(strict=True)
        if not repository.is_dir():
            return TestProtectionResult(False, "PROTECTED_TEST_INPUT_UNSAFE")
        for artifact in contract["protected_files"]:
            path = _safe_location(repository, artifact["path"])
            before = path.lstat()
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                    or getattr(before, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                return TestProtectionResult(False, "PROTECTED_TEST_INPUT_UNSAFE")
            if before.st_size != artifact["size_bytes"]:
                return TestProtectionResult(False, "PROTECTED_TEST_INPUT_CHANGED")
            digest = hashlib.sha256()
            remaining = artifact["size_bytes"]
            with path.open("rb") as source:
                opened = os.fstat(source.fileno())
                if not os.path.samestat(before, opened):
                    return TestProtectionResult(False, "PROTECTED_TEST_INPUT_CHANGED")
                while remaining:
                    chunk = source.read(min(64 * 1024, remaining))
                    if not chunk:
                        return TestProtectionResult(False, "PROTECTED_TEST_INPUT_CHANGED")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if source.read(1):
                    return TestProtectionResult(False, "PROTECTED_TEST_INPUT_CHANGED")
                after = os.fstat(source.fileno())
            final = _safe_location(repository, artifact["path"]).lstat()
            # Windows stat/lstat and fstat can report different ctime values
            # for one unchanged file. Compare ctime only within the same API;
            # identity, size, mtime and approved bytes must still all agree.
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_nlink)
            if (identity(before) != identity(after) or identity(before) != identity(final)
                    or before.st_ctime_ns != final.st_ctime_ns or opened.st_ctime_ns != after.st_ctime_ns
                    or digest.hexdigest() != artifact["sha256"]):
                return TestProtectionResult(False, "PROTECTED_TEST_INPUT_CHANGED")
        return TestProtectionResult(True)
    except (OSError, ValueError, RuntimeError):
        return TestProtectionResult(False, "PROTECTED_TEST_INPUT_UNAVAILABLE")
