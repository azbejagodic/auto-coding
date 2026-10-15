"""Verified Git staging and publishing with explicit review authorization."""

from dataclasses import dataclass
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import stat
import tempfile
import uuid

from auto_coding.change_evidence import (
    ChangeEvidence, UntrackedFile, _external_temporary_directory,
    _nul_paths, _safe_location, _tracked_paths,
)
from auto_coding.plans import PlanValidationError, _branch, _path
from auto_coding.review import (
    ReviewResult, ReviewVerdict, _evidence_fingerprint, build_review_packet,
)
from auto_coding.scope_check import _changed_paths
from auto_coding.test_runner import TestCommandResult, TestRunResult


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


def _git_query(repository, arguments, timeout_seconds, temporary_directory, environment,
               max_output_bytes=_MAX_OUTPUT_BYTES):
    try:
        with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory) as output:
            result = subprocess.run(
                ["git", "--no-optional-locks", *arguments],
                cwd=repository, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.DEVNULL, shell=False, check=False,
                timeout=timeout_seconds, env=environment,
            )
            output.seek(0)
            captured = output.read(max_output_bytes + 1)
            if len(captured) > max_output_bytes:
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
    return _check_publish_state(
        repository_directory, expected_branch, expected_head, expected_origin_url, timeout_seconds,
    )


def _check_publish_state(
    repository_directory, expected_branch, expected_head, expected_origin_url, timeout_seconds,
    *, allow_staged=False, expected_remote_head=None,
):
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
        if code == 1 and not allow_staged:
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
        if fields[0].lower() != (expected_remote_head or expected_head).lower():
            return GitPreflightResult(False, "REMOTE_HEAD_MISMATCH")
        return GitPreflightResult(True)
    except (UnicodeError, ValueError):
        return GitPreflightResult(False, "MALFORMED_GIT_OUTPUT")
    except OSError:
        return GitPreflightResult(False, "GIT_IO_FAILED")


@dataclass(frozen=True)
class GitStageResult:
    passed: bool
    error_code: str | None = None
    staging_attempted: bool = False
    # Observed index delta relative to expected_head, not a reservation.
    staged_paths: tuple[str, ...] | None = None
    staged_changes_sha256: str | None = None
    staged_state_error_code: str | None = None


class _StageFailure(ValueError):
    """Carries only a fixed, sanitized error code."""


def _review_snapshot(packet, review, max_bytes):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise _StageFailure("INVALID_MAX_BYTES")
    if not isinstance(packet, bytes) or not packet or len(packet) > max_bytes:
        raise _StageFailure("INVALID_REVIEW_PACKET")
    if (not isinstance(review, ReviewResult) or review.approved is not True
            or review.error_code is not None or type(review.exit_status) is not int
            or review.exit_status != 0 or review.timed_out is not False
            or review.cleanup_result != "not_needed"
            or not isinstance(review.verdict, ReviewVerdict)):
        raise _StageFailure("REVIEW_NOT_APPROVED")
    session = uuid.UUID(review.session_id) if isinstance(review.session_id, str) else None
    if session is None or session.int == 0 or str(session) != review.session_id.lower():
        raise _StageFailure("INVALID_REVIEW_SESSION")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (review.reviewed_packet_sha256, review.reviewed_evidence_fingerprint)):
        raise _StageFailure("REVIEW_BINDING_MISMATCH")
    if review.reviewed_packet_sha256 != hashlib.sha256(packet).hexdigest():
        raise _StageFailure("REVIEW_BINDING_MISMATCH")

    data = json.loads(packet)
    evidence_data = data["verified_change_evidence"]
    evidence = ChangeEvidence(
        True, tuple(evidence_data["changed_paths"]),
        base64.b64decode(evidence_data["tracked_diff_base64"], validate=True),
        tuple(UntrackedFile(item["path"], base64.b64decode(item["contents_base64"], validate=True))
              for item in evidence_data["untracked_files"]),
        evidence_data["fingerprint"],
    )
    inputs = data["caller_supplied_review_inputs"]
    criteria = inputs["acceptance_criteria"]["items"]
    commands = inputs["approved_test_commands"]["items"]
    results = TestRunResult(True, len(commands), tuple(
        TestCommandResult(
            item["index"], item["exit_status"], item["elapsed_seconds"],
            item["timed_out"], item["cleanup_result"], item["command_identity"],
        )
        for item in inputs["terminal_test_results"]["commands"]
    ), None)
    context = [(item["path"], base64.b64decode(item["contents_base64"], validate=True))
               for item in inputs["selected_code_context"]["files"]]
    if build_review_packet(evidence, criteria, commands, results, context, max_bytes) != packet:
        raise _StageFailure("INVALID_REVIEW_PACKET")
    if review.reviewed_evidence_fingerprint != evidence.fingerprint:
        raise _StageFailure("REVIEW_BINDING_MISMATCH")
    verdict = review.verdict
    if (verdict.decision != "APPROVED" or verdict.confidence not in ("LOW", "MEDIUM", "HIGH")
            or any(not isinstance(values, tuple) or any(not isinstance(item, str) for item in values)
                   for values in (verdict.blocking_issues, verdict.nonblocking_notes,
                                  verdict.checked_requirements))
            or verdict.blocking_issues
            or any(criterion not in verdict.checked_requirements for criterion in criteria)):
        raise _StageFailure("REVIEW_NOT_APPROVED")
    return evidence


def _raw_index_entries(output):
    """Parse complete full-SHA raw records, retaining modes for verification."""
    if output and not output.endswith(b"\0"):
        raise _StageFailure("MALFORMED_GIT_OUTPUT")
    records = output.split(b"\0")[:-1] if output else []
    if len(records) % 2:
        raise _StageFailure("MALFORMED_GIT_OUTPUT")
    entries = {}
    for offset in range(0, len(records), 2):
        match = re.fullmatch(
            rb":([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([AMDTUXB])",
            records[offset],
        )
        if match is None:
            raise _StageFailure("MALFORMED_GIT_OUTPUT")
        path = records[offset + 1].decode("utf-8")
        _path(path, "staged path")
        if path in entries:
            raise _StageFailure("MALFORMED_GIT_OUTPUT")
        entries[path] = match.groups()
    return entries


def _current_review_evidence(repository, expected_head, permitted, max_bytes, query):
    # Reuse existing parsers and fingerprint semantics, with bounded status
    # capture rather than the scope check's unbounded PIPE interface.
    status = query(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    changed = tuple(sorted(set(_changed_paths(status))))
    if any(path not in permitted for path in changed):
        raise _StageFailure("OUT_OF_SCOPE")
    raw = query(["diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                 "--raw", "--abbrev=40", "-z", expected_head, "--"])
    tracked = _tracked_paths(raw)
    entries = _raw_index_entries(raw)
    if any(entry[0] not in (b"000000", b"100644", b"100755")
           or entry[1] not in (b"000000", b"100644", b"100755")
           or entry[4] not in (b"A", b"M", b"D", b"T") for entry in entries.values()):
        raise _StageFailure("UNVERIFIED_FILE_MODE")
    untracked = _nul_paths(query(["ls-files", "--others", "--exclude-standard", "-z"]))
    if set(changed) != set(tracked) | set(untracked):
        raise _StageFailure("INCONSISTENT_EVIDENCE")
    for path in changed:
        _safe_location(repository, path)
    diff = query(["diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                  "--binary", "--full-index", expected_head, "--"])
    total = len(diff) + sum(len(path.encode("utf-8")) for path in changed)
    files = []
    for path in sorted(untracked):
        target = _safe_location(repository, path)
        if not stat.S_ISREG(target.lstat().st_mode):
            raise _StageFailure("UNSAFE_PATH")
        with target.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise _StageFailure("UNSAFE_PATH")
            if total > max_bytes:
                raise _StageFailure("EVIDENCE_TOO_LARGE")
            contents = source.read(max_bytes - total + 1)
            after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise _StageFailure("INCONSISTENT_EVIDENCE")
        total += len(contents)
        if total > max_bytes or len(contents) != before.st_size:
            raise _StageFailure("EVIDENCE_TOO_LARGE")
        files.append(UntrackedFile(path, contents))
    if total > max_bytes:
        raise _StageFailure("EVIDENCE_TOO_LARGE")
    evidence = ChangeEvidence(True, changed, diff, tuple(files))
    return ChangeEvidence(True, changed, diff, tuple(files), _evidence_fingerprint(evidence))



def _verify_staged_snapshot(evidence, expected_head, raw_arguments, query, run):
    raw = query(raw_arguments)
    entries = _raw_index_entries(raw)
    if set(entries) != set(evidence.changed_paths):
        raise _StageFailure("STAGED_PATHS_MISMATCH")
    invisible_arguments = [
        "--ita-invisible-in-index" if item == "--ita-visible-in-index" else item
        for item in raw_arguments
    ]
    if query(invisible_arguments) != raw:
        raise _StageFailure("INTENT_TO_ADD")
    untracked = {item.path: item.contents for item in evidence.untracked_files}
    tracked = [path for path in evidence.changed_paths if path not in untracked]
    if tracked:
        cached = query([
            "--literal-pathspecs", "diff", "--cached", "--no-ext-diff", "--no-textconv",
            "--no-renames", "--binary", "--full-index", expected_head, "--", *tracked,
        ])
        if cached != evidence.tracked_diff:
            raise _StageFailure("STAGED_CONTENT_MISMATCH")
        # Tie the independently observed index object IDs to the full-index
        # headers in the approved diff, rather than trusting filenames or
        # a later diff observation alone.
        pairs = re.findall(
            rb"(?m)^index ([0-9a-f]{40})\.\.([0-9a-f]{40})(?: [0-7]{6})?$",
            evidence.tracked_diff,
        )
        approved_pairs = sorted(pair for pair in pairs if pair[0] != pair[1])
        observed_pairs = sorted(
            (entries[path][2], entries[path][3]) for path in tracked
            if entries[path][2] != entries[path][3]
        )
        if observed_pairs != approved_pairs:
            raise _StageFailure("STAGED_CONTENT_MISMATCH")
    elif evidence.tracked_diff:
        raise _StageFailure("STAGED_CONTENT_MISMATCH")
    for path, entry in entries.items():
        old_mode, new_mode, old_oid, new_oid, status = entry
        if (old_mode not in (b"000000", b"100644", b"100755")
                or new_mode not in (b"000000", b"100644", b"100755")
                or status not in (b"A", b"M", b"D", b"T")):
            raise _StageFailure("UNVERIFIED_FILE_MODE")
        index = query(["--literal-pathspecs", "ls-files", "--stage", "-z", "--", path])
        expected_index = new_mode + b" " + new_oid + b" 0\t" + path.encode("utf-8") + b"\0"
        if index != (b"" if new_mode == b"000000" else expected_index):
            raise _StageFailure("STAGED_CONTENT_MISMATCH")
        if path in untracked:
            if new_mode != b"100644":
                raise _StageFailure("UNVERIFIED_FILE_MODE")
            if query(["cat-file", "blob", new_oid.decode("ascii")]) != untracked[path]:
                raise _StageFailure("STAGED_CONTENT_MISMATCH")
    code, output = run(["diff", "--quiet", "--exit-code", "--no-ext-diff", "--no-textconv", "--"])
    if code != 0 or output:
        raise _StageFailure("WORKTREE_CHANGED_DURING_STAGING")
    if query(["ls-files", "--others", "--exclude-standard", "-z"]):
        raise _StageFailure("WORKTREE_CHANGED_DURING_STAGING")
    return raw


def stage_reviewed_changes(
    repository_directory: str | Path,
    expected_branch: str,
    expected_head: str,
    expected_origin_url: str,
    timeout_seconds: float,
    permitted_files: list[str],
    reviewed_packet: bytes,
    review_result: ReviewResult,
    max_bytes: int,
) -> GitStageResult:
    """Stage a bound, approved snapshot and verify its index delta.

    Permitted paths and expected state are trusted caller inputs; frozen-plan
    provenance is not established here. Checks and staging are not atomic.
    Failure after an add preserves the index and reports its observed delta,
    or explicitly reports that staged state could not be observed. No reset,
    commit, push, or automatic retry is performed.
    """
    attempted = False
    query = None
    raw_arguments = None

    def finish(error=None, verified_raw=None):
        if not attempted:
            return GitStageResult(False, error)
        try:
            raw = verified_raw if verified_raw is not None else query(raw_arguments)
            entries = _raw_index_entries(raw)
            return GitStageResult(
                error is None, error, True, tuple(sorted(entries)),
                hashlib.sha256(raw).hexdigest(),
            )
        except _StageFailure as exc:
            return GitStageResult(False, error or "STAGED_STATE_UNKNOWN", True,
                                  staged_state_error_code=str(exc))
        except (OSError, ValueError):
            return GitStageResult(False, error or "STAGED_STATE_UNKNOWN", True,
                                  staged_state_error_code="STAGED_STATE_UNKNOWN")

    try:
        evidence = _review_snapshot(reviewed_packet, review_result, max_bytes)
        if not isinstance(permitted_files, list) or not permitted_files:
            raise _StageFailure("INVALID_PERMITTED_FILES")
        total = 0
        for path in permitted_files:
            _path(path, "permitted path")
            total += len(path.encode("utf-8")) + 1
            if total > max_bytes:
                raise _StageFailure("INVALID_PERMITTED_FILES")
        permitted = set(permitted_files)
        if len({path.casefold() for path in permitted_files}) != len(permitted_files):
            raise _StageFailure("INVALID_PERMITTED_FILES")
        if not evidence.changed_paths:
            raise _StageFailure("NO_CHANGES")
        if any(path not in permitted for path in evidence.changed_paths):
            raise _StageFailure("OUT_OF_SCOPE")
        preflight = check_publish_preflight(
            repository_directory, expected_branch, expected_head, expected_origin_url, timeout_seconds,
        )
        if not preflight.passed:
            raise _StageFailure(preflight.error_code)
        repository = Path(repository_directory).resolve(strict=True)
        expected_head = expected_head.lower()
        temporary = _external_temporary_directory(repository)
        if temporary is None:
            raise _StageFailure("UNSAFE_TEMP_LOCATION")
        environment = os.environ.copy()
        if any(name in environment for name in _REPOSITORY_ENVIRONMENT):
            raise _StageFailure("UNSAFE_GIT_ENVIRONMENT")
        environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")

        def run(arguments):
            code, output, error = _git_query(
                repository, arguments, timeout_seconds, temporary, environment, max_bytes,
            )
            if error:
                raise _StageFailure(error)
            return code, output

        def query(arguments):
            code, output = run(arguments)
            if code != 0:
                raise _StageFailure("GIT_FAILED")
            return output

        raw_arguments = [
            "diff", "--cached", "--no-ext-diff", "--no-textconv", "--no-renames",
            "--raw", "--abbrev=40", "-z", "--ita-visible-in-index", expected_head, "--",
        ]
        if query(raw_arguments):
            raise _StageFailure("PREEXISTING_INDEX_CHANGES")
        current = _current_review_evidence(repository, expected_head, permitted, max_bytes, query)
        if current.fingerprint != evidence.fingerprint:
            raise _StageFailure("EVIDENCE_CHANGED")
        # Revalidate after evidence reads, immediately before the first add.
        preflight = check_publish_preflight(
            repository, expected_branch, expected_head, expected_origin_url, timeout_seconds,
        )
        if not preflight.passed:
            raise _StageFailure(preflight.error_code)
        if query(raw_arguments):
            raise _StageFailure("PREEXISTING_INDEX_CHANGES")
        for path in evidence.changed_paths:
            target = _safe_location(repository, path)
            try:
                mode = target.lstat().st_mode
            except FileNotFoundError:
                mode = None  # A reviewed deletion is allowed.
            if mode is not None and not stat.S_ISREG(mode):
                raise _StageFailure("UNSAFE_PATH")
            attempted = True  # A launch error or timeout may have an ambiguous outcome.
            output = query(["--literal-pathspecs", "add", "--all", "--", path])
            if output:
                raise _StageFailure("AMBIGUOUS_STAGING_OUTPUT")

        raw = _verify_staged_snapshot(evidence, expected_head, raw_arguments, query, run)
        final = _check_publish_state(
            repository, expected_branch, expected_head, expected_origin_url, timeout_seconds,
            allow_staged=True,
        )
        if not final.passed:
            raise _StageFailure(final.error_code)
        # Ensure the index did not change during the final state queries.
        if query(raw_arguments) != raw:
            raise _StageFailure("INDEX_CHANGED_DURING_VERIFICATION")
        return finish(verified_raw=raw)
    except _StageFailure as exc:
        return finish(str(exc))
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return finish("INVALID_OR_UNVERIFIED_INPUT")
    except OSError:
        return finish("GIT_IO_FAILED")


@dataclass(frozen=True)
class GitPublishResult:
    outcome: str
    error_code: str | None = None
    commit_attempted: bool = False
    push_attempted: bool = False
    # commit_sha is set only after parent/tree verification; observed_head_sha
    # can identify an unverified local commit that must not be pushed.
    commit_sha: str | None = None
    remote_sha: str | None = None
    observed_head_sha: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome == "PUBLISHED"


def publish_reviewed_changes(
    repository_directory: str | Path,
    expected_branch: str,
    expected_head: str,
    expected_origin_url: str,
    timeout_seconds: float,
    staged_result: GitStageResult,
    reviewed_packet: bytes,
    review_result: ReviewResult,
    max_bytes: int,
    commit_message: str,
) -> GitPublishResult:
    """Commit a reverified index and push its exact commit once.

    Outcomes: NOT_ATTEMPTED, COMMIT_FAILED_OR_AMBIGUOUS,
    COMMIT_CREATED_NOT_PUSHED, PUSH_FAILED_OR_AMBIGUOUS, PUBLISHED.
    Errors preserve state, including any commit/push that may already exist.
    Neither checks nor publication are atomic with other repository users.
    Approval objects and expected state are trusted adapter/caller inputs;
    frozen-plan provenance and model comprehension are not established here.
    """
    outcome = "NOT_ATTEMPTED"
    commit_attempted = push_attempted = False
    commit_sha = remote_sha = observed_head_sha = None

    def finish(error=None):
        return GitPublishResult(
            outcome, error, commit_attempted, push_attempted,
            commit_sha, remote_sha, observed_head_sha,
        )

    try:
        evidence = _review_snapshot(reviewed_packet, review_result, max_bytes)
        if (not isinstance(staged_result, GitStageResult) or staged_result.passed is not True
                or staged_result.error_code is not None
                or staged_result.staging_attempted is not True
                or staged_result.staged_state_error_code is not None
                or not evidence.changed_paths
                or staged_result.staged_paths != evidence.changed_paths
                or not isinstance(staged_result.staged_changes_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", staged_result.staged_changes_sha256)):
            raise _StageFailure("INVALID_STAGE_RESULT")
        if (not isinstance(commit_message, str) or not commit_message.strip()
                or len(commit_message) > 4096
                or any(ord(character) < 32 and character not in "\n\t"
                       or ord(character) == 127 for character in commit_message)
                or len(commit_message.encode("utf-8")) > 4096):
            raise _StageFailure("INVALID_COMMIT_MESSAGE")
        if isinstance(expected_branch, str) and expected_branch.casefold() == "main":
            raise _StageFailure("PROTECTED_BRANCH")
        state = _check_publish_state(
            repository_directory, expected_branch, expected_head, expected_origin_url,
            timeout_seconds, allow_staged=True,
        )
        if not state.passed:
            raise _StageFailure(state.error_code)
        repository = Path(repository_directory).resolve(strict=True)
        expected_head = expected_head.lower()
        temporary = _external_temporary_directory(repository)
        if temporary is None:
            raise _StageFailure("UNSAFE_TEMP_LOCATION")
        environment = os.environ.copy()
        if any(name in environment for name in _REPOSITORY_ENVIRONMENT):
            raise _StageFailure("UNSAFE_GIT_ENVIRONMENT")
        environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")

        def run(arguments):
            code, output, error = _git_query(
                repository, arguments, timeout_seconds, temporary, environment, max_bytes,
            )
            if error:
                raise _StageFailure(error)
            return code, output

        def query(arguments):
            code, output = run(arguments)
            if code != 0:
                raise _StageFailure("GIT_FAILED")
            return output

        def sha(arguments):
            value = _line(query(arguments))
            if not _SHA.fullmatch(value):
                raise _StageFailure("MALFORMED_GIT_OUTPUT")
            return value.lower()

        def clean():
            if query(["status", "--porcelain=v1", "-z", "--untracked-files=all"]):
                raise _StageFailure("UNEXPECTED_PENDING_CHANGES")

        def no_mirror():
            code, output = run(["config", "--type=bool", "--get-all", "remote.origin.mirror"])
            if code == 1 and not output:
                return  # No mirror setting: Git defaults to a normal push.
            if code != 0 or _line(output) != "false":
                raise _StageFailure("UNSAFE_PUSH_CONFIGURATION")

        raw_arguments = [
            "diff", "--cached", "--no-ext-diff", "--no-textconv", "--no-renames",
            "--raw", "--abbrev=40", "-z", "--ita-visible-in-index", expected_head, "--",
        ]
        raw = _verify_staged_snapshot(evidence, expected_head, raw_arguments, query, run)
        if hashlib.sha256(raw).hexdigest() != staged_result.staged_changes_sha256:
            raise _StageFailure("STAGE_BINDING_MISMATCH")
        no_mirror()
        # write-tree captures the exact index tree; it does not stage files or
        # update history. It runs only after all authorization checks succeed.
        tree = sha(["write-tree"])
        state = _check_publish_state(
            repository, expected_branch, expected_head, expected_origin_url,
            timeout_seconds, allow_staged=True,
        )
        if not state.passed:
            raise _StageFailure(state.error_code)
        if _verify_staged_snapshot(evidence, expected_head, raw_arguments, query, run) != raw:
            raise _StageFailure("INDEX_CHANGED_DURING_VERIFICATION")

        outcome = "COMMIT_FAILED_OR_AMBIGUOUS"
        commit_attempted = True
        query(["commit", "--cleanup=verbatim", "-m", commit_message])
        observed_head_sha = sha(["rev-parse", "--verify", "HEAD^{commit}"])
        if observed_head_sha == expected_head:
            raise _StageFailure("COMMIT_NOT_CREATED")
        outcome = "COMMIT_CREATED_NOT_PUSHED"
        # Read the physical commit object, without replacement refs changing
        # the parent or tree presented by revision-walking commands.
        commit_object = query(["--no-replace-objects", "cat-file", "commit", observed_head_sha])
        headers, separator, _message = commit_object.partition(b"\n\n")
        if not separator:
            raise _StageFailure("MALFORMED_COMMIT_OBJECT")
        header_lines = headers.split(b"\n")
        parents = [line for line in header_lines if line.startswith(b"parent ")]
        if parents != [b"parent " + expected_head.encode("ascii")]:
            raise _StageFailure("COMMIT_PARENT_MISMATCH")
        if header_lines[0] != b"tree " + tree.encode("ascii"):
            raise _StageFailure("COMMIT_TREE_MISMATCH")
        commit_sha = observed_head_sha
        # The local HEAD is new, but the remote must still be the original
        # expected SHA. Recheck root, branch, URLs, locks and index as well.
        state = _check_publish_state(
            repository, expected_branch, commit_sha, expected_origin_url, timeout_seconds,
            expected_remote_head=expected_head,
        )
        if not state.passed:
            raise _StageFailure(state.error_code)
        clean()
        no_mirror()
        if sha(["rev-parse", "--verify", "HEAD^{commit}"]) != commit_sha:
            raise _StageFailure("HEAD_CHANGED_DURING_VERIFICATION")

        outcome = "PUSH_FAILED_OR_AMBIGUOUS"
        push_attempted = True
        query([
            "push", "--porcelain", "--no-force", "--no-mirror", "--no-follow-tags",
            "--recurse-submodules=no", "origin", f"{commit_sha}:refs/heads/{expected_branch}",
        ])
        remote = _line(query(
            ["ls-remote", "--exit-code", "--refs", "origin", f"refs/heads/{expected_branch}"],
        )).split("\t")
        if (len(remote) != 2 or not _SHA.fullmatch(remote[0])
                or remote[1] != f"refs/heads/{expected_branch}"):
            raise _StageFailure("MALFORMED_REMOTE_OUTPUT")
        remote_sha = remote[0].lower()
        if remote_sha != commit_sha:
            raise _StageFailure("REMOTE_VERIFICATION_MISMATCH")
        state = _check_publish_state(
            repository, expected_branch, commit_sha, expected_origin_url, timeout_seconds,
        )
        if not state.passed:
            raise _StageFailure(state.error_code)
        clean()
        outcome = "PUBLISHED"
        return finish()
    except _StageFailure as exc:
        return finish(str(exc))
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return finish("INVALID_OR_UNVERIFIED_INPUT")
    except OSError:
        return finish("GIT_IO_FAILED")
