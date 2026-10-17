"""Command-line entry point for Auto-Coding v1."""

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys
from urllib.parse import urlsplit

from auto_coding.plan_freeze import PlanIntegrityError, freeze_approved_plan, load_frozen_plan, plan_fingerprint
from auto_coding.plan_preflight import preflight_frozen_plan
from auto_coding.plans import PlanValidationError, validate_plan
from auto_coding.orchestrator import (
    FirstTaskReviewResult, SingleTaskPublishResult, execute_first_task_through_review,
    export_stopped_batch_report, publish_single_task_batch, execute_sequential_batch, SequentialBatchResult,
)


_EXAMPLES = r'''PowerShell examples (also available through the installed auto-coding command):
  py -m auto_coding plan validate "C:\Auto Coding\proposed.json"
  py -m auto_coding plan freeze "C:\Auto Coding\proposed.json" --approved-sha256 $approvedSha --destination "C:\Auto Coding\frozen.json" --repository "C:\Projects\Target App"
  py -m auto_coding plan preflight "C:\Auto Coding\frozen.json" --approved-sha256 $approvedSha --repository "C:\Projects\Target App" --git-timeout-seconds 10
  py -m auto_coding run "C:\Metadata\frozen.json" --approved-sha256 $approvedSha --repository "C:\Projects\DisposableTarget" --metadata-directory "C:\Metadata\batch-001" --repository-id "example/disposable-target" --git-timeout-seconds 10 --evidence-max-bytes 1048576 --packet-max-bytes 2097152 --report-max-bytes 2097152

Review the COMPLETE plan before supplying its confirmed fingerprint to freeze.
Validation is not approval. An explicit fingerprint argument is a caller
assertion; matching it does not independently authenticate a human.
Plan commands do not execute a batch. Run executes and publishes approved tasks
sequentially. Preflight is a snapshot, not a lock. Run never resumes or retries.
'''


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's ordinary errors can echo uncontrolled argument values.
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ARGUMENT_ERROR (use --help)\n")


class _CliFailure(Exception):
    pass


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("positive finite seconds required") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("positive finite seconds required")
    return timeout


def _run_timeout(value: str) -> float:
    timeout = _positive_timeout(value)
    if timeout > 86400:
        raise argparse.ArgumentTypeError("timeout exceeds 86400 seconds")
    return timeout


def _run_bytes(value: str) -> int:
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("integer byte limit required") from None
    if not 0 < limit <= 64 * 1024 * 1024:
        raise argparse.ArgumentTypeError("byte limit must be 1..67108864")
    return limit


def _approval(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("full lowercase SHA-256 required")
    return value


def _repository_id(value: str) -> str:
    if len(value) > 256 or not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", value):
        raise argparse.ArgumentTypeError("owner/repository identifier required")
    return value


def _safe_code(value, fallback):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", value) else fallback


def _metadata_directory(value: str, repository: Path) -> Path:
    """Reject symlink/junction/reparse components and resolved path aliases.

This check observes the filesystem; it does not lock it against other writers.
"""
    requested = Path(value).absolute()
    if ".." in requested.parts:
        raise _CliFailure("UNSAFE_METADATA_DIRECTORY")
    for component in (*reversed(requested.parents), requested):
        info = component.lstat()
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise _CliFailure("UNSAFE_METADATA_DIRECTORY")
    resolved = requested.resolve(strict=True)
    if resolved != requested or not resolved.is_dir() or resolved.is_relative_to(repository):
        raise _CliFailure("UNSAFE_METADATA_DIRECTORY")
    return resolved


def _show_paths(checkpoint: Path, metadata: Path, result=None) -> None:
    # JSON quoting prevents control characters in caller-supplied paths from
    # becoming terminal instructions. Only validated fixed destinations display.
    print("Checkpoint: " + json.dumps(str(checkpoint), ensure_ascii=True))
    if result is not None:
        paths = (*result.report_paths, *result.partial_report_paths)
        for name in ("batch-report.json", "batch-report.md"):
            path = metadata / name
            if path in paths:
                print("Report: " + json.dumps(str(path), ensure_ascii=True))


def _run(args) -> int:
    phase = "preflight"
    checkpoint = metadata = None
    try:
        repository = Path(args.repository).resolve(strict=True)
        frozen = _external_artifact(args.frozen_plan_path, args.repository)
        if Path(args.frozen_plan_path).absolute().is_relative_to(repository):
            raise _CliFailure("UNSAFE_FROZEN_PLAN_LOCATION")
        metadata = _metadata_directory(args.metadata_directory, repository)
        checkpoint = metadata / "batch-checkpoint.json"
        for path in (checkpoint, metadata / "batch-report.json", metadata / "batch-report.md"):
            if os.path.lexists(path):
                raise _CliFailure("METADATA_DESTINATION_EXISTS")
        print("PREFLIGHT")
        preflight = preflight_frozen_plan(frozen, args.approved_sha256, repository, args.git_timeout_seconds)
        if preflight.ready is not True or preflight.error_code is not None:
            raise _CliFailure(_safe_code(preflight.error_code, "PREFLIGHT_FAILED"))
        plan = load_frozen_plan(frozen, args.approved_sha256)
        if plan["schema_version"] != 2:
            raise _CliFailure("PLAN_SCHEMA_NOT_RUNNABLE")
        if len(plan["tasks"]) > 1:
            phase = "execution"
            print("SEQUENTIAL IMPLEMENTATION / TESTS / REVIEW / PUBLICATION")
            result = execute_sequential_batch(
                frozen, args.approved_sha256, repository, metadata, args.repository_id,
                args.git_timeout_seconds, args.evidence_max_bytes, args.packet_max_bytes, args.report_max_bytes)
            if not isinstance(result, SequentialBatchResult):
                raise _CliFailure("BATCH_RESULT_UNCERTAIN")
            complete = (result.outcome == "COMPLETED" and result.error_code is None
                        and result.checkpoint_error_code is None and result.report_error_code is None
                        and result.completed_task_ids == tuple(task["task_id"] for task in plan["tasks"])
                        and tuple(item[0] for item in result.publications) == result.completed_task_ids
                        and all(re.fullmatch(r"[0-9a-fA-F]{40}", item[1]) and item[1] == item[2] for item in result.publications)
                        and result.report_paths == (metadata / "batch-report.json", metadata / "batch-report.md"))
            if complete:
                print("COMPLETED")
            else:
                status = ("PUBLISHED_NOT_FINALIZED" if result.outcome == "PUBLISHED_NOT_FINALIZED"
                          else "PARTIALLY_COMPLETED" if result.publications else "STOPPED")
                safe_stage = result.stage if result.stage in {"preflight", "implementation", "evidence", "testing", "review", "staging", "publishing", "checkpoint", "reporting"} else "execution"
                print(f"{status} stage={safe_stage} reason={_safe_code(result.error_code, 'BATCH_UNVERIFIED')}", file=sys.stderr)
            print("Batch: " + json.dumps(plan["batch_id"]))
            print(f"Completed tasks: {len(result.completed_task_ids)}/{len(plan['tasks'])}")
            for task_id, commit, remote in result.publications:
                if re.fullmatch(r"[0-9a-fA-F]{40}", commit) and re.fullmatch(r"[0-9a-fA-F]{40}", remote):
                    print("Task: " + json.dumps(task_id) + f"; Commit SHA: {commit}; Remote SHA: {remote}")
            for label, code in (("Checkpoint", result.checkpoint_error_code), ("Reporting", result.report_error_code)):
                if code is not None:
                    print(label + ": " + _safe_code(code, "FINALIZATION_FAILED"), file=sys.stderr)
            _show_paths(checkpoint, metadata, result)
            return 0 if complete else 1
        phase = "implementation"
        print("IMPLEMENTATION / TESTS / REVIEW")
        handoff = execute_first_task_through_review(
            frozen, args.approved_sha256, repository, checkpoint, args.git_timeout_seconds,
            args.evidence_max_bytes, args.packet_max_bytes,
        )
        if (not isinstance(handoff, FirstTaskReviewResult) or handoff.outcome != "READY_FOR_PUBLICATION"
                or handoff.error_code is not None or handoff.checkpoint_error_code is not None):
            stage = getattr(handoff, "stage", None)
            stage = stage if stage in ("preflight", "implementation", "evidence", "testing", "review", "publishing") else phase
            reason = _safe_code(getattr(handoff, "error_code", None), "REVIEW_HANDOFF_INVALID")
            print(f"STOPPED stage={stage} reason={reason}", file=sys.stderr)
            checkpoint_error = getattr(handoff, "checkpoint_error_code", None)
            if checkpoint_error is not None:
                print("Checkpoint: " + _safe_code(checkpoint_error, "CHECKPOINT_FAILED"), file=sys.stderr)
            try:
                report = export_stopped_batch_report(handoff, frozen, args.approved_sha256, metadata,
                                                    args.repository_id, args.report_max_bytes)
            except (Exception, KeyboardInterrupt):
                print("Reporting: REPORT_UNAVAILABLE", file=sys.stderr)
                _show_paths(checkpoint, metadata)
                return 1
            if report.error_code is not None:
                print("Reporting: " + _safe_code(report.error_code, "REPORT_FAILED"), file=sys.stderr)
            _show_paths(checkpoint, metadata, report)
            return 1
        phase = "publishing"
        print("PUBLICATION / FINALIZATION")
        result = publish_single_task_batch(
            handoff, frozen, args.approved_sha256, args.git_timeout_seconds,
            args.evidence_max_bytes, args.packet_max_bytes, args.report_max_bytes, metadata, args.repository_id,
        )
        if not isinstance(result, SingleTaskPublishResult):
            raise _CliFailure("PUBLICATION_RESULT_UNCERTAIN")
        complete = (
            result.outcome == "PUBLISHED_FINALIZED" and result.error_code is None
            and result.checkpoint_error_code is None and result.report_error_code is None
            and result.published is True and result.checkpoint_finalized is True and result.reports_exported is True
            and result.staging_invoked is True and result.publishing_invoked is True
            and isinstance(result.commit_sha, str) and re.fullmatch(r"[0-9a-fA-F]{40}", result.commit_sha)
            and result.remote_sha == result.commit_sha
            and result.report_paths == (metadata / "batch-report.json", metadata / "batch-report.md")
        )
        if complete:
            print("COMPLETED")
            print("Batch: " + json.dumps(plan["batch_id"]))
            print("Task: " + json.dumps(plan["tasks"][0]["task_id"]))
        else:
            status = "PUBLISHED_BUT_NOT_FINALIZED" if result.published is True else "STOPPED"
            stage = result.stage if result.stage in ("validation", "staging", "publishing", "checkpoint", "reporting") else phase
            print(f"{status} stage={stage} reason={_safe_code(result.error_code, 'FINALIZATION_UNVERIFIED')}", file=sys.stderr)
            print("Outcome: " + _safe_code(result.outcome, "UNVERIFIED"))
            if result.publishing is not None:
                print("Git outcome: " + _safe_code(result.publishing.outcome, "UNVERIFIED"))
            for label, code in (("Checkpoint", result.checkpoint_error_code), ("Reporting", result.report_error_code)):
                if code is not None:
                    print(label + ": " + _safe_code(code, "FINALIZATION_FAILED"), file=sys.stderr)
        for label, sha in (("Commit SHA", result.commit_sha), ("Remote SHA", result.remote_sha)):
            if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                print(f"{label}: {sha}")
        _show_paths(checkpoint, metadata, result)
        return 0 if complete else 1
    except _CliFailure as error:
        reason = _safe_code(str(error), "RUN_FAILED")
    except (PlanIntegrityError, PlanValidationError):
        reason = "PLAN_INVALID_OR_UNAPPROVED"
    except OSError:
        reason = "IO_FAILED"
    except KeyboardInterrupt:
        reason = "INTERRUPTED_STATE_UNCERTAIN"
    except Exception:
        reason = "RUN_STATE_UNCERTAIN"
    print(f"STOPPED stage={phase} reason={reason}", file=sys.stderr)
    if checkpoint is not None and metadata is not None:
        _show_paths(checkpoint, metadata)
    return 1


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _CliFailure("PLAN_JSON_INVALID")
        result[key] = value
    return result


def _load_proposed_plan(path: str) -> dict:
    try:
        # UTF-8 BOMs are common in Windows-generated text files.
        with Path(path).open("r", encoding="utf-8-sig") as source:
            data = json.load(source, object_pairs_hook=_unique_keys)
    except (json.JSONDecodeError, UnicodeError, RecursionError):
        raise _CliFailure("PLAN_JSON_INVALID") from None
    return validate_plan(data)


def _external_artifact(artifact: str, repository: str) -> Path:
    root = Path(repository).resolve(strict=True)
    # Require an explicit working-tree root, including .git files used by Git
    # for linked repositories. A subdirectory cannot stand in for the root.
    if not root.is_dir() or not (root / ".git").exists():
        raise _CliFailure("TARGET_REPOSITORY_ROOT_REQUIRED")
    destination = Path(artifact).resolve()
    if destination == root or root in destination.parents:
        raise _CliFailure("UNSAFE_FROZEN_PLAN_LOCATION")
    return destination


def _remote_identifier(origin: str) -> str:
    if PureWindowsPath(origin).is_absolute() or PurePosixPath(origin).is_absolute():
        return "local filesystem remote (path omitted)"
    if "://" in origin:
        remote = urlsplit(origin)
        if remote.scheme == "file":
            return "local filesystem remote (path omitted)"
        return json.dumps(remote.hostname, ensure_ascii=True)
    return json.dumps(origin.split("@", 1)[1].split(":", 1)[0], ensure_ascii=True)


def _summary(plan: dict, fingerprint: str) -> None:
    print("VALID (validation is not approval)")
    print(f"Plan SHA-256: {fingerprint}")
    print(f"Schema version: {plan['schema_version']}")
    print("Batch ID: " + json.dumps(plan["batch_id"], ensure_ascii=True))
    print("Branch: " + json.dumps(plan["expected_branch"], ensure_ascii=True))
    print(f"Starting HEAD: {plan['expected_head']}")
    print(f"Task count: {len(plan['tasks'])}")
    if plan["schema_version"] == 2:
        print("Remote identifier: " + _remote_identifier(plan["expected_origin_url"]))
    for task in plan["tasks"]:
        suffix = (f"; test timeout per command: {task['limits']['test_seconds']} seconds"
                  if plan["schema_version"] == 2 else "; not execution-ready (schema v1)")
        print("Task: " + json.dumps(task["task_id"], ensure_ascii=True) + suffix)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="auto-coding", description="Auto-Coding v1: approved plans and sequential task execution",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EXAMPLES,
    )
    commands = parser.add_subparsers(dest="command")
    plan = commands.add_parser("plan", help="inspect, explicitly freeze or preflight a plan",
                               epilog=_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = plan.add_subparsers(dest="plan_action", required=True)
    validate = actions.add_parser("validate", help="validate and fingerprint; no approval or writes")
    validate.add_argument("plan_path", help="proposed UTF-8 JSON plan path (quote paths with spaces)")
    freeze = actions.add_parser(
        "freeze", help="freeze schema v2 after reviewing the complete plan",
        description="Review the complete plan first. The explicit SHA is caller confirmation, not independently authenticated human approval.",
    )
    freeze.add_argument("plan_path", help="proposed UTF-8 JSON plan path")
    freeze.add_argument("--approved-sha256", required=True, help="exact full lowercase SHA-256 independently confirmed by the caller")
    freeze.add_argument("--destination", required=True, help="new frozen-plan path outside the target repository; parent must exist")
    freeze.add_argument("--repository", required=True, help="target Git working-tree root, used to enforce external artifact storage")
    preflight = actions.add_parser("preflight", help="read-only readiness check; does not execute tasks")
    preflight.add_argument("frozen_plan_path", help="frozen-plan path outside the target repository")
    preflight.add_argument("--approved-sha256", required=True, help="explicit approved fingerprint from an independent trusted source")
    preflight.add_argument("--repository", required=True, help="target Git working-tree root")
    preflight.add_argument("--git-timeout-seconds", required=True, type=_positive_timeout, help="positive finite timeout for each read-only Git command")
    run = commands.add_parser("run", help="execute, review and publish approved frozen tasks in order")
    run.add_argument("frozen_plan_path", help="approved schema-v2 frozen artifact outside the target repository")
    run.add_argument("--approved-sha256", required=True, type=_approval, help="explicit trusted approval fingerprint; never inferred")
    run.add_argument("--repository", required=True, help="target Git working-tree root")
    run.add_argument("--metadata-directory", required=True, help="existing external directory; no checkpoint or batch reports may exist")
    run.add_argument("--repository-id", required=True, type=_repository_id, help="owner/repository report label")
    run.add_argument("--git-timeout-seconds", required=True, type=_run_timeout, help="per-command timeout, positive and at most 86400 seconds")
    for name in ("evidence", "packet", "report"):
        run.add_argument(f"--{name}-max-bytes", required=True, type=_run_bytes,
                         help="explicit byte limit, 1..67108864 (64 MiB)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        return 0  # Preserve the original no-argument entry-point behavior.
    if args.command == "run":
        return _run(args)
    try:
        if args.plan_action == "validate":
            plan = _load_proposed_plan(args.plan_path)
            _summary(plan, plan_fingerprint(plan))
        elif args.plan_action == "freeze":
            plan = _load_proposed_plan(args.plan_path)
            if plan["schema_version"] != 2:
                raise _CliFailure("PLAN_SCHEMA_NOT_RUNNABLE")
            fingerprint = plan_fingerprint(plan)
            if args.approved_sha256 != fingerprint:
                raise _CliFailure("APPROVAL_REJECTED")
            destination = _external_artifact(args.destination, args.repository)
            if os.path.lexists(destination):
                raise _CliFailure("FROZEN_PLAN_EXISTS")
            freeze_approved_plan(plan, args.approved_sha256, destination)
            print("FROZEN (explicit caller confirmation; human identity not authenticated)")
            print(f"Plan SHA-256: {fingerprint}")
        else:
            _external_artifact(args.frozen_plan_path, args.repository)
            result = preflight_frozen_plan(args.frozen_plan_path, args.approved_sha256,
                                           args.repository, args.git_timeout_seconds)
            if not result.ready:
                code = result.error_code
                if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code):
                    code = "PREFLIGHT_FAILED"
                print(f"NOT READY: {code}", file=sys.stderr)
                return 1
            print("READY (read-only snapshot; no tasks executed)")
            print("Batch ID: " + json.dumps(result.batch_id, ensure_ascii=True))
            print("Branch: " + json.dumps(result.branch, ensure_ascii=True))
            print(f"Starting HEAD: {result.starting_head}")
            print(f"Plan SHA-256: {result.plan_sha256}")
            for task_id, seconds in result.test_timeouts:
                print("Task: " + json.dumps(task_id, ensure_ascii=True) + f"; test timeout per command: {seconds} seconds")
        return 0
    except _CliFailure as error:
        code = str(error)
    except PlanValidationError:
        code = "PLAN_INVALID"
    except PlanIntegrityError:
        code = "FROZEN_PLAN_REJECTED"
    except OSError:
        code = "IO_FAILED"
    prefix = "NOT READY: " if args.plan_action == "preflight" else "ERROR: "
    print(prefix + code, file=sys.stderr)
    return 1
