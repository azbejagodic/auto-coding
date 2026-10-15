"""Command-line entry point for Auto-Coding v1."""

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from urllib.parse import urlsplit

from auto_coding.plan_freeze import PlanIntegrityError, freeze_approved_plan, plan_fingerprint
from auto_coding.plan_preflight import preflight_frozen_plan
from auto_coding.plans import PlanValidationError, validate_plan


_EXAMPLES = r'''PowerShell examples (also available through the installed auto-coding command):
  py -m auto_coding plan validate "C:\Auto Coding\proposed.json"
  py -m auto_coding plan freeze "C:\Auto Coding\proposed.json" --approved-sha256 $approvedSha --destination "C:\Auto Coding\frozen.json" --repository "C:\Projects\Target App"
  py -m auto_coding plan preflight "C:\Auto Coding\frozen.json" --approved-sha256 $approvedSha --repository "C:\Projects\Target App" --git-timeout-seconds 10

Review the COMPLETE plan before supplying its confirmed fingerprint to freeze.
Validation is not approval. An explicit fingerprint argument is a caller
assertion; matching it does not independently authenticate a human.
These commands do not execute a batch. Preflight is a snapshot, not a lock.
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
        prog="auto-coding", description="Auto-Coding v1: plan inspection, explicit freezing and read-only preflight",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        return 0  # Preserve the original no-argument entry-point behavior.
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
