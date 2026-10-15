"""Tests for the Auto-Coding command-line interface."""

import os
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.cli import main
from auto_coding.plan_freeze import load_frozen_plan, plan_fingerprint
from auto_coding.plan_preflight import PlanPreflightResult
from test_plans import make_plan, make_v2_plan


class CliTests(unittest.TestCase):
    def test_module_help(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(project_root / "src"), env.get("PYTHONPATH")) if part
        )

        result = subprocess.run(
            [sys.executable, "-m", "auto_coding", "--help"],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Auto-Coding v1", result.stdout)
        self.assertIn("usage:", result.stdout)


class PlanCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="Auto Coding CLI ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.repository = self.directory / "Target App"
        (self.repository / ".git").mkdir(parents=True)
        self.proposed = self.directory / "Proposed Plan.json"
        self.destination = self.directory / "Frozen Plan.json"
        self.plan = make_v2_plan("https://example.invalid/private-remote-path.git")
        self.plan["tasks"][0]["implementation"]["prompt"] = "PRIVATE_IMPLEMENTATION_PROMPT"
        self.plan["tasks"][0]["review"]["prompt"] = "PRIVATE_REVIEW_PROMPT"
        self.plan["tasks"][0]["test_commands"] = [["NEVER_EXECUTE", "PRIVATE_COMMAND_ARGUMENT"]]
        self.write_plan()
        self.approval = plan_fingerprint(self.plan)

    def write_plan(self, encoding="utf-8"):
        self.proposed.write_text(json.dumps(self.plan), encoding=encoding)

    def module(self, *arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(mode="w+b") as stderr:
            process = subprocess.run(
                [sys.executable, "-m", "auto_coding", *map(str, arguments)],
                cwd=self.directory, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False, check=False, timeout=10,
            )
            stdout.seek(0)
            stderr.seek(0)
            output, errors = stdout.read(65537), stderr.read(65537)
            self.assertLessEqual(len(output), 65536)
            self.assertLessEqual(len(errors), 65536)
        return process.returncode, output.decode("utf-8"), errors.decode("utf-8")

    def freeze_args(self, approval=None, destination=None, repository=None):
        return ["plan", "freeze", str(self.proposed), "--approved-sha256", self.approval if approval is None else approval,
                "--destination", str(self.destination if destination is None else destination),
                "--repository", str(self.repository if repository is None else repository)]

    def direct(self, args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = main(args)
        return result, output.getvalue(), errors.getvalue()

    @staticmethod
    def snapshot(directory):
        return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}

    def assert_private(self, output):
        for secret in ("PRIVATE_IMPLEMENTATION_PROMPT", "PRIVATE_REVIEW_PROMPT", "PRIVATE_COMMAND_ARGUMENT",
                       "NEVER_EXECUTE", "private-remote-path", "Traceback"):
            self.assertNotIn(secret, output)

    def test_help_lists_supported_commands_and_powershell_examples(self):
        for arguments in (("--help",), ("plan", "--help"), ("plan", "freeze", "--help"),
                          ("plan", "validate", "--help"), ("plan", "preflight", "--help")):
            with self.subTest(arguments=arguments):
                status, output, errors = self.module(*arguments)
                self.assertEqual(status, 0, errors)
                self.assertIn("usage:", output)
        status, output, _ = self.module("--help")
        self.assertIn("py -m auto_coding plan validate", output)
        self.assertIn("$approvedSha", output)
        self.assertIn("COMPLETE plan", output)
        self.assertEqual(self.module()[0], 0)

    def test_validate_prints_canonical_identity_and_safe_timeouts_without_writes(self):
        before = self.snapshot(self.directory)
        status, output, errors = self.module("plan", "validate", self.proposed)
        self.assertEqual(status, 0, errors)
        self.assertIn(self.approval, output)
        self.assertIn("Schema version: 2", output)
        self.assertIn("batch-1", output)
        self.assertIn("feature/002-v1-clean", output)
        self.assertIn("Task count: 1", output)
        self.assertIn("task-1", output)
        self.assertIn("120 seconds", output)
        self.assertIn("example.invalid", output)
        self.assertIn("validation is not approval", output)
        self.assert_private(output + errors)
        self.assertEqual(self.snapshot(self.directory), before)
        self.assertFalse(self.destination.exists())

    def test_validate_accepts_utf8_bom_and_paths_with_spaces(self):
        self.write_plan(encoding="utf-8-sig")
        status, output, errors = self.module("plan", "validate", self.proposed)
        self.assertEqual(status, 0, errors)
        self.assertIn(self.approval, output)

    def test_invalid_json_and_duplicate_keys_are_sanitized(self):
        for raw in (b"{PRIVATE", b"\xffPRIVATE", b'{"PRIVATE":1,"PRIVATE":2}'):
            self.proposed.write_bytes(raw)
            status, output, errors = self.module("plan", "validate", self.proposed)
            self.assertEqual(status, 1)
            self.assertIn("PLAN_JSON_INVALID", errors)
            self.assertNotIn("PRIVATE", output + errors)
            self.assertNotIn("Traceback", errors)
            self.assertEqual(self.proposed.read_bytes(), raw)

    def test_invalid_schema_diagnostics_do_not_echo_private_field_names(self):
        self.plan["PRIVATE_UNKNOWN_FIELD"] = "PRIVATE_CONTENT"
        self.write_plan()
        status, output, errors = self.module("plan", "validate", self.proposed)
        self.assertEqual(status, 1)
        self.assertIn("PLAN_INVALID", errors)
        self.assertNotIn("PRIVATE", output + errors)

    def test_freeze_requires_explicit_matching_full_fingerprint(self):
        for approval in ("", "bad", "0" * 64, self.approval.upper()):
            with self.subTest(approval=approval):
                status, output, errors = self.module(*self.freeze_args(approval=approval))
                self.assertEqual(status, 1)
                self.assertIn("APPROVAL_REJECTED", errors)
                self.assertFalse(self.destination.exists())
                self.assert_private(output + errors)

    def test_missing_approval_is_not_read_from_environment_or_yes_flag(self):
        arguments = self.freeze_args()
        del arguments[3:5]
        with patch.dict(os.environ, {"APPROVED_SHA256": self.approval}):
            self.assertEqual(self.module(*arguments)[0], 2)
        status, output, errors = self.module(*self.freeze_args(), "--yes", "PRIVATE_ARGUMENT")
        self.assertEqual(status, 2)
        self.assertNotIn("PRIVATE_ARGUMENT", output + errors)
        self.assertFalse(self.destination.exists())

    def test_freeze_reuses_library_and_passes_exact_approval(self):
        with patch("auto_coding.cli.freeze_approved_plan") as freeze:
            status, output, errors = self.direct(self.freeze_args())
        self.assertEqual(status, 0, errors)
        freeze.assert_called_once_with(self.plan, self.approval, self.destination)
        self.assertIn("human identity not authenticated", output)

    def test_freeze_creates_loadable_write_once_artifact(self):
        before = self.snapshot(self.repository)
        original = self.proposed.read_bytes()
        status, output, errors = self.module(*self.freeze_args())
        self.assertEqual(status, 0, errors)
        self.assertIn("FROZEN", output)
        self.assertEqual(load_frozen_plan(self.destination, self.approval), self.plan)
        frozen = self.destination.read_bytes()
        status, _, errors = self.module(*self.freeze_args())
        self.assertEqual(status, 1)
        self.assertIn("FROZEN_PLAN_EXISTS", errors)
        self.assertEqual(self.destination.read_bytes(), frozen)
        status, _, errors = self.module(*self.freeze_args(approval="0" * 64))
        self.assertEqual(status, 1)
        self.assertEqual(self.destination.read_bytes(), frozen)
        self.assertEqual(self.proposed.read_bytes(), original)
        self.assertEqual(self.snapshot(self.repository), before)

    def test_v1_can_be_inspected_but_not_frozen_as_runnable(self):
        self.plan = make_plan()
        self.write_plan()
        self.approval = plan_fingerprint(self.plan)
        status, output, errors = self.module("plan", "validate", self.proposed)
        self.assertEqual(status, 0, errors)
        self.assertIn("not execution-ready", output)
        status, _, errors = self.module(*self.freeze_args())
        self.assertEqual(status, 1)
        self.assertIn("PLAN_SCHEMA_NOT_RUNNABLE", errors)
        self.assertFalse(self.destination.exists())

    def test_freeze_rejects_inside_repository_and_resolved_aliases(self):
        for destination in (self.repository / "frozen.json", self.directory / "Target App" / ".." / "Target App" / "frozen.json"):
            status, _, errors = self.module(*self.freeze_args(destination=destination))
            self.assertEqual(status, 1)
            self.assertIn("UNSAFE_FROZEN_PLAN_LOCATION", errors)
            self.assertFalse(destination.exists())

    def test_freeze_requires_root_information_and_existing_parent(self):
        child = self.repository / "child"
        child.mkdir()
        status, _, errors = self.module(*self.freeze_args(repository=child))
        self.assertEqual(status, 1)
        self.assertIn("TARGET_REPOSITORY_ROOT_REQUIRED", errors)
        self.assertEqual(self.module(*self.freeze_args(destination=self.directory / "missing" / "frozen.json"))[0], 1)
        args = self.freeze_args()
        del args[-2:]
        self.assertEqual(self.module(*args)[0], 2)

    def test_preflight_calls_existing_gate_with_explicit_inputs(self):
        ready = PlanPreflightResult(True, batch_id="batch-1", plan_sha256=self.approval,
                                    branch="feature/example", starting_head="a" * 40,
                                    repository_directory=self.repository, origin_url=self.plan["expected_origin_url"],
                                    test_timeouts=(("task-1", 120), ("task-2", 35)))
        arguments = ["plan", "preflight", str(self.destination), "--approved-sha256", self.approval,
                     "--repository", str(self.repository), "--git-timeout-seconds", "5"]
        with patch("auto_coding.cli.preflight_frozen_plan", return_value=ready) as preflight:
            status, output, errors = self.direct(arguments)
        preflight.assert_called_once_with(str(self.destination), self.approval, str(self.repository), 5.0)
        self.assertEqual(status, 0, errors)
        self.assertTrue(output.startswith("READY"))
        self.assertLess(output.index("task-1"), output.index("task-2"))
        self.assertIn("35 seconds", output)

    def test_preflight_failed_or_uncontrolled_reason_never_prints_ready(self):
        arguments = ["plan", "preflight", str(self.destination), "--approved-sha256", self.approval,
                     "--repository", str(self.repository), "--git-timeout-seconds", "5"]
        for reason in ("GIT_TIMEOUT", "PRIVATE secret\n", None):
            with patch("auto_coding.cli.preflight_frozen_plan", return_value=PlanPreflightResult(False, reason)):
                status, output, errors = self.direct(arguments)
            self.assertEqual(status, 1)
            self.assertEqual(output, "")
            self.assertIn("NOT READY", errors)
            self.assertNotIn("secret", errors)

    def test_preflight_requires_explicit_positive_timeout_and_sha(self):
        arguments = ["plan", "preflight", str(self.destination), "--approved-sha256", self.approval,
                     "--repository", str(self.repository)]
        self.assertEqual(self.module(*arguments)[0], 2)
        for timeout in ("0", "-1", "nan", "inf", "PRIVATE_TIMEOUT"):
            status, output, errors = self.module(*arguments, "--git-timeout-seconds", timeout)
            self.assertEqual(status, 2)
            self.assertNotIn("PRIVATE_TIMEOUT", output + errors)
        del arguments[3:5]
        self.assertEqual(self.module(*arguments, "--git-timeout-seconds", "5")[0], 2)

    def test_preflight_cannot_use_an_internal_artifact(self):
        internal = self.repository / "frozen.json"
        status, _, errors = self.module("plan", "preflight", internal, "--approved-sha256", self.approval,
                                       "--repository", self.repository, "--git-timeout-seconds", "5")
        self.assertEqual(status, 1)
        self.assertIn("UNSAFE_FROZEN_PLAN_LOCATION", errors)
        self.assertIn("NOT READY", errors)

    def test_io_failures_are_sanitized_and_defects_do_not_become_success(self):
        status, output, errors = self.module("plan", "validate", self.directory / "PRIVATE_FILE.json")
        self.assertEqual(status, 1)
        self.assertIn("IO_FAILED", errors)
        self.assertNotIn("PRIVATE_FILE", output + errors)
        with patch("auto_coding.cli.validate_plan", side_effect=RuntimeError("programming defect")):
            with self.assertRaises(RuntimeError):
                self.direct(["plan", "validate", str(self.proposed)])

    def test_no_run_or_other_execution_command_is_available(self):
        for arguments in (("run",), ("plan", "run"), ("plan",), ("plan", "validate", "--unknown", "PRIVATE")):
            status, output, errors = self.module(*arguments)
            self.assertEqual(status, 2)
            self.assertNotIn("PRIVATE", output + errors)


class PlanCliRepositoryTests(unittest.TestCase):
    module = PlanCliTests.module
    write_plan = PlanCliTests.write_plan
    freeze_args = PlanCliTests.freeze_args
    snapshot = staticmethod(PlanCliTests.snapshot)
    assert_private = PlanCliTests.assert_private

    def setUp(self):
        PlanCliTests.setUp(self)
        (self.repository / ".git").rmdir()
        self.remote = self.directory / "Bare Remote.git"
        self.git("init", "-q")
        self.git("config", "user.name", "CLI Tests")
        self.git("config", "user.email", "cli@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "core.autocrlf", "false")
        (self.repository / "tracked.txt").write_bytes(b"original\n")
        self.git("add", "--", "tracked.txt")
        self.git("commit", "-qm", "initial")
        self.branch = "feature/cli"
        self.git("branch", "-M", self.branch)
        self.git("init", "--bare", "-q", str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "-q", "origin", self.branch)
        self.head = self.git("rev-parse", "HEAD").decode().strip()
        self.plan.update(expected_branch=self.branch, expected_head=self.head, expected_origin_url=str(self.remote))
        self.write_plan()
        self.approval = plan_fingerprint(self.plan)
        self.assertEqual(self.module(*self.freeze_args())[0], 0)

    def git(self, *arguments):
        with tempfile.TemporaryFile(mode="w+b") as output:
            subprocess.run(["git", *arguments], cwd=self.repository, stdin=subprocess.DEVNULL,
                           stdout=output, stderr=subprocess.DEVNULL, shell=False, check=True, timeout=10)
            output.seek(0)
            raw = output.read(4097)
            self.assertLessEqual(len(raw), 4096)
            return raw

    def preflight(self, approval=None):
        return self.module("plan", "preflight", self.destination, "--approved-sha256", self.approval if approval is None else approval,
                           "--repository", self.repository, "--git-timeout-seconds", "5")

    def test_real_preflight_ready_without_target_or_frozen_artifact_changes(self):
        before = self.snapshot(self.directory)
        status, output, errors = self.preflight()
        self.assertEqual(status, 0, errors)
        self.assertTrue(output.startswith("READY"))
        self.assertIn(self.head, output)
        self.assertIn("120 seconds", output)
        self.assert_private(output + errors)
        self.assertEqual(self.snapshot(self.directory), before)

    def test_real_preflight_wrong_sha_branch_head_origin_and_dirty_state_fail(self):
        status, output, errors = self.preflight("0" * 64)
        self.assertEqual(status, 1)
        self.assertIn("NOT READY", errors)
        self.git("checkout", "-qb", "feature/wrong")
        self.assertIn("BRANCH_MISMATCH", self.preflight()[2])
        self.git("checkout", "-q", self.branch)
        (self.repository / "tracked.txt").write_bytes(b"modified\n")
        self.assertIn("WORKTREE_NOT_CLEAN", self.preflight()[2])
        (self.repository / "tracked.txt").write_bytes(b"original\n")
        self.git("remote", "set-url", "origin", str(self.directory / "Other Remote.git"))
        self.assertIn("ORIGIN_MISMATCH", self.preflight()[2])
        self.git("remote", "set-url", "origin", str(self.remote))
        self.git("commit", "--allow-empty", "-qm", "different HEAD")
        self.assertIn("HEAD_MISMATCH", self.preflight()[2])


if __name__ == "__main__":
    unittest.main()
