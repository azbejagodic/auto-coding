"""Tests for the Auto-Coding command-line interface."""

import os
from copy import deepcopy
from dataclasses import replace
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
from auto_coding.operations import TimeoutDiagnostic
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

    def test_preflight_defaults_git_timeout_and_requires_positive_override_and_sha(self):
        arguments = ["plan", "preflight", str(self.destination), "--approved-sha256", self.approval,
                     "--repository", str(self.repository)]
        with patch("auto_coding.cli.preflight_frozen_plan", return_value=PlanPreflightResult(False, "GIT_FAILED")) as preflight:
            self.assertEqual(self.direct(arguments)[0], 1)
        self.assertEqual(preflight.call_args.args[-1], 120)
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

    def test_incomplete_run_and_unsupported_execution_commands_are_rejected(self):
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


class RunCliTests(unittest.TestCase):
    """Real CLI orchestration and Git gates; only Codex/test adapters are fake."""

    def test_default_git_timeout_preserves_explicit_frozen_limits(self):
        args = self.args[:]
        index = args.index("--git-timeout-seconds")
        del args[index:index + 2]
        frozen = self.frozen_path.read_bytes()
        result = self.invoke(args)
        self.assertEqual(result[0], 0, result)
        self.assertEqual(self.execute_mock.call_args.args[4], 120)
        self.assertEqual(self.a_calls[0][0][4], 11)
        self.assertEqual(self.test_calls[0][0][2], 13)
        self.assertEqual(self.b_calls[0][1]["timeout_seconds"], 12)
        self.assertEqual(self.frozen_path.read_bytes(), frozen)
        self.assertNotIn("TIMEOUT", result[1] + result[2])

    def test_preflight_timeout_notice_does_not_create_execution_metadata(self):
        failure = PlanPreflightResult(False, "GIT_TIMEOUT", batch_id=self.plan["batch_id"],
                                      timeout=TimeoutDiagnostic("git_preflight", 5))
        with patch("auto_coding.cli.preflight_frozen_plan", return_value=failure):
            result = self.invoke()
        self.assert_no_execution(result)
        self.assertIn("AUTO-CODING STOPPED - TIMEOUT", result[2])
        self.assertIn("Operation: Git preflight", result[2])
        self.assertIn("Configured limit: 5 seconds", result[2])

    def setUp(self):
        import test_orchestrator as fixtures
        fixtures.OrchestratorTests.setUp(self)
        self.plan["tasks"] = self.plan["tasks"][:1]
        self.frozen_path.unlink()
        self.refreeze()
        self.directory = self.parent
        self.metadata = self.parent / "Batch Metadata With Spaces"
        self.metadata.mkdir()
        self.checkpoint_path = self.metadata / "batch-checkpoint.json"
        self.args = ["run", str(self.frozen_path), "--approved-sha256", self.approval,
                     "--repository", str(self.repository), "--metadata-directory", str(self.metadata),
                     "--repository-id", "example/disposable", "--git-timeout-seconds", "5",
                     "--evidence-max-bytes", "100000", "--packet-max-bytes", "200000",
                     "--report-max-bytes", "200000"]

    def git(self, *args):
        import test_orchestrator as fixtures
        return fixtures.OrchestratorTests.git(self, *args)

    def refreeze(self):
        from auto_coding.plan_freeze import freeze_approved_plan
        self.approval = plan_fingerprint(self.plan)
        freeze_approved_plan(self.plan, self.approval, self.frozen_path)

    module = PlanCliTests.module
    direct = PlanCliTests.direct

    def changed_args(self, **changes):
        args = self.args[:]
        for name, value in changes.items():
            args[args.index("--" + name.replace("_", "-")) + 1] = str(value)
        return args

    def invoke(self, args=None, implement=None, tests=None, review=None):
        import test_orchestrator as fixtures
        import auto_coding.orchestrator as orchestrator
        with patch("auto_coding.orchestrator.run_implementation", side_effect=implement or (lambda *a, **k: fixtures.OrchestratorTests.implement(self, *a, **k))) as a, \
             patch("auto_coding.orchestrator.run_approved_tests", side_effect=tests or (lambda *a, **k: fixtures.OrchestratorTests.run_tests_fake(self, *a, **k))) as terminal, \
             patch("auto_coding.orchestrator.run_review_packet", side_effect=review or (lambda *a, **k: fixtures.OrchestratorTests.review(self, *a, **k))) as b, \
             patch("auto_coding.cli.execute_first_task_through_review", wraps=orchestrator.execute_first_task_through_review) as execute, \
             patch("auto_coding.cli.publish_single_task_batch", wraps=orchestrator.publish_single_task_batch) as publish:
            result = self.direct(self.args if args is None else args)
        self.adapters = (a, terminal, b)
        self.execute_mock, self.publish_mock = execute, publish
        return result

    def assert_no_execution(self, result):
        self.assertEqual(result[0], 1, result)
        self.execute_mock.assert_not_called()
        self.publish_mock.assert_not_called()
        for adapter in self.adapters:
            adapter.assert_not_called()
        self.assertFalse(self.checkpoint_path.exists())
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)

    def report(self):
        return json.loads((self.metadata / "batch-report.json").read_bytes())

    def assert_stopped(self, result, reason):
        self.assertEqual(result[0], 1, result)
        self.assertIn(reason, result[2])
        self.publish_mock.assert_not_called()
        self.assertEqual(self.git("diff", "--cached"), b"")
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.head)

    def test_actual_module_run_help_and_missing_arguments(self):
        status, output, errors = self.module("run", "--help")
        self.assertEqual(status, 0, errors)
        for flag in ("--approved-sha256", "--metadata-directory", "--report-max-bytes"):
            self.assertIn(flag, output)
        self.assertEqual(self.module("run")[0], 2)

    def test_required_approval_and_limits_are_argument_errors(self):
        for flag in ("--approved-sha256", "--repository", "--metadata-directory", "--repository-id",
                     "--evidence-max-bytes", "--packet-max-bytes", "--report-max-bytes"):
            args = self.args[:]
            index = args.index(flag)
            del args[index:index + 2]
            with self.subTest(flag=flag), patch.dict(os.environ, {"APPROVED_SHA256": self.approval}):
                self.assertEqual(self.module(*args)[0], 2)
        for name, values in (("approved_sha256", ("PRIVATE_BAD_SHA", "", "a" * 63, "A" * 64)),
                             ("git_timeout_seconds", ("0", "-1", "nan", "inf", "86401")),
                             ("evidence_max_bytes", ("0", "-1", "1.5", "67108865")),
                             ("packet_max_bytes", ("0",)), ("report_max_bytes", ("0",)),
                             ("repository_id", ("PRIVATE\nINVALID",))):
            for value in values:
                with self.subTest(name=name, value=value):
                    status, output, errors = self.module(*self.changed_args(**{name: value}))
                    self.assertEqual(status, 2)
                    self.assertNotIn("PRIVATE", output + errors)

    def test_success_real_git_publication_and_original_handoff(self):
        from auto_coding.batch_state import load_checkpoint
        import auto_coding.cli as cli
        original_execute = cli.execute_first_task_through_review
        retained = []
        def execute(*args, **kwargs):
            handoff = original_execute(*args, **kwargs)
            retained.append(handoff)
            return handoff
        with patch("auto_coding.orchestrator.execute_first_task_through_review", side_effect=execute):
            result = self.invoke()
        self.assertEqual(result[0], 0, result)
        self.assertIn("COMPLETED", result[1])
        self.assertIs(self.publish_mock.call_args.args[0], retained[0])
        self.execute_mock.assert_called_once_with(self.frozen_path, self.approval, self.repository,
                                                  self.checkpoint_path, 5.0, 100000, 200000)
        self.publish_mock.assert_called_once()
        for adapter in self.adapters:
            adapter.assert_called_once()
        sha = self.git("rev-parse", "HEAD").decode().strip()
        self.assertNotEqual(sha, self.head)
        self.assertEqual(self.git("ls-remote", "origin", f"refs/heads/{self.branch}").decode().split()[0], sha)
        self.assertIn(f"Commit SHA: {sha}", result[1])
        self.assertIn(f"Remote SHA: {sha}", result[1])
        self.assertIn(str(self.checkpoint_path), result[1].replace("\\\\", "\\"))
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "completed")
        self.assertEqual(self.report()["batch"]["completed_tasks"], 1)
        self.assertTrue((self.metadata / "batch-report.md").is_file())
        self.assertEqual(self.git("status", "--porcelain"), b"")
        self.assertEqual(self.git("show", "HEAD:new.bin"), b"\0\xffPRIVATE_BINARY\x80")
        for private in ("PRIVATE_LITERAL", "PRIVATE_ARG", "PRIVATE_BINARY", "PRIVATE_MODEL_SUMMARY", "tracked_diff_base64"):
            self.assertNotIn(private, result[1] + result[2])

    def test_mismatched_approval_stops_before_execution(self):
        self.assert_no_execution(self.invoke(self.changed_args(approved_sha256="0" * 64)))

    def test_modified_frozen_plan_stops_before_execution(self):
        self.frozen_path.write_bytes(self.frozen_path.read_bytes() + b" ")
        self.assert_no_execution(self.invoke())

    def test_initial_git_preflight_rejects_dirty_target_before_checkpoint(self):
        (self.repository / "app.txt").write_bytes(b"pre-existing user change")
        result = self.invoke()
        self.assert_no_execution(result)
        self.assertIn("WORKTREE_NOT_CLEAN", result[2])
        self.assertEqual((self.repository / "app.txt").read_bytes(), b"pre-existing user change")

    def test_v1_plan_stops_before_execution(self):
        from test_plans import make_plan
        self.plan = make_plan()
        self.frozen_path.unlink()
        self.refreeze()
        self.assert_no_execution(self.invoke(self.changed_args(approved_sha256=self.approval)))

    def test_multi_task_plan_reaches_sequential_controller_with_exact_inputs(self):
        from auto_coding.orchestrator import SequentialBatchResult
        second = deepcopy(self.plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        self.plan["tasks"].append(second)
        self.frozen_path.unlink()
        self.refreeze()
        with patch("auto_coding.cli.execute_sequential_batch", return_value=SequentialBatchResult(
                "STOPPED", "implementation", "CODEX_FAILED")) as batch, \
             patch("auto_coding.cli.execute_first_task_through_review") as single:
            result = self.direct(self.changed_args(approved_sha256=self.approval))
        self.assertEqual(result[0], 1)
        self.assertIn("CODEX_FAILED", result[2])
        single.assert_not_called()
        batch.assert_called_once_with(self.frozen_path, self.approval, self.repository, self.metadata,
                                      "example/disposable", 5.0, 100000, 200000, 200000)

    def test_sequential_report_failure_preserves_all_publication_shas_and_nonzero_status(self):
        from auto_coding.orchestrator import SequentialBatchResult
        second = deepcopy(self.plan["tasks"][0])
        second.update(task_id="task-2", order=2, depends_on=["task-1"])
        self.plan["tasks"].append(second)
        self.frozen_path.unlink()
        self.refreeze()
        publication = SequentialBatchResult("PUBLISHED_NOT_FINALIZED", "reporting", "REPORT_UNAVAILABLE",
                        report_error_code="REPORT_UNAVAILABLE", completed_task_ids=("task-1", "task-2"),
                        publications=(("task-1", "a" * 40, "a" * 40), ("task-2", "b" * 40, "b" * 40)))
        with patch("auto_coding.cli.execute_sequential_batch", return_value=publication) as batch:
            result = self.direct(self.changed_args(approved_sha256=self.approval))
        self.assertEqual(result[0], 1)
        self.assertIn("PUBLISHED_NOT_FINALIZED", result[2])
        self.assertIn("Completed tasks: 2/2", result[1])
        self.assertIn("Commit SHA: " + "b" * 40, result[1])
        self.assertIn("Remote SHA: " + "b" * 40, result[1])
        self.assertNotIn("\nCOMPLETED\n", result[1])
        batch.assert_called_once()

    def test_metadata_inside_target_and_missing_or_non_directory_rejected(self):
        inside = self.repository / "metadata"
        inside.mkdir()
        file = self.parent / "not a directory"
        file.write_bytes(b"preserve")
        for directory in (inside, self.repository, self.parent / "missing", file,
                          self.metadata / ".." / self.metadata.name):
            with self.subTest(directory=directory):
                self.assert_no_execution(self.invoke(self.changed_args(metadata_directory=directory)))
        self.assertEqual(file.read_bytes(), b"preserve")

    def test_frozen_plan_inside_target_rejected(self):
        inside = self.repository / "frozen.json"
        inside.write_bytes(self.frozen_path.read_bytes())
        args = self.args[:]
        args[1] = str(inside)
        self.assert_no_execution(self.invoke(args))
        self.assertEqual(inside.read_bytes(), self.frozen_path.read_bytes())

    def test_existing_checkpoint_or_reports_block_before_codex(self):
        for name in ("batch-checkpoint.json", "batch-report.json", "batch-report.md"):
            path = self.metadata / name
            path.write_bytes(b"PRIVATE_EXISTING_CONTENT")
            with self.subTest(name=name):
                result = self.invoke()
                self.assertEqual(result[0], 1)
                self.execute_mock.assert_not_called()
                self.publish_mock.assert_not_called()
                self.assertEqual(path.read_bytes(), b"PRIVATE_EXISTING_CONTENT")
                self.assertNotIn("PRIVATE", result[1] + result[2])
            path.unlink()

    def test_metadata_symlink_component_rejected_without_following(self):
        # Deterministic simulation covers Windows systems without symlink rights.
        import stat
        real = Path.lstat
        metadata = self.metadata
        def lstat(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            if path == metadata:
                class LinkInfo:
                    st_mode = stat.S_IFLNK
                return LinkInfo()
            return info
        with patch.object(Path, "lstat", lstat):
            self.assert_no_execution(self.invoke())

    def test_metadata_reparse_ancestor_rejected(self):
        real = Path.lstat
        parent = self.metadata.parent
        def lstat(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            if path == parent:
                class ReparseInfo:
                    st_mode = info.st_mode
                    st_file_attributes = 1024
                return ReparseInfo()
            return info
        with patch.object(Path, "lstat", lstat):
            self.assert_no_execution(self.invoke())

    def test_metadata_resolved_alias_is_rejected(self):
        real = Path.resolve
        metadata = self.metadata
        def resolve(path, *args, **kwargs):
            if path == metadata:
                return metadata.parent
            return real(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve):
            self.assert_no_execution(self.invoke())

    def test_ready_label_with_errors_cannot_invoke_publication(self):
        from auto_coding.orchestrator import FirstTaskReviewResult
        for error, checkpoint_error in (("REVIEW_FAILED", None), (None, "CHECKPOINT_WRITE_FAILED")):
            handoff = FirstTaskReviewResult("READY_FOR_PUBLICATION", "review", error, checkpoint_error)
            with patch("auto_coding.cli.execute_first_task_through_review", return_value=handoff), \
                 patch("auto_coding.cli.publish_single_task_batch") as publish:
                result = self.direct(self.args)
            self.assertEqual(result[0], 1, result)
            self.assertIn("STOPPED", result[2])
            publish.assert_not_called()

    def test_success_requires_every_finalization_observation(self):
        from auto_coding.orchestrator import FirstTaskReviewResult, SingleTaskPublishResult
        handoff = FirstTaskReviewResult("READY_FOR_PUBLICATION", "publishing")
        success = SingleTaskPublishResult(
            "PUBLISHED_FINALIZED", "reporting", published=True, checkpoint_finalized=True,
            staging_invoked=True, publishing_invoked=True, reports_exported=True,
            commit_sha=self.head, remote_sha=self.head,
            report_paths=(self.metadata / "batch-report.json", self.metadata / "batch-report.md"))
        for changes in ({"published": False}, {"checkpoint_finalized": False}, {"reports_exported": False},
                        {"staging_invoked": False}, {"publishing_invoked": False}, {"remote_sha": "0" * 40},
                        {"report_paths": ()}, {"error_code": "FINALIZATION_FAILED"},
                        {"checkpoint_error_code": "CHECKPOINT_FAILED"}, {"report_error_code": "REPORT_FAILED"}):
            with self.subTest(changes=changes), \
                 patch("auto_coding.cli.execute_first_task_through_review", return_value=handoff), \
                 patch("auto_coding.cli.publish_single_task_batch", return_value=replace(success, **changes)) as publish:
                result = self.direct(self.args)
            self.assertEqual(result[0], 1, result)
            self.assertNotIn("COMPLETED", result[1])
            publish.assert_called_once()

    def test_implementation_failure_reports_only_observed_results(self):
        failure = replace(self.a_result, succeeded=False, exit_status=1, error_code="CODEX_FAILED", report=None)
        result = self.invoke(implement=lambda *a, **k: failure)
        self.assert_stopped(result, "stage=implementation reason=CODEX_FAILED")
        self.adapters[1].assert_not_called()
        self.adapters[2].assert_not_called()
        task = self.report()["tasks"][0]
        self.assertEqual(task["implementation"]["exit_status"], 1)
        for name in ("evidence", "terminal_results", "review", "staging", "publishing"):
            self.assertIsNone(task[name])
        self.assertIsNone(task["implementation"]["model_claims"])
        self.assertIsNone(task["reviewer"]["session_id"])

    def test_out_of_scope_change_never_publishes(self):
        def implement(*a, **k):
            (self.repository / "outside.txt").write_bytes(b"preserve actual change")
            return self.a_result
        result = self.invoke(implement=implement)
        self.assert_stopped(result, "stage=evidence reason=OUT_OF_SCOPE")
        self.adapters[1].assert_not_called()
        self.adapters[2].assert_not_called()
        self.assertEqual((self.repository / "outside.txt").read_bytes(), b"preserve actual change")
        self.assertIsNone(self.report()["tasks"][0]["evidence"])

    def test_failed_and_timed_out_tests_never_launch_reviewer(self):
        from auto_coding.test_runner import TestCommandResult, TestRunResult, _command_identity
        for timed_out in (False, True):
            command = TestCommandResult(1, None if timed_out else 1, .1, timed_out,
                                        "taskkill_completed" if timed_out else "not_needed",
                                        _command_identity(self.plan["tasks"][0]["test_commands"][0]))
            result = self.invoke(tests=lambda *a, **k: TestRunResult(False, 1, (command,), 1))
            self.assert_stopped(result, "TEST_TIMEOUT" if timed_out else "TESTS_NOT_PASSED")
            self.adapters[2].assert_not_called()
            self.assertIsNone(self.report()["tasks"][0]["review"])
            self.assertEqual(self.report()["tasks"][0]["terminal_results"]["command_results"][0]["timed_out"], timed_out)
            # A fresh attempt is test setup only, never runtime recovery.
            self.checkpoint_path.unlink()
            for path in self.metadata.glob("batch-report.*"):
                path.unlink()
            (self.repository / "app.txt").write_bytes(b"original\n")
            (self.repository / "new.bin").unlink()

    def test_rejected_and_uncertain_review_never_stage(self):
        from auto_coding.review import ReviewResult, ReviewVerdict
        import test_orchestrator as fixtures
        for decision in ("REJECTED", "UNCERTAIN"):
            result = self.invoke(review=lambda *a, **k: ReviewResult(
                False, ReviewVerdict(decision, (), (), tuple(a[1]), "LOW"), "REVIEW_NOT_APPROVED",
                0, False, "not_needed", fixtures._B))
            self.assert_stopped(result, "stage=review reason=REVIEW_NOT_APPROVED")
            self.assertEqual(self.report()["tasks"][0]["review"]["model_verdict"]["decision"], decision)
            self.checkpoint_path.unlink()
            for path in self.metadata.glob("batch-report.*"):
                path.unlink()
            (self.repository / "app.txt").write_bytes(b"original\n")
            (self.repository / "new.bin").unlink()

    def test_stopped_report_failure_preserves_original_reason(self):
        failure = replace(self.a_result, succeeded=False, exit_status=1, error_code="CODEX_FAILED", report=None)
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=OSError("PRIVATE_EXCEPTION")):
            result = self.invoke(implement=lambda *a, **k: failure)
        self.assert_stopped(result, "stage=implementation reason=CODEX_FAILED")
        self.assertIn("Reporting: REPORT_UNAVAILABLE", result[2])
        self.assertNotIn("PRIVATE", result[1] + result[2])
        self.assertTrue(self.checkpoint_path.exists())

    def test_unexpected_report_exception_cannot_hide_primary_failure(self):
        failure = replace(self.a_result, succeeded=False, exit_status=1, error_code="CODEX_FAILED", report=None)
        with patch("auto_coding.cli.export_stopped_batch_report", side_effect=RuntimeError("PRIVATE_EXCEPTION")):
            result = self.invoke(implement=lambda *a, **k: failure)
        self.assert_stopped(result, "stage=implementation reason=CODEX_FAILED")
        self.assertIn("Reporting: REPORT_UNAVAILABLE", result[2])
        self.assertNotIn("PRIVATE", result[1] + result[2])

    def test_published_but_checkpoint_failed_returns_nonzero_with_shas(self):
        import auto_coding.orchestrator as orchestrator
        original = orchestrator.update_checkpoint
        def update(path, state, **kwargs):
            if state["status"] == "completed":
                raise OSError("PRIVATE_EXCEPTION")
            return original(path, state, **kwargs)
        with patch("auto_coding.orchestrator.update_checkpoint", side_effect=update):
            result = self.invoke()
        self.assertEqual(result[0], 1, result)
        self.assertIn("PUBLISHED_BUT_NOT_FINALIZED", result[2])
        sha = self.git("rev-parse", "HEAD").decode().strip()
        self.assertNotEqual(sha, self.head)
        self.assertIn(f"Commit SHA: {sha}", result[1])
        self.assertIn(f"Remote SHA: {sha}", result[1])
        self.assertIn("CHECKPOINT_FINALIZATION_FAILED", result[2])
        self.assertNotIn("COMPLETED", result[1])
        self.publish_mock.assert_called_once()

    def test_published_but_report_failed_returns_nonzero_with_shas(self):
        from auto_coding.batch_state import load_checkpoint
        with patch("auto_coding.orchestrator.export_batch_report", side_effect=OSError("PRIVATE_EXCEPTION")):
            result = self.invoke()
        self.assertEqual(result[0], 1)
        self.assertIn("PUBLISHED_BUT_NOT_FINALIZED", result[2])
        self.assertIn("Remote SHA: " + self.git("rev-parse", "HEAD").decode().strip(), result[1])
        self.assertEqual(load_checkpoint(self.checkpoint_path)["status"], "completed")
        self.assertNotIn("COMPLETED", result[1])

    def test_ambiguous_commit_or_push_stops_without_retry(self):
        from auto_coding.git_publish import GitPublishResult
        for outcome in ("COMMIT_FAILED_OR_AMBIGUOUS", "PUSH_FAILED_OR_AMBIGUOUS"):
            pushed = outcome.startswith("PUSH")
            observation = GitPublishResult(outcome, "GIT_TIMEOUT", True, pushed,
                                           self.head if pushed else None, None, self.head if pushed else None)
            with patch("auto_coding.orchestrator.publish_reviewed_changes", return_value=observation) as gate:
                result = self.invoke()
            self.assertEqual(result[0], 1, result)
            self.assertIn(outcome, result[1])
            gate.assert_called_once()
            self.publish_mock.assert_called_once()
            self.assertNotIn("COMPLETED", result[1])
            self.checkpoint_path.unlink()
            for path in self.metadata.glob("batch-report.*"):
                path.unlink()
            # Disposable fixture only: remove the index staged by the real gate.
            self.git("restore", "--staged", "--", "app.txt", "new.bin")
            (self.repository / "app.txt").write_bytes(b"original\n")
            (self.repository / "new.bin").unlink()


if __name__ == "__main__":
    unittest.main()
