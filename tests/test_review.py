"""Tests for the independent, read-only Codex reviewer."""

import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.integrations.windows_process import WindowsProcessResult
from auto_coding.change_evidence import ChangeEvidence, UntrackedFile
from auto_coding.review import build_review_packet, run_review
from auto_coding.test_runner import TestCommandResult, TestRunResult


class ReviewTests(unittest.TestCase):
    SESSION_ID = "a26947d0-1111-4444-8888-849398fad344"

    def setUp(self):
        self.repository = tempfile.TemporaryDirectory()
        self.addCleanup(self.repository.cleanup)
        self.prompt = "Review this exact change; private prompt marker"
        self.criteria = ["First exact criterion", "Second exact criterion"]
        self.calls = []
        self.artifact_paths = []
        self.output_files = []

    def events(self, session_id=None):
        return (json.dumps({"type": "thread.started",
                            "thread_id": session_id or self.SESSION_ID}) + "\n").encode("utf-8")

    def verdict(self, **changes):
        value = {
            "decision": "APPROVED",
            "blocking_issues": [],
            "nonblocking_notes": [],
            "checked_requirements": self.criteria.copy(),
            "confidence": "HIGH",
        }
        value.update(changes)
        return value

    def packet_inputs(self):
        changed_paths = ("new.bin", "src/app.py")
        tracked_diff = b"diff --binary\x00\xff"
        untracked_files = (UntrackedFile("new.bin", b"\x00\x80\xffuntracked"),)
        digest = hashlib.sha256()
        digest.update(len(changed_paths).to_bytes(8, "big"))
        for path in changed_paths:
            value = path.encode("utf-8")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        digest.update(len(tracked_diff).to_bytes(8, "big"))
        digest.update(tracked_diff)
        digest.update(len(untracked_files).to_bytes(8, "big"))
        for item in untracked_files:
            for value in (item.path.encode("utf-8"), item.contents):
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        evidence = ChangeEvidence(
            True, changed_paths, tracked_diff, untracked_files, digest.hexdigest(), None,
        )
        commands = [["py", "-m", "unittest", "discover"], ["ruff", "check", "src"]]
        results = TestRunResult(
            True, 2,
            (TestCommandResult(1, 0, 1.25, False, "not_needed"),
             TestCommandResult(2, 0, 0.5, False, "not_needed")),
            None,
        )
        return {
            "change_evidence": evidence,
            "acceptance_criteria": self.criteria.copy(),
            "approved_test_commands": commands,
            "test_results": results,
            "code_context": [("src/app.py", b"code context\x00\xff")],
            "max_bytes": 100_000,
        }

    def test_review_packet_is_deterministic_complete_and_preserves_binary(self):
        inputs = self.packet_inputs()
        packet = build_review_packet(**inputs)
        self.assertEqual(packet, build_review_packet(**inputs))
        value = json.loads(packet.decode("utf-8"))
        evidence = value["verified_change_evidence"]
        self.assertEqual(evidence["fingerprint"], inputs["change_evidence"].fingerprint)
        self.assertTrue(evidence["fingerprint_revalidated"])
        self.assertEqual(evidence["changed_paths"], ["new.bin", "src/app.py"])
        self.assertEqual(base64.b64decode(evidence["tracked_diff_base64"]),
                         inputs["change_evidence"].tracked_diff)
        self.assertEqual(base64.b64decode(evidence["untracked_files"][0]["contents_base64"]),
                         b"\x00\x80\xffuntracked")

        caller_inputs = value["caller_supplied_review_inputs"]
        self.assertEqual(caller_inputs["acceptance_criteria"]["items"], self.criteria)
        self.assertEqual(caller_inputs["approved_test_commands"]["items"],
                         inputs["approved_test_commands"])
        self.assertIn("not verified", caller_inputs["acceptance_criteria"]["provenance"])
        test_commands = caller_inputs["terminal_test_results"]["commands"]
        self.assertEqual([item["command"] for item in test_commands],
                         inputs["approved_test_commands"])
        self.assertEqual([item["index"] for item in test_commands], [1, 2])
        self.assertIn("does not retain executed argv",
                      caller_inputs["terminal_test_results"]["command_binding"])
        self.assertEqual(caller_inputs["selected_code_context"]["files"][0]["path"],
                         "src/app.py")
        self.assertEqual(base64.b64decode(
            caller_inputs["selected_code_context"]["files"][0]["contents_base64"]),
            b"code context\x00\xff")

    def test_review_packet_rejects_missing_failed_or_inconsistent_evidence(self):
        inputs = self.packet_inputs()
        evidence = inputs["change_evidence"]
        invalid_evidence = (
            None,
            replace(evidence, passed=False),
            replace(evidence, fingerprint="0" * 64),
            replace(evidence, tracked_diff=None),
        )
        for value in invalid_evidence:
            with self.subTest(evidence=value):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"change_evidence": value}))

    def test_review_packet_rejects_missing_or_invalid_criteria(self):
        inputs = self.packet_inputs()
        for criteria in ([], [" "], [42], None):
            with self.subTest(criteria=criteria):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"acceptance_criteria": criteria}))

    def test_review_packet_rejects_missing_or_invalid_commands_and_size_limit(self):
        inputs = self.packet_inputs()
        for commands in ([], None, [[]], [["py", " "]], [["py", 3]]):
            with self.subTest(commands=commands):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"approved_test_commands": commands}))
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"max_bytes": limit}))

    def test_review_packet_rejects_missing_inconsistent_or_failed_test_results(self):
        inputs = self.packet_inputs()
        valid = inputs["test_results"]
        invalid_results = (
            None,
            replace(valid, passed=False),
            replace(valid, commands_executed=1),
            replace(valid, command_results=valid.command_results[:1]),
            replace(valid, command_results=(valid.command_results[1], valid.command_results[0])),
            replace(valid, command_results=(
                replace(valid.command_results[0], index=2), valid.command_results[1],
            )),
            replace(valid, command_results=(
                replace(valid.command_results[0], exit_status=1), valid.command_results[1],
            )),
        )
        for result in invalid_results:
            with self.subTest(results=result):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"test_results": result}))

    def test_review_packet_rejects_timeout_and_uncertain_cleanup(self):
        inputs = self.packet_inputs()
        valid = inputs["test_results"]
        cases = (
            replace(valid.command_results[0], timed_out=True),
            replace(valid.command_results[0], cleanup_result="taskkill_failed"),
        )
        for first in cases:
            with self.subTest(result=first):
                failed = replace(valid, command_results=(first, valid.command_results[1]))
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"test_results": failed}))

    def test_review_packet_rejects_missing_or_invalid_code_context(self):
        inputs = self.packet_inputs()
        for context in ([], None, [("../private.py", b"secret")],
                        [("src/file.py", "not bytes")],
                        [("src/file.py", b"one"), ("src/file.py", b"duplicate")]):
            with self.subTest(context=context):
                with self.assertRaises(ValueError):
                    build_review_packet(**(inputs | {"code_context": context}))

    def test_review_packet_enforces_exact_encoded_size_boundary_without_truncation(self):
        inputs = self.packet_inputs()
        complete_packet = build_review_packet(**inputs)
        exact = build_review_packet(**(inputs | {"max_bytes": len(complete_packet)}))
        self.assertEqual(exact, complete_packet)
        with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
            build_review_packet(**(inputs | {"max_bytes": len(complete_packet) - 1}))
        self.assertEqual(build_review_packet(**inputs), complete_packet)

        unicode_inputs = inputs | {
            "acceptance_criteria": ['Unicode ☃ 😀, quotes " and slash \\'],
        }
        unicode_packet = build_review_packet(**unicode_inputs)
        self.assertEqual(
            build_review_packet(**(unicode_inputs | {"max_bytes": len(unicode_packet)})),
            unicode_packet,
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
            build_review_packet(**(
                unicode_inputs | {"max_bytes": len(unicode_packet) - 1}
            ))

    def test_review_packet_rejects_oversized_binary_fields_before_base64_encoding(self):
        inputs = self.packet_inputs()
        evidence = inputs["change_evidence"]

        def evidence_with(tracked_diff, untracked_contents):
            changed_paths = evidence.changed_paths
            files = (UntrackedFile("new.bin", untracked_contents),)
            digest = hashlib.sha256()
            digest.update(len(changed_paths).to_bytes(8, "big"))
            for path in changed_paths:
                value = path.encode("utf-8")
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
            digest.update(len(tracked_diff).to_bytes(8, "big"))
            digest.update(tracked_diff)
            digest.update(len(files).to_bytes(8, "big"))
            for item in files:
                for value in (item.path.encode("utf-8"), item.contents):
                    digest.update(len(value).to_bytes(8, "big"))
                    digest.update(value)
            return ChangeEvidence(
                True, changed_paths, tracked_diff, files, digest.hexdigest(), None,
            )

        cases = (
            {"change_evidence": evidence_with(b"x" * 4096, evidence.untracked_files[0].contents)},
            {"change_evidence": evidence_with(evidence.tracked_diff, b"x" * 4096)},
            {"code_context": [("src/app.py", b"x" * 4096)]},
        )
        for changes in cases:
            with self.subTest(field=tuple(changes)):
                with patch("auto_coding.review.base64.b64encode",
                           side_effect=AssertionError("base64 encoding occurred before size check")):
                    with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
                        build_review_packet(**(inputs | changes | {"max_bytes": 512}))

    def test_review_packet_rejects_oversized_text_before_json_serialization(self):
        inputs = self.packet_inputs()
        cases = (
            {"acceptance_criteria": ["criterion " + "x" * 4096]},
            {"approved_test_commands": [["py", "x" * 4096], inputs["approved_test_commands"][1]]},
        )
        for changes in cases:
            with self.subTest(field=tuple(changes)):
                with patch("auto_coding.review.json.dumps",
                           side_effect=AssertionError("JSON serialization occurred before size check")):
                    with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
                        build_review_packet(**(inputs | changes | {"max_bytes": 512}))

    def test_review_packet_bounds_many_metadata_entries_incrementally(self):
        inputs = self.packet_inputs()
        evidence = inputs["change_evidence"]

        class CountingList(list):
            def __iter__(self):
                self.iterated = 0
                for item in super().__iter__():
                    self.iterated += 1
                    yield item

        criteria = CountingList(["x"] * 1000)
        commands = CountingList([["x"] for _ in range(1000)])
        contexts = CountingList(
            [(f"src/file_{index:04}.py", b"") for index in range(1000)]
        )
        for field, value in (
            ("acceptance_criteria", criteria),
            ("approved_test_commands", commands),
            ("code_context", contexts),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
                    build_review_packet(**(inputs | {field: value, "max_bytes": 256}))
                self.assertLess(value.iterated, len(value))

        many_paths = tuple(f"src/file_{index:04}.py" for index in range(1000))
        too_many_paths = replace(evidence, changed_paths=many_paths)
        with patch("builtins.sorted", side_effect=AssertionError("paths were sorted")):
            with patch("builtins.set", side_effect=AssertionError("paths were copied to a set")):
                with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
                    build_review_packet(**(
                        inputs | {"change_evidence": too_many_paths, "max_bytes": 256}
                    ))

        paths = tuple("a" * index for index in range(1, 13))
        files = tuple(UntrackedFile(path, b"") for path in paths)
        many_files = replace(evidence, changed_paths=paths, untracked_files=files)
        with patch("builtins.set", side_effect=AssertionError("untracked paths were copied to a set")):
            with self.assertRaisesRegex(ValueError, "exceeds max_bytes"):
                build_review_packet(**(
                    inputs | {"change_evidence": many_files, "max_bytes": 300}
                ))

    def run_with_verdict(self, verdict=None, *, raw=None, process=None, events=None,
                         session_id=None):
        if process is None:
            process = WindowsProcessResult(0, False, "not_needed")

        def execute(command, **kwargs):
            self.calls.append((command, kwargs))
            self.output_files.append(kwargs["stdout_file"])
            schema_path = Path(command[command.index("--output-schema") + 1])
            verdict_path = Path(command[command.index("--output-last-message") + 1])
            self.artifact_paths.extend((schema_path, verdict_path))
            if process.exit_status == 0:
                kwargs["stdout_file"].write(self.events() if events is None else events)
            if process.exit_status == 0 and raw is not None:
                verdict_path.write_bytes(raw)
            elif process.exit_status == 0 and verdict is not None:
                verdict_path.write_text(json.dumps(verdict), encoding="utf-8")
            return process

        with patch("auto_coding.review.run_windows_process", side_effect=execute) as runner:
            result = run_review(
                self.prompt, "gpt-6-sol", "medium", self.repository.name, 12,
                self.criteria, session_id=session_id,
            )
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(all(not path.exists() for path in self.artifact_paths))
        self.assertTrue(all(file.closed for file in self.output_files))
        self.assertTrue(all(not Path(file.name).exists() for file in self.output_files))
        return result

    def test_approved_verdict_and_exact_fresh_command(self):
        result = self.run_with_verdict(self.verdict())
        self.assertTrue(result.approved)
        self.assertEqual(result.session_id, self.SESSION_ID)
        self.assertEqual(result.verdict.checked_requirements, tuple(self.criteria))
        command, options = self.calls[0]
        schema_path, verdict_path = self.artifact_paths
        self.assertEqual(command, [
            "codex", "--ask-for-approval", "never", "exec",
            "--model", "gpt-6-sol", "-c", "model_reasoning_effort=medium",
            "--sandbox", "read-only", "--output-schema", str(schema_path),
            "--output-last-message", str(verdict_path), "--json", "-",
        ])
        output_file = options.pop("stdout_file")
        self.assertEqual(options, {
            "input_bytes": self.prompt.encode("utf-8"),
            "cwd": self.repository.name,
            "timeout_seconds": 12,
        })
        self.assertIn(output_file, self.output_files)
        self.assertFalse(Path(self.repository.name) in Path(output_file.name).resolve().parents)
        self.assertNotIn("resume", command)
        self.assertFalse(Path(self.repository.name) in schema_path.parents)
        self.assertFalse(Path(self.repository.name) in verdict_path.parents)

    def test_schema_requires_exact_verdict_fields(self):
        def inspect(command, **kwargs):
            kwargs["stdout_file"].write(self.events())
            schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), {
                "decision", "blocking_issues", "nonblocking_notes",
                "checked_requirements", "confidence",
            })
            Path(command[command.index("--output-last-message") + 1]).write_text(
                json.dumps(self.verdict()), encoding="utf-8",
            )
            return WindowsProcessResult(0, False, "not_needed")

        with patch("auto_coding.review.run_windows_process", side_effect=inspect):
            result = run_review(self.prompt, "gpt-6-sol", "medium", self.repository.name,
                                12, self.criteria)
        self.assertTrue(result.approved)

    def test_rejected_and_uncertain_are_not_approved(self):
        for decision in ("REJECTED", "UNCERTAIN"):
            with self.subTest(decision=decision):
                result = self.run_with_verdict(self.verdict(decision=decision))
                self.assertFalse(result.approved)
                self.assertEqual(result.verdict.decision, decision)
                self.assertEqual(result.session_id, self.SESSION_ID)

    def test_resume_uses_exact_read_only_command_and_verdict_gate(self):
        result = self.run_with_verdict(self.verdict(), session_id=self.SESSION_ID)
        self.assertTrue(result.approved)
        self.assertEqual(result.session_id, self.SESSION_ID)
        command, options = self.calls[-1]
        schema_path, verdict_path = self.artifact_paths[-2:]
        self.assertEqual(command, [
            "codex", "--ask-for-approval", "never", "exec", "resume",
            "--model", "gpt-6-sol", "-c", "model_reasoning_effort=medium",
            "-c", 'sandbox_mode="read-only"',
            "--output-schema", str(schema_path),
            "--output-last-message", str(verdict_path),
            "--json", self.SESSION_ID, "-",
        ])
        self.assertNotIn("--last", command)
        self.assertNotIn("fork", command)
        self.assertEqual(options["input_bytes"], self.prompt.encode("utf-8"))
        self.assertEqual(options["cwd"], self.repository.name)
        self.assertEqual(options["timeout_seconds"], 12)
        self.assertNotIn(self.prompt, repr(result))

    def test_resumed_rejected_uncertain_and_incomplete_approvals(self):
        for decision in ("REJECTED", "UNCERTAIN"):
            with self.subTest(decision=decision):
                result = self.run_with_verdict(self.verdict(decision=decision),
                                               session_id=self.SESSION_ID)
                self.assertFalse(result.approved)
                self.assertEqual(result.verdict.decision, decision)
                self.assertEqual(result.session_id, self.SESSION_ID)
        for verdict, code in (
            (self.verdict(blocking_issues=["issue"]), "BLOCKING_ISSUES"),
            (self.verdict(checked_requirements=[self.criteria[0]]),
             "CRITERIA_NOT_COVERED"),
        ):
            with self.subTest(code=code):
                result = self.run_with_verdict(verdict, session_id=self.SESSION_ID)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertEqual(result.session_id, self.SESSION_ID)
        invalid = self.run_with_verdict(self.verdict(extra="unexpected"),
                                        session_id=self.SESSION_ID)
        self.assertFalse(invalid.approved)
        self.assertEqual(invalid.error_code, "VERDICT_INVALID")

    def test_invalid_resume_id_stops_before_execution(self):
        invalid = ("", "not-a-uuid", "00000000-0000-0000-0000-000000000000",
                   "a26947d0111144448888849398fad344", 42)
        with patch("auto_coding.review.run_windows_process") as runner:
            for value in invalid:
                with self.subTest(value=value):
                    result = run_review(self.prompt, "gpt-6-sol", "medium",
                                        self.repository.name, 12, self.criteria,
                                        session_id=value)
                    self.assertFalse(result.approved)
                    self.assertEqual(result.error_code, "SESSION_ID_INVALID")
                    self.assertIsNone(result.session_id)
            runner.assert_not_called()

    def test_resume_missing_or_mismatched_session_fails_closed(self):
        other_id = "ee39e95b-2222-4444-8888-2cb53be63093"
        for events, code in (
            (b'{"type":"turn.completed"}\n', "SESSION_ID_MISSING"),
            (self.events(other_id), "SESSION_ID_MISMATCH"),
        ):
            with self.subTest(code=code):
                result = self.run_with_verdict(self.verdict(), events=events,
                                               session_id=self.SESSION_ID)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertIsNone(result.session_id)
                self.assertEqual(self.calls[-1][0][4], "resume")

    def test_resume_process_failures_never_retry_or_approve(self):
        for process, code in (
            (WindowsProcessResult(7, False, "not_needed"), "PROCESS_FAILED"),
            (WindowsProcessResult(None, False, "launch_failed"), "PROCESS_LAUNCH_FAILED"),
            (WindowsProcessResult(None, True, "taskkill_completed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(None, True, "taskkill_failed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(0, False, "parent_not_reaped"),
             "PROCESS_CLEANUP_UNCERTAIN"),
        ):
            with self.subTest(code=code, process=process):
                result = self.run_with_verdict(self.verdict(), process=process,
                                               session_id=self.SESSION_ID)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertIsNone(result.session_id)
                self.assertEqual(self.calls[-1][0][4], "resume")

    def test_missing_malformed_ambiguous_and_oversized_session_evidence(self):
        other_id = "ee39e95b-2222-4444-8888-2cb53be63093"
        cases = [
            (b"", "SESSION_ID_MISSING"),
            (b'{"type":"turn.completed"}\n', "SESSION_ID_MISSING"),
            (b'{raw private model output}\n', "SESSION_EVIDENCE_MALFORMED"),
            (self.events()[:-1], "SESSION_EVIDENCE_MALFORMED"),
            (self.events("not-a-uuid"), "SESSION_ID_INVALID"),
            (self.events() + self.events(other_id), "SESSION_ID_AMBIGUOUS"),
            (b"x" * (8 * 1024 * 1024 + 1), "OUTPUT_TOO_LARGE"),
        ]
        for events, code in cases:
            with self.subTest(code=code):
                result = self.run_with_verdict(self.verdict(), events=events)
                self.assertFalse(result.approved)
                self.assertIsNone(result.session_id)
                self.assertEqual(result.error_code, code)
                self.assertNotIn(self.prompt, repr(result))
                self.assertNotIn("raw private model output", repr(result))

    def test_blocking_issues_and_missing_criteria_stop_approval(self):
        blocked = self.run_with_verdict(self.verdict(blocking_issues=["issue"]))
        self.assertFalse(blocked.approved)
        self.assertEqual(blocked.error_code, "BLOCKING_ISSUES")
        uncovered = self.run_with_verdict(self.verdict(
            checked_requirements=[self.criteria[0]],
        ))
        self.assertFalse(uncovered.approved)
        self.assertEqual(uncovered.error_code, "CRITERIA_NOT_COVERED")

    def test_malformed_missing_and_oversized_verdicts(self):
        cases = [
            (b"{raw private response", "VERDICT_INVALID_JSON"),
            (b'\xff', "VERDICT_INVALID_JSON"),
            (b"x" * (64 * 1024 + 1), "VERDICT_TOO_LARGE"),
        ]
        for raw, code in cases:
            with self.subTest(code=code):
                result = self.run_with_verdict(raw=raw)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertNotIn(self.prompt, repr(result))
                self.assertNotIn("raw private response", repr(result))
        missing = self.run_with_verdict()
        self.assertEqual(missing.error_code, "VERDICT_MISSING")

    def test_invalid_verdict_fields_fail_closed(self):
        invalid = [
            {key: value for key, value in self.verdict().items() if key != "confidence"},
            self.verdict(extra="unexpected"),
            self.verdict(decision="MAYBE"),
            self.verdict(confidence="CERTAIN"),
            self.verdict(blocking_issues="none"),
            self.verdict(checked_requirements=[42]),
        ]
        for verdict in invalid:
            with self.subTest(verdict=verdict):
                result = self.run_with_verdict(verdict)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, "VERDICT_INVALID")
        duplicate = b'{"decision":"APPROVED","decision":"APPROVED"}'
        self.assertEqual(self.run_with_verdict(raw=duplicate).error_code,
                         "VERDICT_INVALID_JSON")

    def test_process_failures_stop_before_verdict(self):
        cases = [
            (WindowsProcessResult(1, False, "not_needed"), "PROCESS_FAILED"),
            (WindowsProcessResult(None, False, "launch_failed"), "PROCESS_LAUNCH_FAILED"),
            (WindowsProcessResult(None, True, "taskkill_completed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(None, True, "taskkill_failed"), "PROCESS_TIMEOUT"),
            (WindowsProcessResult(0, False, "parent_not_reaped"), "PROCESS_CLEANUP_UNCERTAIN"),
        ]
        for process, code in cases:
            with self.subTest(process=process):
                result = self.run_with_verdict(self.verdict(), process=process)
                self.assertFalse(result.approved)
                self.assertEqual(result.error_code, code)
                self.assertEqual(result.cleanup_result, process.cleanup_result)

    def test_invalid_inputs_do_not_launch(self):
        with patch("auto_coding.review.run_windows_process") as runner:
            for timeout in (0, -1, float("nan"), True):
                with self.subTest(timeout=timeout):
                    with self.assertRaises(ValueError):
                        run_review(self.prompt, "gpt-6-sol", "medium",
                                   self.repository.name, timeout, self.criteria)
            for criteria in ([], [""], [1], "one"):
                with self.subTest(criteria=criteria):
                    with self.assertRaises(ValueError):
                        run_review(self.prompt, "gpt-6-sol", "medium",
                                   self.repository.name, 12, criteria)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
