"""Controller integrity checks use approved bytes, never model file claims."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_coding.test_protection import check_test_protection


class TestProtectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name).resolve()
        self.path = self.repository / "checks.py"
        self.contents = b"trusted assertion\n"
        self.path.write_bytes(self.contents)
        self.contract = {"inputs_complete": True, "failure_protocols": ["exit-status-v1"],
                         "protected_files": [{"path": "checks.py", "size_bytes": len(self.contents),
                                              "sha256": hashlib.sha256(self.contents).hexdigest()}]}

    def check(self, contract=None):
        return check_test_protection(self.repository, self.contract if contract is None else contract, 1)

    def test_exact_regular_file_integrity_passes_without_writes(self):
        before = self.path.stat()
        self.assertTrue(self.check().passed)
        self.assertEqual(self.path.read_bytes(), self.contents)
        self.assertEqual(self.path.stat().st_mtime_ns, before.st_mtime_ns)

    def test_same_size_modification_deletion_and_resize_are_rejected(self):
        for contents in (b"X" * len(self.contents), b"short", self.contents + b"more", None):
            with self.subTest(contents=contents):
                if contents is None:
                    self.path.unlink()
                else:
                    self.path.write_bytes(contents)
                self.assertFalse(self.check().passed)

    def test_no_contract_is_not_trusted_and_inline_needs_no_fictional_files(self):
        result = check_test_protection(self.repository, None, 1)
        self.assertFalse(result.passed)
        self.assertEqual(result.error_code, "TEST_CONTRACT_UNAVAILABLE")
        self.assertTrue(self.check(dict(self.contract, protected_files=[])).passed)

    def test_directory_and_hard_link_are_rejected(self):
        self.path.unlink()
        self.path.mkdir()
        self.assertEqual(self.check().error_code, "PROTECTED_TEST_INPUT_UNSAFE")
        self.path.rmdir()
        original = self.repository / "other.py"
        original.write_bytes(self.contents)
        os.link(original, self.path)
        self.assertEqual(self.check().error_code, "PROTECTED_TEST_INPUT_UNSAFE")

    def test_symbolic_link_is_rejected_even_when_bytes_match(self):
        # Mock the link observation so Windows developer-mode privileges are
        # unnecessary; regular-file tests above exercise real filesystem bytes.
        with patch("pathlib.Path.is_symlink", return_value=True):
            self.assertFalse(self.check().passed)

    def test_inaccessible_and_malformed_inputs_have_fixed_sanitized_reasons(self):
        with patch("pathlib.Path.open", side_effect=PermissionError("PRIVATE_SECRET")):
            result = self.check()
        self.assertEqual(result.error_code, "PROTECTED_TEST_INPUT_UNAVAILABLE")
        self.assertNotIn("PRIVATE_SECRET", repr(result))
        malformed = deepcopy(self.contract)
        malformed["protected_files"][0]["path"] = "../escape"
        self.assertEqual(self.check(malformed).error_code, "TEST_CONTRACT_INVALID")

    def test_hash_reads_are_bounded_chunks_and_one_overflow_byte(self):
        contents = b"x" * (128 * 1024 + 17)
        self.path.write_bytes(contents)
        self.contract["protected_files"][0].update(size_bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
        sizes = []
        original = Path.open

        class Observer:
            def __init__(self, source):
                self.source = source

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.source.close()

            def fileno(self):
                return self.source.fileno()

            def read(self, size):
                sizes.append(size)
                return self.source.read(size)

        with patch("pathlib.Path.open", lambda path, *args, **kwargs: Observer(original(path, *args, **kwargs))):
            self.assertTrue(self.check().passed)
        self.assertEqual(sizes, [65536, 65536, 17, 1])

    def test_windows_path_and_handle_ctime_difference_does_not_reject_unchanged_bytes(self):
        original = os.fstat
        def handle_stat(descriptor):
            observed = original(descriptor)
            values = {key: getattr(observed, key) for key in
                      ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")}
            values["st_ctime_ns"] = self.path.stat().st_ctime_ns + 1000
            return SimpleNamespace(**values)
        with patch("auto_coding.test_protection.os.fstat", side_effect=handle_stat):
            self.assertTrue(self.check().passed)
