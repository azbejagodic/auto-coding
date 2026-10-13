"""Tests for the Auto-Coding command-line interface."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
