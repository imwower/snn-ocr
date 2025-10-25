"""Minimal smoke test for the repo self-check command."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class SelfCheckSmokeTest(unittest.TestCase):
    def test_self_check_runs_cleanly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "self_check.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("第三方依赖扫描：0 条", result.stdout)


if __name__ == "__main__":
    unittest.main()
