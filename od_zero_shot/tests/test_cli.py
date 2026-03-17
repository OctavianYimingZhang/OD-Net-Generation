"""CLI 最小集成测试。"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


class CLISmokeTest(unittest.TestCase):
    def test_build_samples_with_fixture(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "od_zero_shot.cli",
                "build_samples",
                "--config",
                "od_zero_shot/configs/default.yaml",
                "--fixture",
                "synthetic_toy100",
            ],
            cwd=PROJECT_ROOT.parent,
            env={"PYTHONPATH": str(SRC_ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("train", result.stdout)


if __name__ == "__main__":
    unittest.main()
