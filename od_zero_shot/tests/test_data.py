"""canonical pipeline 数据与 CLI 测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from od_zero_shot.data.fixtures import generate_synthetic_toy100, load_five_node_fixture
from od_zero_shot.data.raw import sanitize_raw_data
from od_zero_shot.data.sample_builder import build_single_fixture_sample, split_seed_ids_by_county


class CanonicalDataPipelineTest(unittest.TestCase):
    def test_sanitize_raw_data_keeps_intersection(self) -> None:
        raw = generate_synthetic_toy100()
        raw.centroid["36999000000"] = [-73.0, 40.0]
        sanitized, report = sanitize_raw_data(raw)
        self.assertNotIn("36999000000", sanitized.node_ids)
        self.assertGreaterEqual(report["num_dropped_nodes"], 1)

    def test_fixture_sample_contains_metadata(self) -> None:
        sample = build_single_fixture_sample(
            load_five_node_fixture(),
            split="fixture",
            knn_k=4,
            ordering="xy",
            lap_pe_dim=4,
            rw_steps=2,
            neighbor_metric="haversine",
        )
        self.assertEqual(sample.y_od.shape, (5, 5))
        self.assertEqual(sample.metadata["ordering"], "xy")
        self.assertEqual(sample.metadata["lap_pe_dim"], 4)

    def test_county_split_respects_heldout(self) -> None:
        split = split_seed_ids_by_county(generate_synthetic_toy100(), heldout_counties=["061"], val_counties=["047"])
        self.assertTrue(all(node_id[2:5] == "061" for node_id in split["test"]))
        self.assertTrue(all(node_id[2:5] == "047" for node_id in split["val"]))
        self.assertTrue(all(node_id[2:5] not in {"061", "047"} for node_id in split["train"]))

    def test_cli_check_data_with_fixture(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "od_zero_shot.cli",
                "check_data",
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
        self.assertIn("num_intersection_nodes", result.stdout)


if __name__ == "__main__":
    unittest.main()
