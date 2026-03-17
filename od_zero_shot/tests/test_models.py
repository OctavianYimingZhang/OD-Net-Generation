"""canonical pipeline 模型 smoke 测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

if torch is not None:
    from od_zero_shot.data.dataset import sample_to_tensor_dict
    from od_zero_shot.data.fixtures import generate_synthetic_toy100
    from od_zero_shot.data.sample_builder import build_single_fixture_sample
    from od_zero_shot.models.autoencoder import ODAutoencoder
    from od_zero_shot.models.diffusion import ConditionalLatentDiffusion
    from od_zero_shot.models.graphgps import GraphGPSRegressor


@unittest.skipIf(torch is None, f"torch 不可用: {TORCH_IMPORT_ERROR}")
class CanonicalModelShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_single_fixture_sample(
            generate_synthetic_toy100(),
            split="fixture",
            knn_k=8,
            ordering="xy",
            lap_pe_dim=8,
            rw_steps=2,
            neighbor_metric="haversine",
        )
        cls.sample = sample_to_tensor_dict(fixture)
        cls.batch = {key: value.unsqueeze(0) if torch.is_tensor(value) else value for key, value in cls.sample.items()}

    def test_graphgps_forward_shape(self) -> None:
        model = GraphGPSRegressor()
        output = model(self.batch)
        self.assertEqual(tuple(output["y_pred"].shape), (1, 100, 100))
        self.assertEqual(tuple(output["pair_condition_map"].shape), (1, 64, 100, 100))

    def test_autoencoder_shape(self) -> None:
        model = ODAutoencoder(latent_channels=16)
        output = model(self.sample["y_od"].unsqueeze(0))
        self.assertEqual(tuple(output["reconstruction"].shape), (1, 1, 100, 100))
        self.assertEqual(tuple(output["latent"].shape), (1, 16, 25, 25))

    def test_diffusion_shape(self) -> None:
        regressor = GraphGPSRegressor()
        pair_condition = regressor(self.batch)["pair_condition_map"]
        autoencoder = ODAutoencoder(latent_channels=16)
        latent = autoencoder.encode(self.batch["y_od"])
        diffusion = ConditionalLatentDiffusion(latent_channels=16, pair_dim=64, diffusion_steps=10, conditional=True)
        output = diffusion.training_loss(clean_latent=latent, pair_condition=pair_condition)
        self.assertEqual(tuple(output["pred_noise"].shape), tuple(latent.shape))


if __name__ == "__main__":
    unittest.main()
