"""旧训练入口的薄包装。

canonical pipeline 现在是 `train.runner`。本模块仅保留兼容调用名。
"""

from __future__ import annotations

from od_zero_shot.train.runner import (
    train_ae_stage as train_autoencoder,
    train_diffusion_stage,
    train_gravity_stage,
    train_pair_mlp_stage,
    train_regressor_stage,
)
