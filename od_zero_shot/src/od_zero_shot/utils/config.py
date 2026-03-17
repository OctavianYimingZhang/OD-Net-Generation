"""配置 dataclass 与 YAML 解析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    """数据层配置。"""

    raw_root: str = "data/ny_state"
    sample_size: int = 100
    knn_k: int = 8
    ordering: str = "xy"
    neighbor_metric: str = "haversine"
    split_mode: str = "county"
    built_root: str = "od_zero_shot/artifacts/datasets"
    heldout_counties: list[str] = field(default_factory=lambda: ["061"])
    val_counties: list[str] = field(default_factory=lambda: ["047"])
    num_train_samples: int = 32
    num_val_samples: int = 8
    num_test_samples: int = 8
    batch_size: int = 4


@dataclass
class ModelConfig:
    """模型结构配置。"""

    gps_layers: int = 4
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.1
    lap_pe_dim: int = 8
    rw_steps: int = 2
    pair_dim: int = 64
    latent_channels: int = 16
    diffusion_steps: int = 100


@dataclass
class TrainConfig:
    """训练参数配置。"""

    optimizer: str = "AdamW"
    lr_gravity: float = 1e-3
    lr_pair_mlp: float = 2e-4
    lr_regressor: float = 2e-4
    lr_ae: float = 1e-3
    lr_diffusion: float = 2e-4
    weight_decay: float = 1e-4
    epochs: dict[str, int] = field(
        default_factory=lambda: {
            "gravity": 1,
            "pair_mlp": 10,
            "regressor": 10,
            "ae": 10,
            "diffusion": 10,
        }
    )
    device: str = "auto"
    seed: int = 20260317


@dataclass
class EvalConfig:
    """评估与可视化配置。"""

    threshold: float = 0.0
    top_k: int = 5
    distance_bins: list[float] = field(default_factory=lambda: [0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    figures_dir: str = "od_zero_shot/artifacts/figures"
    metrics_path: str = "od_zero_shot/artifacts/metrics/metrics.json"


@dataclass
class ProjectConfig:
    """完整工程配置。"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        """将配置转成字典。"""
        return asdict(self)


def _update_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    """用字典值覆盖 dataclass 默认值。"""
    for key, value in data.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(config_path: str | Path) -> ProjectConfig:
    """从 YAML 文件读取配置。"""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    config = ProjectConfig()
    if "dataset" in payload:
        config.dataset = _update_dataclass(config.dataset, payload["dataset"])
    if "model" in payload:
        config.model = _update_dataclass(config.model, payload["model"])
    if "train" in payload:
        config.train = _update_dataclass(config.train, payload["train"])
    if "eval" in payload:
        config.eval = _update_dataclass(config.eval, payload["eval"])
    return config


def load_dataclass(config_path: str | Path, cls):
    """兼容旧接口：把 YAML 直接加载到某个 dataclass。

    这里允许文件是 sectionless，也允许是完整 ProjectConfig 的单个 section。
    """

    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if isinstance(payload, dict) and all(not isinstance(value, dict) for value in payload.values()):
        instance = cls()
        for key, value in payload.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    section_name = {
        DatasetConfig: "dataset",
        ModelConfig: "model",
        TrainConfig: "train",
        EvalConfig: "eval",
    }.get(cls)
    if section_name is None:
        raise ValueError(f"不支持的配置类型: {cls}")
    sub_payload = payload.get(section_name, {})
    instance = cls()
    for key, value in sub_payload.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def save_config_snapshot(config: ProjectConfig, output_path: str | Path) -> None:
    """将当前配置保存为 YAML 快照。"""
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
