"""训练公共工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from od_zero_shot.data.dataset import ODSampleDataset, sample_to_tensor_dict
from od_zero_shot.data.fixtures import load_fixture
from od_zero_shot.data.sample_builder import build_single_fixture_sample, load_manifest_paths, load_sample
from od_zero_shot.utils.common import choose_device, ensure_dir, save_json, set_global_seed


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要先安装 torch 才能训练模型。") from exc
    return torch, DataLoader


def collate_tensor_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _ = _require_torch()
    collated: dict[str, Any] = {}
    for key in batch[0]:
        first = batch[0][key]
        if torch.is_tensor(first):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            collated[key] = [item[key] for item in batch]
    return collated


def to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    torch, _ = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_dataloader(dataset_cfg, model_cfg, manifest_path: str | None, split: str, batch_size: int, fixture_name: str | None = None):
    _, DataLoader = _require_torch()
    if fixture_name is not None:
        lap_pe_dim = 8 if model_cfg is None else model_cfg.lap_pe_dim
        rw_steps = 2 if model_cfg is None else model_cfg.rw_steps
        sample = build_single_fixture_sample(
            load_fixture(fixture_name),
            split=split,
            knn_k=dataset_cfg.knn_k,
            ordering=dataset_cfg.ordering,
            lap_pe_dim=lap_pe_dim,
            rw_steps=rw_steps,
            neighbor_metric=dataset_cfg.neighbor_metric,
        )
        return DataLoader([sample_to_tensor_dict(sample)], batch_size=1, shuffle=False, collate_fn=collate_tensor_dict)
    if manifest_path is None:
        raise ValueError("未提供 manifest_path，且未指定 fixture。")
    dataset = ODSampleDataset(load_manifest_paths(manifest_path, split))
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_tensor_dict)


def load_numpy_samples_for_gravity(dataset_cfg, model_cfg, manifest_path: str | None, split: str, fixture_name: str | None = None) -> list[dict[str, np.ndarray]]:
    if fixture_name is not None:
        lap_pe_dim = 8 if model_cfg is None else model_cfg.lap_pe_dim
        rw_steps = 2 if model_cfg is None else model_cfg.rw_steps
        return [
            build_single_fixture_sample(
                load_fixture(fixture_name),
                split=split,
                knn_k=dataset_cfg.knn_k,
                ordering=dataset_cfg.ordering,
                lap_pe_dim=lap_pe_dim,
                rw_steps=rw_steps,
                neighbor_metric=dataset_cfg.neighbor_metric,
            ).to_numpy_dict()
        ]
    if manifest_path is None:
        raise ValueError("Gravity 训练需要 manifest_path 或 fixture。")
    return [load_sample(path).to_numpy_dict() for path in load_manifest_paths(manifest_path, split)]


def masked_three_way_mse(pred, target, mask_diag, mask_pos_off, mask_zero_off):
    torch, _ = _require_torch()

    def masked_mean(square_error, mask):
        denom = torch.clamp(mask.sum(), min=1.0)
        return (square_error * mask).sum() / denom

    square_error = (pred - target) ** 2
    return (
        masked_mean(square_error, mask_diag)
        + masked_mean(square_error, mask_pos_off)
        + masked_mean(square_error, mask_zero_off)
    ) / 3.0


def create_optimizer(model, lr: float, weight_decay: float, name: str = "AdamW"):
    torch, _ = _require_torch()
    if name != "AdamW":
        raise ValueError(f"当前仅实现 AdamW，收到 {name}")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def save_torch_checkpoint(path: str | Path, model, extra: dict[str, Any] | None = None) -> None:
    torch, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_torch_checkpoint(path: str | Path, model):
    torch, _ = _require_torch()
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return payload


def save_history(path: str | Path, history: list[dict[str, float]]) -> None:
    save_json(path, history)


def prepare_run(seed: int, artifacts_dir: str | Path, device_name: str = "auto") -> str:
    set_global_seed(seed)
    ensure_dir(artifacts_dir)
    return choose_device(device_name)
