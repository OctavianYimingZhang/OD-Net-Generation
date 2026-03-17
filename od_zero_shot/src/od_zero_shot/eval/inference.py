"""统一推理与评估入口。"""

from __future__ import annotations

from pathlib import Path

import torch

from od_zero_shot.data.dataset import sample_to_tensor_dict
from od_zero_shot.data.fixtures import load_fixture
from od_zero_shot.data.sample_builder import build_single_fixture_sample, load_manifest_paths, load_sample
from od_zero_shot.eval.metrics import aggregate_metrics, compute_all_metrics
from od_zero_shot.eval.plots import plot_distance_decay, plot_heatmap, plot_row_col_sum, plot_scatter, plot_top_k_edges
from od_zero_shot.models.autoencoder import ODAutoencoder
from od_zero_shot.models.baselines import GravityModel, PairMLP, build_pair_features_torch, gravity_predict_sample
from od_zero_shot.models.diffusion import ConditionalLatentDiffusion
from od_zero_shot.models.graphgps import GraphGPSRegressor
from od_zero_shot.utils.common import choose_device, save_json


def _load_eval_samples(manifest_path: str | None, split: str, fixture_name: str | None, dataset_cfg, model_cfg):
    if fixture_name is not None:
        return [
            build_single_fixture_sample(
                load_fixture(fixture_name),
                split=split,
                knn_k=dataset_cfg.knn_k,
                ordering=dataset_cfg.ordering,
                lap_pe_dim=model_cfg.lap_pe_dim,
                rw_steps=model_cfg.rw_steps,
            ).to_numpy_dict()
        ]
    if manifest_path is None:
        raise ValueError("评估需要 manifest_path 或 fixture。")
    return [load_sample(path).to_numpy_dict() for path in load_manifest_paths(manifest_path, split)]


def _resolve_metrics_path(base_path: str, model_kind: str, split: str) -> Path:
    path = Path(base_path)
    if path.name == "metrics.json":
        return path.with_name(f"{split}_{model_kind}_metrics.json")
    return path


def evaluate_model(dataset_cfg, model_cfg, eval_cfg, model_kind: str, checkpoint: str, manifest_path: str | None, split: str, fixture_name: str | None, regressor_checkpoint: str | None = None, ae_checkpoint: str | None = None) -> dict[str, object]:
    normalized_kind = "conditional_diffusion" if model_kind == "diffusion" else model_kind
    samples = _load_eval_samples(manifest_path=manifest_path, split=split, fixture_name=fixture_name, dataset_cfg=dataset_cfg, model_cfg=model_cfg)
    figures_dir = Path(eval_cfg.figures_dir) / split / normalized_kind
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_collection = []
    device = choose_device("cpu")

    gravity = None
    pair_mlp = None
    regressor = None
    autoencoder = None
    diffusion = None
    if normalized_kind == "gravity":
        gravity = GravityModel.load(checkpoint)
    elif normalized_kind == "pair_mlp":
        pair_mlp = PairMLP(hidden_dim=model_cfg.hidden_dim, dropout=model_cfg.dropout).to(device)
        pair_mlp.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        pair_mlp.eval()
    elif normalized_kind == "regressor":
        regressor = GraphGPSRegressor(
            hidden_dim=model_cfg.hidden_dim,
            heads=model_cfg.heads,
            num_layers=model_cfg.gps_layers,
            pair_dim=model_cfg.pair_dim,
            dropout=model_cfg.dropout,
            lap_pe_dim=model_cfg.lap_pe_dim,
        ).to(device)
        regressor.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        regressor.eval()
    elif normalized_kind in {"conditional_diffusion", "unconditional_diffusion"}:
        if ae_checkpoint is None:
            raise ValueError("评估 diffusion 需要提供 ae_checkpoint。")
        autoencoder = ODAutoencoder(latent_channels=model_cfg.latent_channels).to(device)
        autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device)["state_dict"])
        autoencoder.eval()
        conditional = normalized_kind == "conditional_diffusion"
        if conditional:
            if regressor_checkpoint is None:
                raise ValueError("评估 conditional diffusion 需要 regressor_checkpoint。")
            regressor = GraphGPSRegressor(
                hidden_dim=model_cfg.hidden_dim,
                heads=model_cfg.heads,
                num_layers=model_cfg.gps_layers,
                pair_dim=model_cfg.pair_dim,
                dropout=model_cfg.dropout,
                lap_pe_dim=model_cfg.lap_pe_dim,
            ).to(device)
            regressor.load_state_dict(torch.load(regressor_checkpoint, map_location=device)["state_dict"])
            regressor.eval()
        diffusion = ConditionalLatentDiffusion(
            latent_channels=model_cfg.latent_channels,
            pair_dim=model_cfg.pair_dim,
            diffusion_steps=model_cfg.diffusion_steps,
            conditional=conditional,
        ).to(device)
        diffusion.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        diffusion.eval()
    else:
        raise ValueError(f"未知模型类型: {model_kind}")

    for idx, sample in enumerate(samples):
        true_log = sample["y_od"]
        if normalized_kind == "gravity":
            pred_log = gravity_predict_sample(gravity, sample)
        elif normalized_kind == "pair_mlp":
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            with torch.no_grad():
                pred_log = pair_mlp(build_pair_features_torch({"pair_baseline": tensor_sample["pair_baseline"].unsqueeze(0)})).squeeze(0).cpu().numpy()
        elif normalized_kind == "regressor":
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            batch = {key: value.unsqueeze(0) for key, value in tensor_sample.items()}
            with torch.no_grad():
                pred_log = regressor(batch)["y_pred"].squeeze(0).cpu().numpy()
        else:
            tensor_sample = sample_to_tensor_dict(sample, device=device)
            batch = {key: value.unsqueeze(0) for key, value in tensor_sample.items()}
            with torch.no_grad():
                pair_condition = regressor(batch)["pair_condition_map"] if regressor is not None else None
                latent = diffusion.sample(num_samples=1, device=device, pair_condition=pair_condition)
                pred_log = autoencoder.decode(latent).squeeze(0).squeeze(0).cpu().numpy()

        metrics = compute_all_metrics(sample=sample, pred_log=pred_log, threshold=eval_cfg.threshold, top_k=eval_cfg.top_k, distance_bins=eval_cfg.distance_bins)
        metrics["sample_id"] = str(sample["sample_id"])
        metrics_collection.append(metrics)
        plot_heatmap(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_heatmap.png")
        plot_scatter(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_scatter.png")
        plot_row_col_sum(true_log=true_log, pred_log=pred_log, path=figures_dir / f"{idx:03d}_row_col_sum.png")
        plot_top_k_edges(true_log=true_log, pred_log=pred_log, top_k=eval_cfg.top_k, path=figures_dir / f"{idx:03d}_topk.png")
        plot_distance_decay(curves=metrics["distance_decay_curve"], path=figures_dir / f"{idx:03d}_distance_decay.png")

    aggregate = aggregate_metrics(metrics_collection)
    metrics_path = _resolve_metrics_path(eval_cfg.metrics_path, normalized_kind, split)
    output = {
        "model_kind": normalized_kind,
        "split": split,
        "metrics": metrics_collection,
        "aggregate": aggregate,
        "figures_dir": str(figures_dir),
    }
    save_json(metrics_path, output)
    return output
