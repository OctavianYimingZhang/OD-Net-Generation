"""各训练阶段的执行函数。"""

from __future__ import annotations

from pathlib import Path

from od_zero_shot.models.autoencoder import ODAutoencoder
from od_zero_shot.models.baselines import PairMLP, build_pair_features_torch, fit_gravity_from_sample_dicts
from od_zero_shot.models.diffusion import ConditionalLatentDiffusion
from od_zero_shot.models.graphgps import GraphGPSRegressor
from od_zero_shot.train.common import (
    build_dataloader,
    create_optimizer,
    load_numpy_samples_for_gravity,
    load_torch_checkpoint,
    masked_three_way_mse,
    prepare_run,
    save_history,
    save_torch_checkpoint,
    to_device,
)
from od_zero_shot.utils.common import choose_device, ensure_dir


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("需要先安装 torch 才能运行训练器。") from exc
    return torch


def _epochs(train_cfg, stage: str) -> int:
    return int(train_cfg.epochs.get(stage, 1)) if isinstance(train_cfg.epochs, dict) else int(train_cfg.epochs)


def train_gravity_stage(dataset_cfg, train_cfg, split: str, manifest_path: str | None, fixture_name: str | None, checkpoint_dir: str | Path):
    checkpoint_dir = ensure_dir(checkpoint_dir)
    samples = load_numpy_samples_for_gravity(dataset_cfg, None, manifest_path=manifest_path, split=split, fixture_name=fixture_name)
    model = fit_gravity_from_sample_dicts(samples)
    model_path = checkpoint_dir / "gravity_model.json"
    model.save(model_path)
    return {"checkpoint": str(model_path), "num_samples": len(samples)}


def train_pair_mlp_stage(dataset_cfg, model_cfg, train_cfg, split: str, manifest_path: str | None, fixture_name: str | None, checkpoint_dir: str | Path):
    torch = _require_torch()
    device = choose_device(train_cfg.device)
    prepare_run(train_cfg.seed, checkpoint_dir, train_cfg.device)
    dataloader = build_dataloader(dataset_cfg, model_cfg, manifest_path=manifest_path, split=split, batch_size=dataset_cfg.batch_size, fixture_name=fixture_name)
    model = PairMLP(hidden_dim=model_cfg.hidden_dim, dropout=model_cfg.dropout).to(device)
    optimizer = create_optimizer(model, lr=train_cfg.lr_pair_mlp, weight_decay=train_cfg.weight_decay, name=train_cfg.optimizer)
    history = []
    for epoch in range(_epochs(train_cfg, "pair_mlp")):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = to_device(batch, device)
            pred = model(build_pair_features_torch(batch))
            loss = masked_three_way_mse(pred=pred, target=batch["y_od"], mask_diag=batch["mask_diag"], mask_pos_off=batch["mask_pos_off"], mask_zero_off=batch["mask_zero_off"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, len(dataloader))})
    save_torch_checkpoint(Path(checkpoint_dir) / "pair_mlp.pt", model)
    save_history(Path(checkpoint_dir) / "pair_mlp_history.json", history)
    return history


def train_regressor_stage(dataset_cfg, model_cfg, train_cfg, split: str, manifest_path: str | None, fixture_name: str | None, checkpoint_dir: str | Path):
    device = choose_device(train_cfg.device)
    prepare_run(train_cfg.seed, checkpoint_dir, train_cfg.device)
    dataloader = build_dataloader(dataset_cfg, model_cfg, manifest_path=manifest_path, split=split, batch_size=dataset_cfg.batch_size, fixture_name=fixture_name)
    model = GraphGPSRegressor(hidden_dim=model_cfg.hidden_dim, heads=model_cfg.heads, num_layers=model_cfg.gps_layers, pair_dim=model_cfg.pair_dim, dropout=model_cfg.dropout).to(device)
    optimizer = create_optimizer(model, lr=train_cfg.lr_regressor, weight_decay=train_cfg.weight_decay, name=train_cfg.optimizer)
    history = []
    for epoch in range(_epochs(train_cfg, "regressor")):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = to_device(batch, device)
            output = model(batch)
            loss = masked_three_way_mse(pred=output["y_pred"], target=batch["y_od"], mask_diag=batch["mask_diag"], mask_pos_off=batch["mask_pos_off"], mask_zero_off=batch["mask_zero_off"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, len(dataloader))})
    save_torch_checkpoint(Path(checkpoint_dir) / "graphgps_regressor.pt", model)
    save_history(Path(checkpoint_dir) / "graphgps_regressor_history.json", history)
    return history


def train_ae_stage(dataset_cfg, model_cfg, train_cfg, split: str, manifest_path: str | None, fixture_name: str | None, checkpoint_dir: str | Path):
    device = choose_device(train_cfg.device)
    prepare_run(train_cfg.seed, checkpoint_dir, train_cfg.device)
    dataloader = build_dataloader(dataset_cfg, model_cfg, manifest_path=manifest_path, split=split, batch_size=dataset_cfg.batch_size, fixture_name=fixture_name)
    model = ODAutoencoder(latent_channels=model_cfg.latent_channels).to(device)
    optimizer = create_optimizer(model, lr=train_cfg.lr_ae, weight_decay=train_cfg.weight_decay, name=train_cfg.optimizer)
    history = []
    for epoch in range(_epochs(train_cfg, "ae")):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = to_device(batch, device)
            if batch["y_od"].shape[-1] != 100:
                raise ValueError("OD autoencoder 固定要求输入为 100x100。")
            output = model(batch["y_od"])
            reconstruction = output["reconstruction"]
            loss = masked_three_way_mse(
                pred=reconstruction.squeeze(1),
                target=batch["y_od"],
                mask_diag=batch["mask_diag"],
                mask_pos_off=batch["mask_pos_off"],
                mask_zero_off=batch["mask_zero_off"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, len(dataloader))})
    save_torch_checkpoint(Path(checkpoint_dir) / "od_autoencoder.pt", model)
    save_history(Path(checkpoint_dir) / "od_autoencoder_history.json", history)
    return history


def train_diffusion_stage(dataset_cfg, model_cfg, train_cfg, split: str, manifest_path: str | None, fixture_name: str | None, checkpoint_dir: str | Path, conditional: bool, regressor_checkpoint: str | None, ae_checkpoint: str | None):
    torch = _require_torch()
    device = choose_device(train_cfg.device)
    prepare_run(train_cfg.seed, checkpoint_dir, train_cfg.device)
    dataloader = build_dataloader(dataset_cfg, model_cfg, manifest_path=manifest_path, split=split, batch_size=dataset_cfg.batch_size, fixture_name=fixture_name)
    if ae_checkpoint is None:
        raise ValueError("训练 diffusion 时必须提供 autoencoder checkpoint。")
    autoencoder = ODAutoencoder(latent_channels=model_cfg.latent_channels).to(device)
    load_torch_checkpoint(ae_checkpoint, autoencoder)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    regressor = None
    if conditional:
        if regressor_checkpoint is None:
            raise ValueError("训练 conditional diffusion 时必须提供 regressor checkpoint。")
        regressor = GraphGPSRegressor(hidden_dim=model_cfg.hidden_dim, heads=model_cfg.heads, num_layers=model_cfg.gps_layers, pair_dim=model_cfg.pair_dim, dropout=model_cfg.dropout).to(device)
        load_torch_checkpoint(regressor_checkpoint, regressor)
        regressor.eval()
        for parameter in regressor.parameters():
            parameter.requires_grad_(False)
    diffusion = ConditionalLatentDiffusion(latent_channels=model_cfg.latent_channels, pair_dim=model_cfg.pair_dim, diffusion_steps=model_cfg.diffusion_steps, conditional=conditional).to(device)
    optimizer = create_optimizer(diffusion, lr=train_cfg.lr_diffusion, weight_decay=train_cfg.weight_decay, name=train_cfg.optimizer)
    history = []
    for epoch in range(_epochs(train_cfg, "diffusion")):
        diffusion.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = to_device(batch, device)
            with torch.no_grad():
                latent = autoencoder.encode(batch["y_od"]).detach()
                pair_condition = regressor(batch)["pair_condition_map"].detach() if conditional and regressor is not None else None
            output = diffusion.training_loss(clean_latent=latent, pair_condition=pair_condition)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch + 1, "loss": epoch_loss / max(1, len(dataloader))})
    filename = "conditional_diffusion.pt" if conditional else "unconditional_diffusion.pt"
    history_name = "conditional_diffusion_history.json" if conditional else "unconditional_diffusion_history.json"
    save_torch_checkpoint(Path(checkpoint_dir) / filename, diffusion)
    save_history(Path(checkpoint_dir) / history_name, history)
    return history
