"""项目唯一 CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from od_zero_shot.data.fixtures import load_fixture
from od_zero_shot.data.raw import load_raw_pickles, sanitize_raw_data, validate_raw_data
from od_zero_shot.data.sample_builder import build_and_save_split_samples, build_single_fixture_sample, save_sample
from od_zero_shot.eval.inference import evaluate_model
from od_zero_shot.train.runner import train_ae_stage, train_diffusion_stage, train_gravity_stage, train_pair_mlp_stage, train_regressor_stage
from od_zero_shot.utils.config import load_config
from od_zero_shot.utils.common import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="极简零样本城市 OD 网络生成基线工程")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd):
        cmd.add_argument("--config", default="od_zero_shot/configs/default.yaml", help="ProjectConfig YAML 路径")
        cmd.add_argument("--manifest-path", default=None, help="build_samples 生成的 manifest.json 路径")
        cmd.add_argument("--fixture", default=None, choices=["five_node", "synthetic_toy100"], help="直接使用内置夹具，不读取原始数据")
        cmd.add_argument("--split", default="train", choices=["train", "val", "test"], help="当前命令处理的 split")
        cmd.add_argument("--checkpoint-dir", default="od_zero_shot/artifacts/checkpoints", help="模型与日志输出目录")

    check_data = subparsers.add_parser("check_data", help="检查原始数据或夹具")
    check_data.add_argument("--config", default="od_zero_shot/configs/default.yaml")
    check_data.add_argument("--fixture", default=None, choices=["five_node", "synthetic_toy100"])

    build_samples = subparsers.add_parser("build_samples", help="从原始数据或夹具构造样本")
    add_common(build_samples)

    train_gravity = subparsers.add_parser("train_gravity", help="训练 Gravity 回归")
    add_common(train_gravity)

    train_pair = subparsers.add_parser("train_pair_mlp", help="训练 Pair MLP")
    add_common(train_pair)

    train_reg = subparsers.add_parser("train_regressor", help="训练 GraphGPS deterministic regressor")
    add_common(train_reg)

    train_ae = subparsers.add_parser("train_ae", help="训练 OD autoencoder")
    add_common(train_ae)

    train_diffusion = subparsers.add_parser("train_diffusion", help="训练 latent diffusion")
    add_common(train_diffusion)
    train_diffusion.add_argument("--conditional", action="store_true", help="是否训练 conditional diffusion")
    train_diffusion.add_argument("--regressor-checkpoint", default=None, help="GraphGPS regressor checkpoint 路径")
    train_diffusion.add_argument("--ae-checkpoint", default=None, help="OD autoencoder checkpoint 路径")

    evaluate = subparsers.add_parser("evaluate_infer", help="执行推理、评估与可视化")
    add_common(evaluate)
    evaluate.add_argument(
        "--model-kind",
        required=True,
        choices=["gravity", "pair_mlp", "regressor", "unconditional_diffusion", "conditional_diffusion", "diffusion"],
        help="要评估的模型类型",
    )
    evaluate.add_argument("--checkpoint", required=True, help="目标模型 checkpoint 路径")
    evaluate.add_argument("--regressor-checkpoint", default=None, help="diffusion 评估时所需的 regressor checkpoint")
    evaluate.add_argument("--ae-checkpoint", default=None, help="diffusion 评估时所需的 autoencoder checkpoint")
    return parser


def infer_manifest_path(config, manifest_path: str | None) -> str:
    if manifest_path is not None:
        return manifest_path
    return str(Path(config.dataset.built_root) / "manifest.json")


def handle_check_data(args) -> None:
    config = load_config(args.config)
    if args.fixture is not None:
        print(load_fixture(args.fixture).summary())
        return
    raw_data = load_raw_pickles(config.dataset.raw_root)
    _, sanitize_report = sanitize_raw_data(raw_data)
    print({"raw_summary": validate_raw_data(raw_data), "sanitize_report": sanitize_report})


def handle_build_samples(args) -> None:
    config = load_config(args.config)
    built_root = Path(config.dataset.built_root)
    if args.fixture is not None:
        raw_data = load_fixture(args.fixture)
        if args.fixture == "five_node":
            sample = build_single_fixture_sample(raw_data, split="train", knn_k=min(config.dataset.knn_k, 4))
            sample_path = built_root / "train" / "train_fixture.npz"
            save_sample(sample, sample_path)
            save_json(built_root / "manifest.json", {"train": [str(sample_path)], "val": [], "test": []})
            print({"manifest": str(built_root / "manifest.json")})
            return
        manifest = build_and_save_split_samples(
            raw_data=raw_data,
            built_root=config.dataset.built_root,
            sample_size=config.dataset.sample_size,
            knn_k=config.dataset.knn_k,
            heldout_counties=config.dataset.heldout_counties,
            val_counties=config.dataset.val_counties,
            num_train_samples=config.dataset.num_train_samples,
            num_val_samples=config.dataset.num_val_samples,
            num_test_samples=config.dataset.num_test_samples,
            ordering=config.dataset.ordering,
            lap_pe_dim=config.model.lap_pe_dim,
            rw_steps=config.model.rw_steps,
            split_mode=config.dataset.split_mode,
        )
        print(manifest)
        return
    raw_data = load_raw_pickles(config.dataset.raw_root)
    manifest = build_and_save_split_samples(
        raw_data=raw_data,
        built_root=config.dataset.built_root,
        sample_size=config.dataset.sample_size,
        knn_k=config.dataset.knn_k,
        heldout_counties=config.dataset.heldout_counties,
        val_counties=config.dataset.val_counties,
        num_train_samples=config.dataset.num_train_samples,
        num_val_samples=config.dataset.num_val_samples,
        num_test_samples=config.dataset.num_test_samples,
        ordering=config.dataset.ordering,
        lap_pe_dim=config.model.lap_pe_dim,
        rw_steps=config.model.rw_steps,
        split_mode=config.dataset.split_mode,
    )
    print(manifest)


def handle_train_gravity(args) -> None:
    config = load_config(args.config)
    print(
        train_gravity_stage(
            dataset_cfg=config.dataset,
            train_cfg=config.train,
            split=args.split,
            manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
            fixture_name=args.fixture,
            checkpoint_dir=args.checkpoint_dir,
        )
    )


def handle_train_pair_mlp(args) -> None:
    config = load_config(args.config)
    history = train_pair_mlp_stage(
        dataset_cfg=config.dataset,
        model_cfg=config.model,
        train_cfg=config.train,
        split=args.split,
        manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
        fixture_name=args.fixture,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(history[-1] if history else {})


def handle_train_regressor(args) -> None:
    config = load_config(args.config)
    history = train_regressor_stage(
        dataset_cfg=config.dataset,
        model_cfg=config.model,
        train_cfg=config.train,
        split=args.split,
        manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
        fixture_name=args.fixture,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(history[-1] if history else {})


def handle_train_ae(args) -> None:
    config = load_config(args.config)
    history = train_ae_stage(
        dataset_cfg=config.dataset,
        model_cfg=config.model,
        train_cfg=config.train,
        split=args.split,
        manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
        fixture_name=args.fixture,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(history[-1] if history else {})


def handle_train_diffusion(args) -> None:
    config = load_config(args.config)
    history = train_diffusion_stage(
        dataset_cfg=config.dataset,
        model_cfg=config.model,
        train_cfg=config.train,
        split=args.split,
        manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
        fixture_name=args.fixture,
        checkpoint_dir=args.checkpoint_dir,
        conditional=args.conditional,
        regressor_checkpoint=args.regressor_checkpoint,
        ae_checkpoint=args.ae_checkpoint,
    )
    print(history[-1] if history else {})


def handle_evaluate(args) -> None:
    config = load_config(args.config)
    result = evaluate_model(
        dataset_cfg=config.dataset,
        model_cfg=config.model,
        eval_cfg=config.eval,
        model_kind=args.model_kind,
        checkpoint=args.checkpoint,
        manifest_path=None if args.fixture is not None else infer_manifest_path(config, args.manifest_path),
        split=args.split,
        fixture_name=args.fixture,
        regressor_checkpoint=args.regressor_checkpoint,
        ae_checkpoint=args.ae_checkpoint,
    )
    print({"num_samples": len(result["metrics"]), "metrics_path": config.eval.metrics_path})


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    {
        "check_data": handle_check_data,
        "build_samples": handle_build_samples,
        "train_gravity": handle_train_gravity,
        "train_pair_mlp": handle_train_pair_mlp,
        "train_regressor": handle_train_regressor,
        "train_ae": handle_train_ae,
        "train_diffusion": handle_train_diffusion,
        "evaluate_infer": handle_evaluate,
    }[args.command](args)


if __name__ == "__main__":
    main()
