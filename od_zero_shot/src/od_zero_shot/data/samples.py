"""旧数据样本入口的薄包装。

canonical pipeline 现在是 `data.sample_builder`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from od_zero_shot.data.fixtures import generate_synthetic_toy100, load_five_node_fixture
from od_zero_shot.data.raw import RawMobilityData, load_raw_pickles
from od_zero_shot.data.sample_builder import build_and_save_split_samples, build_single_fixture_sample, load_manifest_paths, load_sample
from od_zero_shot.utils.common import ensure_dir, load_pickle, save_pickle


@dataclass
class SampleBundle:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    fixtures: dict[str, dict[str, Any]]
    summary: dict[str, Any]


def sample_artifact_path(project_root: str | Path) -> Path:
    return Path(project_root) / "artifacts" / "datasets" / "samples.pkl"


def build_fixture_samples(project_root: str | Path, config) -> dict[str, dict[str, Any]]:
    return {
        "mini5": build_single_fixture_sample(
            load_five_node_fixture(),
            split="fixture",
            knn_k=min(config.dataset.knn_k, 4),
            ordering=config.dataset.ordering,
            lap_pe_dim=config.model.lap_pe_dim,
            rw_steps=config.model.rw_steps,
        ).to_numpy_dict(),
        "synthetic_toy100": build_single_fixture_sample(
            generate_synthetic_toy100(),
            split="fixture",
            knn_k=config.dataset.knn_k,
            ordering=config.dataset.ordering,
            lap_pe_dim=config.model.lap_pe_dim,
            rw_steps=config.model.rw_steps,
        ).to_numpy_dict(),
    }


def build_sample_bundle(project_root: str | Path, config, raw_data: RawMobilityData | None = None) -> SampleBundle:
    fixtures = build_fixture_samples(project_root, config)
    if raw_data is None:
        raw_root = Path(project_root) / config.dataset.raw_root
        raw_data = load_raw_pickles(raw_root) if raw_root.exists() else None
    manifest = {"train": [], "val": [], "test": []}
    if raw_data is not None:
        built_root = Path(project_root) / "artifacts" / "datasets"
        manifest = build_and_save_split_samples(
            raw_data=raw_data,
            built_root=built_root,
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
    bundle = SampleBundle(
        train=[load_sample(path).to_numpy_dict() for path in manifest["train"]],
        val=[load_sample(path).to_numpy_dict() for path in manifest["val"]],
        test=[load_sample(path).to_numpy_dict() for path in manifest["test"]],
        fixtures=fixtures,
        summary={
            "num_train_samples": len(manifest["train"]),
            "num_val_samples": len(manifest["val"]),
            "num_test_samples": len(manifest["test"]),
            "fixtures": list(fixtures.keys()),
        },
    )
    return bundle


def save_sample_bundle(project_root: str | Path, bundle: SampleBundle) -> Path:
    path = sample_artifact_path(project_root)
    ensure_dir(path.parent)
    save_pickle(bundle, path)
    return path


def load_or_build_sample_bundle(project_root: str | Path, config) -> SampleBundle:
    path = sample_artifact_path(project_root)
    if path.exists():
        return load_pickle(path)
    bundle = build_sample_bundle(project_root, config)
    save_sample_bundle(project_root, bundle)
    return bundle
