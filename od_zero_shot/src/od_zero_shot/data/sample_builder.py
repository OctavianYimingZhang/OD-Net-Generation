"""从原始图构造固定大小局部样本。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from od_zero_shot.data.raw import RawMobilityData, sanitize_raw_data
from od_zero_shot.utils.common import ensure_dir, load_json, save_json
from od_zero_shot.utils.geometry import (
    build_knn_graph,
    coordinate_delta_matrices,
    county_code_from_fips,
    degree_feature,
    haversine_matrix,
    laplacian_positional_encoding,
    log1p_safe,
    normalize_coords,
    order_indices_xy,
    rw_diagonal_feature,
    stable_sample,
)


def _order_indices(coords: np.ndarray, ordering: str) -> np.ndarray:
    if ordering == "xy":
        return order_indices_xy(coords)
    if ordering == "morton":
        coords_norm = normalize_coords(coords)
        grid = np.clip((coords_norm * 1023).astype(np.int64), 0, 1023)

        def part1by1(value: np.ndarray) -> np.ndarray:
            out = value.copy()
            out = (out | (out << 16)) & 0x0000FFFF0000FFFF
            out = (out | (out << 8)) & 0x00FF00FF00FF00FF
            out = (out | (out << 4)) & 0x0F0F0F0F0F0F0F0F
            out = (out | (out << 2)) & 0x3333333333333333
            out = (out | (out << 1)) & 0x5555555555555555
            return out

        morton = part1by1(grid[:, 0]) | (part1by1(grid[:, 1]) << 1)
        return np.argsort(morton, kind="stable")
    raise ValueError(f"不支持的 ordering: {ordering}")


@dataclass(slots=True)
class GraphSample:
    sample_id: str
    split: str
    node_ids: list[str]
    counties: list[str]
    coords: np.ndarray
    population: np.ndarray
    x_node: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    adjacency_geo: np.ndarray
    lap_pe: np.ndarray
    se_feature: np.ndarray
    distance_matrix: np.ndarray
    dx_matrix: np.ndarray
    dy_matrix: np.ndarray
    pair_geo: np.ndarray
    pair_baseline: np.ndarray
    y_od: np.ndarray
    mask_diag: np.ndarray
    mask_pos_off: np.ndarray
    mask_zero_off: np.ndarray
    row_sum: np.ndarray
    col_sum: np.ndarray
    metadata: dict[str, object]

    def to_numpy_dict(self) -> dict[str, np.ndarray]:
        return {
            "sample_id": np.asarray(self.sample_id),
            "split": np.asarray(self.split),
            "node_ids": np.asarray(self.node_ids),
            "counties": np.asarray(self.counties),
            "coords": self.coords.astype(np.float32),
            "population": self.population.astype(np.float32),
            "x_node": self.x_node.astype(np.float32),
            "edge_index": self.edge_index.astype(np.int64),
            "edge_attr": self.edge_attr.astype(np.float32),
            "adjacency_geo": self.adjacency_geo.astype(np.int8),
            "lap_pe": self.lap_pe.astype(np.float32),
            "se_feature": self.se_feature.astype(np.float32),
            "distance_matrix": self.distance_matrix.astype(np.float32),
            "dx_matrix": self.dx_matrix.astype(np.float32),
            "dy_matrix": self.dy_matrix.astype(np.float32),
            "pair_geo": self.pair_geo.astype(np.float32),
            "pair_baseline": self.pair_baseline.astype(np.float32),
            "y_od": self.y_od.astype(np.float32),
            "mask_diag": self.mask_diag.astype(np.int8),
            "mask_pos_off": self.mask_pos_off.astype(np.int8),
            "mask_zero_off": self.mask_zero_off.astype(np.int8),
            "row_sum": self.row_sum.astype(np.float32),
            "col_sum": self.col_sum.astype(np.float32),
            "metadata_json": np.asarray(json.dumps(self.metadata, ensure_ascii=False), dtype=object),
        }


def _flow_matrix(raw_data: RawMobilityData, node_ids: list[str]) -> np.ndarray:
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    flow = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    for (origin, destination), value in raw_data.flows.items():
        if origin in index and destination in index:
            flow[index[origin], index[destination]] = float(value)
    return flow


def _candidate_distances(raw_data: RawMobilityData, seed_id: str, candidate_ids: list[str], neighbor_metric: str) -> np.ndarray:
    coords = np.asarray([raw_data.centroids[node_id] for node_id in candidate_ids], dtype=np.float32)
    if neighbor_metric == "haversine":
        seed_coord = np.asarray(raw_data.centroids[seed_id], dtype=np.float32)[None, :]
        all_coords = np.concatenate([seed_coord, coords], axis=0)
        dist = haversine_matrix(all_coords)[0, 1:]
        return dist.astype(np.float32)
    if neighbor_metric == "euclidean":
        seed = np.asarray(raw_data.centroids[seed_id], dtype=np.float32)[None, :]
        return np.sqrt(((coords - seed) ** 2).sum(axis=1)).astype(np.float32)
    raise ValueError(f"不支持的 neighbor_metric: {neighbor_metric}")


def _make_sample(
    raw_data: RawMobilityData,
    node_ids: list[str],
    sample_id: str,
    split: str,
    knn_k: int,
    ordering: str,
    lap_pe_dim: int,
    rw_steps: int,
    seed_id: str,
    neighbor_metric: str,
) -> GraphSample:
    coords = np.asarray([raw_data.centroids[node_id] for node_id in node_ids], dtype=np.float32)
    population = np.asarray([raw_data.populations[node_id] for node_id in node_ids], dtype=np.float32)
    order = _order_indices(coords, ordering)
    node_ids = [node_ids[idx] for idx in order.tolist()]
    coords = coords[order]
    population = population[order]
    counties = [county_code_from_fips(node_id) for node_id in node_ids]
    coords_norm = normalize_coords(coords).astype(np.float32)
    x_node = np.concatenate([coords_norm, log1p_safe(population).reshape(-1, 1)], axis=1).astype(np.float32)
    distance_matrix = haversine_matrix(coords).astype(np.float32)
    dx_matrix, dy_matrix = coordinate_delta_matrices(coords)
    effective_k = min(knn_k, max(1, len(node_ids) - 1))
    edge_index, adjacency_geo = build_knn_graph(distance_matrix, k=effective_k)
    edge_attr = np.stack(
        [
            log1p_safe(distance_matrix[edge_index[0], edge_index[1]]),
            dx_matrix[edge_index[0], edge_index[1]],
            dy_matrix[edge_index[0], edge_index[1]],
        ],
        axis=1,
    ).astype(np.float32)
    lap_pe = laplacian_positional_encoding(adjacency_geo, dim=lap_pe_dim).astype(np.float32)
    se_feature = np.concatenate(
        [degree_feature(adjacency_geo), rw_diagonal_feature(adjacency_geo, steps=rw_steps)],
        axis=1,
    ).astype(np.float32)
    flow = _flow_matrix(raw_data, node_ids)
    y_od = log1p_safe(flow).astype(np.float32)
    mask_diag = np.eye(len(node_ids), dtype=bool)
    mask_pos_off = np.logical_and(flow > 0.0, ~mask_diag)
    mask_zero_off = np.logical_and(flow <= 0.0, ~mask_diag)
    row_sum = flow.sum(axis=1).astype(np.float32)
    col_sum = flow.sum(axis=0).astype(np.float32)
    self_loop = np.eye(len(node_ids), dtype=np.float32)
    pair_geo = np.stack([log1p_safe(distance_matrix), dx_matrix, dy_matrix, self_loop], axis=-1).astype(np.float32)
    log_pop = log1p_safe(population).astype(np.float32)
    pair_baseline = np.stack(
        [
            np.repeat(log_pop[:, None], len(node_ids), axis=1),
            np.repeat(log_pop[None, :], len(node_ids), axis=0),
            pair_geo[..., 0],
            pair_geo[..., 1],
            pair_geo[..., 2],
            pair_geo[..., 3],
        ],
        axis=-1,
    ).astype(np.float32)
    metadata = {
        "seed_id": seed_id,
        "split": split,
        "ordering": ordering,
        "knn_k": knn_k,
        "lap_pe_dim": lap_pe_dim,
        "rw_steps": rw_steps,
        "sample_size": len(node_ids),
        "neighbor_metric": neighbor_metric,
        "counties": sorted(set(counties)),
    }
    return GraphSample(
        sample_id=sample_id,
        split=split,
        node_ids=node_ids,
        counties=counties,
        coords=coords,
        population=population,
        x_node=x_node,
        edge_index=edge_index,
        edge_attr=edge_attr,
        adjacency_geo=adjacency_geo.astype(bool),
        lap_pe=lap_pe,
        se_feature=se_feature,
        distance_matrix=distance_matrix,
        dx_matrix=dx_matrix.astype(np.float32),
        dy_matrix=dy_matrix.astype(np.float32),
        pair_geo=pair_geo,
        pair_baseline=pair_baseline,
        y_od=y_od,
        mask_diag=mask_diag,
        mask_pos_off=mask_pos_off,
        mask_zero_off=mask_zero_off,
        row_sum=row_sum,
        col_sum=col_sum,
        metadata=metadata,
    )


def build_single_fixture_sample(
    raw_data: RawMobilityData,
    split: str,
    knn_k: int,
    ordering: str = "xy",
    lap_pe_dim: int = 8,
    rw_steps: int = 2,
    neighbor_metric: str = "haversine",
) -> GraphSample:
    return _make_sample(
        raw_data,
        raw_data.node_ids,
        sample_id=f"{split}_fixture",
        split=split,
        knn_k=knn_k,
        ordering=ordering,
        lap_pe_dim=lap_pe_dim,
        rw_steps=rw_steps,
        seed_id="fixture",
        neighbor_metric=neighbor_metric,
    )


def split_seed_ids_by_county(raw_data: RawMobilityData, heldout_counties: Iterable[str], val_counties: Iterable[str]) -> dict[str, list[str]]:
    heldout = set(heldout_counties)
    val = set(val_counties)
    split = {"train": [], "val": [], "test": []}
    for node_id in raw_data.node_ids:
        county = county_code_from_fips(node_id)
        if county in heldout:
            split["test"].append(node_id)
        elif county in val:
            split["val"].append(node_id)
        else:
            split["train"].append(node_id)
    return split


def build_sample_from_seed(
    raw_data: RawMobilityData,
    seed_id: str,
    sample_size: int,
    knn_k: int,
    split: str,
    sample_id: str,
    candidate_node_ids: list[str] | None = None,
    ordering: str = "xy",
    lap_pe_dim: int = 8,
    rw_steps: int = 2,
    neighbor_metric: str = "haversine",
) -> GraphSample:
    candidate_ids = raw_data.node_ids if candidate_node_ids is None else list(candidate_node_ids)
    if seed_id not in candidate_ids:
        raise ValueError(f"seed 节点 {seed_id} 不在 split={split} 的候选池中。")
    distances = _candidate_distances(raw_data, seed_id=seed_id, candidate_ids=candidate_ids, neighbor_metric=neighbor_metric)
    order = np.argsort(distances)[: min(sample_size, len(candidate_ids))]
    node_ids = [candidate_ids[idx] for idx in order.tolist()]
    return _make_sample(
        raw_data,
        node_ids,
        sample_id=sample_id,
        split=split,
        knn_k=knn_k,
        ordering=ordering,
        lap_pe_dim=lap_pe_dim,
        rw_steps=rw_steps,
        seed_id=seed_id,
        neighbor_metric=neighbor_metric,
    )


def save_sample(sample: GraphSample, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample.to_numpy_dict())


def load_sample(path: str | Path) -> GraphSample:
    with np.load(Path(path), allow_pickle=True) as data:
        metadata = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"].item()))
        return GraphSample(
            sample_id=str(data["sample_id"]),
            split=str(data["split"]),
            node_ids=[str(value) for value in data["node_ids"].tolist()],
            counties=[str(value) for value in data["counties"].tolist()],
            coords=data["coords"].astype(np.float32),
            population=data["population"].astype(np.float32),
            x_node=data["x_node"].astype(np.float32),
            edge_index=data["edge_index"].astype(np.int64),
            edge_attr=data["edge_attr"].astype(np.float32),
            adjacency_geo=data["adjacency_geo"].astype(bool),
            lap_pe=data["lap_pe"].astype(np.float32),
            se_feature=data["se_feature"].astype(np.float32),
            distance_matrix=data["distance_matrix"].astype(np.float32),
            dx_matrix=data["dx_matrix"].astype(np.float32),
            dy_matrix=data["dy_matrix"].astype(np.float32),
            pair_geo=data["pair_geo"].astype(np.float32),
            pair_baseline=data["pair_baseline"].astype(np.float32),
            y_od=data["y_od"].astype(np.float32),
            mask_diag=data["mask_diag"].astype(bool),
            mask_pos_off=data["mask_pos_off"].astype(bool),
            mask_zero_off=data["mask_zero_off"].astype(bool),
            row_sum=data["row_sum"].astype(np.float32),
            col_sum=data["col_sum"].astype(np.float32),
            metadata=metadata,
        )


def load_manifest_paths(manifest_path: str | Path, split: str) -> list[Path]:
    manifest = load_json(manifest_path)
    return [Path(path) for path in manifest.get(split, [])]


def build_and_save_split_samples(
    raw_data: RawMobilityData,
    built_root: str | Path,
    sample_size: int,
    knn_k: int,
    heldout_counties: Iterable[str],
    val_counties: Iterable[str],
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    ordering: str = "xy",
    lap_pe_dim: int = 8,
    rw_steps: int = 2,
    split_mode: str = "county",
    neighbor_metric: str = "haversine",
) -> dict[str, list[str]]:
    if split_mode != "county":
        raise ValueError(f"当前仅实现 split_mode='county'，收到 {split_mode}")
    sanitized_raw, sanitize_report = sanitize_raw_data(raw_data)
    built_root = ensure_dir(built_root)
    split_seeds = split_seed_ids_by_county(sanitized_raw, heldout_counties=heldout_counties, val_counties=val_counties)
    quotas = {"train": num_train_samples, "val": num_val_samples, "test": num_test_samples}
    manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counties = {
        "train": sorted({county_code_from_fips(node_id) for node_id in split_seeds["train"]}),
        "val": sorted({county_code_from_fips(node_id) for node_id in split_seeds["val"]}),
        "test": sorted({county_code_from_fips(node_id) for node_id in split_seeds["test"]}),
    }
    for split, seeds in split_seeds.items():
        split_dir = ensure_dir(built_root / split)
        chosen_seeds = stable_sample(sorted(seeds), quotas[split])
        for index, seed_id in enumerate(chosen_seeds):
            sample = build_sample_from_seed(
                sanitized_raw,
                seed_id=seed_id,
                sample_size=sample_size,
                knn_k=knn_k,
                split=split,
                sample_id=f"{split}_{index:04d}",
                candidate_node_ids=split_seeds[split],
                ordering=ordering,
                lap_pe_dim=lap_pe_dim,
                rw_steps=rw_steps,
                neighbor_metric=neighbor_metric,
            )
            path = split_dir / f"{sample.sample_id}.npz"
            save_sample(sample, path)
            manifest[split].append(str(path))
    save_json(built_root / "manifest.json", manifest)
    save_json(
        built_root / "dataset_summary.json",
        {
            "sanitize_report": sanitize_report,
            "split_counts": {key: len(value) for key, value in manifest.items()},
            "split_counties": split_counties,
            "sample_size": sample_size,
            "knn_k": knn_k,
            "ordering": ordering,
            "lap_pe_dim": lap_pe_dim,
            "rw_steps": rw_steps,
            "neighbor_metric": neighbor_metric,
            "split_mode": split_mode,
        },
    )
    return manifest
