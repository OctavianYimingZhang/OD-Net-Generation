"""原始 `.pkl` 数据读取、校验与 sanitize。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from od_zero_shot.utils.common import load_pickle
from od_zero_shot.utils.geometry import county_code_from_fips, parse_fips


@dataclass(slots=True)
class RawMobilityData:
    """原始图结构。

    兼容保留旧字段名 `centroid/population/od2flow`，同时提供
    `centroids/populations/flows` 只读别名，方便主链逐步收敛。
    """

    centroid: dict[str, list[float]]
    population: dict[str, int]
    od2flow: dict[tuple[str, str], float]

    @property
    def centroids(self) -> dict[str, list[float]]:
        return self.centroid

    @property
    def populations(self) -> dict[str, int]:
        return self.population

    @property
    def flows(self) -> dict[tuple[str, str], float]:
        return self.od2flow

    @property
    def node_ids(self) -> list[str]:
        """返回 `centroid ∩ population` 的稳定节点列表。"""

        return sorted(set(self.centroid.keys()) & set(self.population.keys()))

    def summary(self) -> dict[str, int]:
        """返回当前对象上的摘要。"""

        centroid_keys = set(self.centroid.keys())
        population_keys = set(self.population.keys())
        valid_nodes = centroid_keys & population_keys
        invalid_edges = 0
        for origin, destination in self.od2flow.keys():
            if origin not in valid_nodes or destination not in valid_nodes:
                invalid_edges += 1
        counties = {county_code_from_fips(node_id) for node_id in valid_nodes}
        zero_pop = sum(1 for node_id in valid_nodes if int(self.population[node_id]) <= 0)
        return {
            "num_centroid_nodes": len(centroid_keys),
            "num_population_nodes": len(population_keys),
            "num_intersection_nodes": len(valid_nodes),
            "missing_centroid_nodes": len(population_keys - centroid_keys),
            "missing_population_nodes": len(centroid_keys - population_keys),
            "num_edges": len(self.od2flow),
            "invalid_edges": invalid_edges,
            "num_counties": len(counties),
            "num_zero_population_nodes": zero_pop,
        }


def _normalize_centroid_dict(payload: dict[str, Any]) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for node_id, coord in payload.items():
        parse_fips(str(node_id))
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValueError(f"节点 {node_id} 的坐标格式非法。")
        normalized[str(node_id)] = [float(coord[0]), float(coord[1])]
    return normalized


def _normalize_population_dict(payload: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for node_id, value in payload.items():
        parse_fips(str(node_id))
        normalized[str(node_id)] = int(value)
    return normalized


def _normalize_flow_dict(payload: dict[Any, Any]) -> dict[tuple[str, str], float]:
    normalized: dict[tuple[str, str], float] = {}
    for (origin, destination), value in payload.items():
        origin_id = str(origin)
        destination_id = str(destination)
        parse_fips(origin_id)
        parse_fips(destination_id)
        normalized[(origin_id, destination_id)] = float(value)
    return normalized


def load_raw_pickles(raw_root: str | Path) -> RawMobilityData:
    """读取原始三件套。"""

    root = Path(raw_root)
    centroid_path = root / "centroid.pkl"
    population_path = root / "population.pkl"
    od2flow_path = root / "od2flow.pkl"
    missing = [path.name for path in (centroid_path, population_path, od2flow_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少原始数据文件: {missing}")
    centroid = _normalize_centroid_dict(load_pickle(centroid_path))
    population = _normalize_population_dict(load_pickle(population_path))
    od2flow = _normalize_flow_dict(load_pickle(od2flow_path))
    return RawMobilityData(centroid=centroid, population=population, od2flow=od2flow)


def validate_raw_data(raw_data: RawMobilityData) -> dict[str, int]:
    """验证一致性并返回摘要。"""

    return raw_data.summary()


def sanitize_raw_data(raw_data: RawMobilityData) -> tuple[RawMobilityData, dict[str, int]]:
    """把原始三件套收敛到可用节点集。

    规则：
    - 只保留 `centroid ∩ population` 节点；
    - 删除所有 origin 或 destination 不在交集内的边；
    - 输出裁剪报告，供 `check_data` 和 `build_samples` 持久化。
    """

    centroid_keys = set(raw_data.centroid.keys())
    population_keys = set(raw_data.population.keys())
    valid_nodes = centroid_keys & population_keys
    sanitized_centroid = {node_id: raw_data.centroid[node_id] for node_id in sorted(valid_nodes)}
    sanitized_population = {node_id: raw_data.population[node_id] for node_id in sorted(valid_nodes)}
    sanitized_edges = {
        (origin, destination): value
        for (origin, destination), value in raw_data.od2flow.items()
        if origin in valid_nodes and destination in valid_nodes
    }
    report = {
        "num_raw_centroid_nodes": len(centroid_keys),
        "num_raw_population_nodes": len(population_keys),
        "num_intersection_nodes": len(valid_nodes),
        "num_dropped_nodes": len((centroid_keys | population_keys) - valid_nodes),
        "num_dropped_edges": len(raw_data.od2flow) - len(sanitized_edges),
        "num_zero_population_nodes": sum(1 for node_id in valid_nodes if int(raw_data.population[node_id]) <= 0),
    }
    return RawMobilityData(centroid=sanitized_centroid, population=sanitized_population, od2flow=sanitized_edges), report


def intersect_node_ids(raw_data: RawMobilityData) -> list[str]:
    """返回可用节点交集。"""

    return raw_data.node_ids
