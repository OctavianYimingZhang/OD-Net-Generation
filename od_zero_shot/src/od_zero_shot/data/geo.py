"""旧几何模块的薄包装。

主逻辑已收敛到 `od_zero_shot.utils.geometry`，这里仅保留转发，
避免历史导入路径立即失效。
"""

from __future__ import annotations

from od_zero_shot.utils.geometry import (
    build_knn_graph,
    coordinate_delta_matrices,
    county_code_from_fips,
    degree_feature,
    haversine_matrix,
    inverse_log1p,
    laplacian_positional_encoding,
    log1p_safe,
    normalize_coords,
    order_indices_xy,
    parse_fips,
    rw_diagonal_feature,
)


def distance_matrix(coords):
    return haversine_matrix(coords)


def normalize_xy(coords):
    return normalize_coords(coords) * 2.0 - 1.0


def structural_features(edge_index, num_nodes, rw_steps: int = 2):
    import numpy as np

    adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
    adjacency[edge_index[0], edge_index[1]] = True
    adjacency = np.logical_or(adjacency, adjacency.T)
    return np.concatenate([degree_feature(adjacency), rw_diagonal_feature(adjacency, steps=rw_steps)], axis=1)
