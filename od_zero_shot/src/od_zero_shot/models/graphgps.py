"""纯 PyTorch 版本的最小 GraphGPS regressor。"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseGINEConv(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.update_mlp = MLP(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.edge_mlp(x[src] + edge_attr)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        return self.update_mlp((1.0 + self.eps) * x + aggregated)


class GraphGPSLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.local_conv = DenseGINEConv(hidden_dim, dropout=dropout)
        self.global_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.local_conv(self.local_norm(x), edge_index, edge_attr))
        global_input = self.global_norm(x).unsqueeze(0)
        global_out, _ = self.global_attn(global_input, global_input, global_input, need_weights=False)
        x = x + self.dropout(global_out.squeeze(0))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class PairConditionHead(nn.Module):
    def __init__(self, hidden_dim: int, pair_dim: int, geo_dim: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        input_dim = hidden_dim * 4 + geo_dim
        self.mapper = MLP(input_dim, hidden_dim, pair_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_repr: torch.Tensor, pair_geo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_repr.shape[0]
        left = node_repr.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        right = node_repr.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        pair_features = torch.cat([left, right, left * right, left - right, pair_geo], dim=-1)
        pair_latent = self.mapper(pair_features)
        pred = self.decoder(pair_latent).squeeze(-1)
        cond_map = pair_latent.permute(2, 0, 1).contiguous()
        return cond_map, pred


class GraphGPSRegressor(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 3,
        struct_input_dim: int = 2,
        lap_pe_dim: int = 8,
        hidden_dim: int = 128,
        pair_dim: int = 64,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim + struct_input_dim + lap_pe_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([GraphGPSLayer(hidden_dim=hidden_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.pair_head = PairConditionHead(hidden_dim=hidden_dim, pair_dim=pair_dim, geo_dim=4, dropout=dropout)

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if sample["x_node"].dim() == 3:
            predictions = []
            conditions = []
            node_reprs = []
            for idx in range(sample["x_node"].shape[0]):
                result = self.forward({key: value[idx] for key, value in sample.items()})
                predictions.append(result["y_pred"])
                conditions.append(result["pair_condition_map"])
                node_reprs.append(result["node_repr"])
            stacked_pred = torch.stack(predictions, dim=0)
            stacked_cond = torch.stack(conditions, dim=0)
            return {
                "node_repr": torch.stack(node_reprs, dim=0),
                "pair_condition_map": stacked_cond,
                "y_pred": stacked_pred,
            }
        struct_feature = sample["se_feature"]
        node_features = torch.cat([sample["x_node"], struct_feature, sample["lap_pe"]], dim=-1)
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(sample["edge_attr"])
        for layer in self.layers:
            x = layer(x, sample["edge_index"], edge_attr)
        cond_map, pred = self.pair_head(x, sample["pair_geo"])
        return {"node_repr": x, "pair_condition_map": cond_map, "y_pred": pred}
