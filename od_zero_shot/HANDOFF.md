# 项目交接与代码审查

本文档对应 `OD-Net-Generation/main` 当前代码与已提交 artifact，用于本地接手训练与代码审查。

## 1. 系统架构与数据流分析

### 1.1 整体框架

```text
Kang 原始三件套
  centroid.pkl / population.pkl / od2flow.pkl
        |
        v
data/raw.py
  读取 -> 校验 -> sanitize
        |
        v
data/sample_builder.py
  county split -> 100 节点局部样本 -> 几何图/PE/SE/pair 特征
        |
        v
data/dataset.py + train/common.py
  .npz -> tensor dict -> DataLoader
        |
        v
train/runner.py
  1. Gravity
  2. Pair MLP
  3. GraphGPS regressor
  4. OD autoencoder
  5. unconditional diffusion
  6. conditional diffusion
        |
        v
eval/inference.py
  checkpoint 加载 -> test 推理
        |
        v
eval/metrics.py + eval/plots.py
  JSON 指标 + PNG 图
```

### 1.2 数据流转

1. `data/raw.py` 读取 `centroid.pkl`、`population.pkl`、`od2flow.pkl`，并统一转成 `RawMobilityData`。
2. `sanitize_raw_data()` 只保留 `centroid ∩ population` 节点，删除端点不在交集内的 OD 边。
3. `data/sample_builder.py` 以县为单位做 split。默认：
   - `test counties = ["061"]`
   - `val counties = ["047"]`
   - 其余为 train
4. 对每个 split 的候选池，按 seed tract 的 haversine 距离取最近 `N=100` 个节点。
5. 样本内部按 `xy` 排序，构造：
   - `x_node = [x_norm, y_norm, log1p(pop)]`
   - 无向几何 `kNN(k=8)` 图
   - `edge_attr = [log1p(distance), dx, dy]`
   - `lap_pe`
   - `se_feature = [degree, 2-step random-walk diagonal]`
   - `pair_geo`
   - `pair_baseline`
   - `y_od = log1p(flow)`
   - `mask_diag / mask_pos_off / mask_zero_off`
   - `row_sum / col_sum`
6. 每个样本保存成 `.npz`，并写出：
   - `manifest.json`
   - `dataset_summary.json`
7. 训练阶段由 `train/common.py` 把样本转成 tensor dict，送入各 stage。
8. `GraphGPSRegressor` 输出：
   - `node_repr`
   - `pair_condition_map`
   - `y_pred`
9. `ODAutoencoder` 将 `100x100` 的 log-OD 压缩到 `16x25x25` latent。
10. `ConditionalLatentDiffusion` 在 latent 空间做 epsilon-prediction，最后由 decoder 恢复完整加权有向 OD 矩阵。
11. `eval/inference.py` 统一加载 checkpoint，输出 test metrics 与可视化。

### 1.3 当前实现中的关键技术选择

- 只使用 `numpy`、`torch`、`scikit-learn`、`matplotlib`、`PyYAML`，没有引入额外图学习框架。
- 样本格式选 `.npz`，原因是样本是固定形状矩阵，序列化和读取都直接。
- 配置格式选 YAML + dataclass，原因是 smoke 和 baseline 的差异主要是参数，不是代码路径。
- GraphGPS 使用最小手写实现：
  - local block: `GINE`
  - global block: `MultiheadAttention`
- 重建损失统一为 three-way masked MSE，避免对角线和零边支配训练。

## 2. 目录树与文件级拆解

### 2.1 当前仓库目录树

```text
repo/
├── .gitignore
├── README.md
└── od_zero_shot/
    ├── README.md
    ├── HANDOFF.md
    ├── pyproject.toml
    ├── configs/
    │   ├── smoke.yaml
    │   ├── baseline.yaml
    │   └── default.yaml
    ├── data/fixtures/
    │   ├── mini5.yaml
    │   ├── mini5/{centroid.json,population.json,od2flow.json}
    │   └── synthetic_toy100.yaml
    ├── src/od_zero_shot/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── cli.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── dataset.py
    │   │   ├── fixtures.py
    │   │   ├── geo.py
    │   │   ├── raw.py
    │   │   ├── sample_builder.py
    │   │   └── samples.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── autoencoder.py
    │   │   ├── baselines.py
    │   │   ├── diffusion.py
    │   │   └── graphgps.py
    │   ├── train/
    │   │   ├── __init__.py
    │   │   ├── common.py
    │   │   ├── datasets.py
    │   │   ├── losses.py
    │   │   ├── runner.py
    │   │   └── trainers.py
    │   ├── eval/
    │   │   ├── __init__.py
    │   │   ├── inference.py
    │   │   ├── metrics.py
    │   │   └── plots.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── common.py
    │       ├── config.py
    │       ├── geometry.py
    │       └── misc.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_cli.py
    │   ├── test_data.py
    │   └── test_models.py
    └── artifacts/smoke_run/
        ├── datasets/
        ├── checkpoints/
        ├── metrics/
        └── figures/
```

### 2.2 文件职责与依赖

#### 根目录

- `README.md`
  - 职责：仓库入口说明。
  - 依赖：无运行时依赖。
  - 被谁使用：开发者阅读。

#### `od_zero_shot/`

- `README.md`
  - 职责：项目主说明、安装方法、命令顺序、artifact 路径。
  - 依赖：与当前 `cli.py`、配置文件、artifact 目录一致。

- `HANDOFF.md`
  - 职责：当前交接文档。
  - 依赖：基于已提交代码与 artifact。

- `pyproject.toml`
  - 职责：项目打包与依赖定义。
  - 被谁使用：`pip install -e ./od_zero_shot[dev]`

#### `configs/`

- `smoke.yaml`
  - 职责：当前已验证 smoke 配置。
  - 被谁使用：`cli.py -> utils/config.py`

- `baseline.yaml`
  - 职责：更长训练轮次的 baseline 配置。
  - 被谁使用：`cli.py -> utils/config.py`

- `default.yaml`
  - 职责：旧兼容配置。
  - 状态：保留，但当前 README 与 CLI 默认不再以它为主。

#### `src/od_zero_shot/`

- `__init__.py`
  - 职责：版本号定义。

- `__main__.py`
  - 职责：支持 `python -m od_zero_shot`。
  - 依赖：调用 `cli.main()`

- `cli.py`
  - 职责：项目唯一 CLI 入口。
  - 核心函数：
    - `build_parser()`
    - `handle_check_data()`
    - `handle_build_samples()`
    - `handle_train_*()`
    - `handle_evaluate()`
  - 它调用谁：
    - `data/raw.py`
    - `data/sample_builder.py`
    - `train/runner.py`
    - `eval/inference.py`
    - `utils/config.py`
  - 谁调用它：
    - 命令行
    - `__main__.py`

#### `data/`

- `raw.py`
  - 职责：原始 `.pkl` 读取、FIPS 校验、sanitize、摘要。
  - 核心对象：
    - `RawMobilityData`
    - `load_raw_pickles()`
    - `sanitize_raw_data()`
    - `validate_raw_data()`
  - 它调用谁：
    - `utils.common.load_pickle`
    - `utils.geometry`
  - 谁调用它：
    - `cli.py`
    - `fixtures.py`
    - `sample_builder.py`

- `sample_builder.py`
  - 职责：构造 100 节点样本并持久化。
  - 核心对象：
    - `GraphSample`
    - `build_sample_from_seed()`
    - `build_and_save_split_samples()`
    - `load_sample()`
    - `load_manifest_paths()`
  - 它调用谁：
    - `raw.py`
    - `utils.geometry.py`
    - `utils.common.py`
  - 谁调用它：
    - `cli.py`
    - `train/common.py`
    - `eval/inference.py`
    - `data/samples.py`

- `dataset.py`
  - 职责：把 `.npz` 样本转成 tensor dict。
  - 核心对象：
    - `sample_to_tensor_dict()`
    - `ODSampleDataset`
  - 谁调用它：
    - `train/common.py`
    - `eval/inference.py`
    - `train/datasets.py`

- `fixtures.py`
  - 职责：测试夹具，不参与真实 Kang 主流程。
  - 核心对象：
    - `load_fixture()`
    - `load_five_node_fixture()`
    - `generate_synthetic_toy100()`
  - 谁调用它：
    - `cli.py`
    - `train/common.py`
    - `eval/inference.py`
    - 测试

- `geo.py`
  - 职责：旧几何接口包装。
  - 状态：兼容层。

- `samples.py`
  - 职责：旧样本接口包装。
  - 状态：兼容层。

#### `models/`

- `baselines.py`
  - 职责：Gravity 和 Pair MLP 基线。
  - 核心对象：
    - `GravityModel`
    - `PairMLP`
    - `build_pair_features_torch()`
  - 谁调用它：
    - `train/runner.py`
    - `eval/inference.py`

- `graphgps.py`
  - 职责：最小 GraphGPS regressor。
  - 核心对象：
    - `DenseGINEConv`
    - `GraphGPSLayer`
    - `PairConditionHead`
    - `GraphGPSRegressor`
  - 输入限制：
    - 只用几何图与人口/坐标派生特征
  - 谁调用它：
    - `train/runner.py`
    - `eval/inference.py`

- `autoencoder.py`
  - 职责：OD 矩阵卷积自编码器。
  - 核心对象：
    - `ODAutoencoder`
  - 谁调用它：
    - `train/runner.py`
    - `eval/inference.py`

- `diffusion.py`
  - 职责：latent diffusion 训练与采样。
  - 核心对象：
    - `TinyLatentUNet`
    - `GaussianDiffusion`
    - `ConditionalLatentDiffusion`
  - 谁调用它：
    - `train/runner.py`
    - `eval/inference.py`

#### `train/`

- `common.py`
  - 职责：DataLoader、loss、optimizer、checkpoint、seed、device。
  - 核心对象：
    - `build_dataloader()`
    - `masked_three_way_mse()`
    - `save_torch_checkpoint()`
    - `load_torch_checkpoint()`
  - 谁调用它：
    - `runner.py`

- `runner.py`
  - 职责：六阶段训练主干。
  - 核心函数：
    - `train_gravity_stage()`
    - `train_pair_mlp_stage()`
    - `train_regressor_stage()`
    - `train_ae_stage()`
    - `train_diffusion_stage()`
  - 它调用谁：
    - `models/*`
    - `train/common.py`
  - 谁调用它：
    - `cli.py`
    - `train/trainers.py`

- `losses.py`
  - 职责：旧损失名包装。

- `datasets.py`
  - 职责：旧数据入口包装。

- `trainers.py`
  - 职责：旧训练入口包装。

#### `eval/`

- `inference.py`
  - 职责：统一推理、评估、画图、保存 JSON。
  - 核心函数：
    - `evaluate_model()`
  - 它调用谁：
    - `dataset.py`
    - `metrics.py`
    - `plots.py`
    - `models/*`
  - 谁调用它：
    - `cli.py`

- `metrics.py`
  - 职责：OD 评估指标。
  - 核心函数：
    - `compute_all_metrics()`
    - `aggregate_metrics()`

- `plots.py`
  - 职责：PNG 诊断图。
  - 核心函数：
    - `plot_heatmap()`
    - `plot_scatter()`
    - `plot_row_col_sum()`
    - `plot_top_k_edges()`
    - `plot_distance_decay()`

#### `utils/`

- `config.py`
  - 职责：YAML -> dataclass。
  - 核心对象：
    - `ProjectConfig`
    - `load_config()`
  - 谁调用它：
    - `cli.py`

- `geometry.py`
  - 职责：几何核心函数。
  - 核心函数：
    - `haversine_matrix()`
    - `build_knn_graph()`
    - `laplacian_positional_encoding()`
    - `rw_diagonal_feature()`
    - `normalize_coords()`
  - 谁调用它：
    - `raw.py`
    - `sample_builder.py`
    - `data/geo.py`

- `common.py`
  - 职责：pickle/json I/O、随机种子、device 选择。
  - 谁调用它：
    - 数据、训练、评估多个模块

- `misc.py`
  - 职责：旧工具包装。
  - 状态：主线基本不用。

#### `tests/`

- `test_data.py`
  - 职责：验证 sanitize、split、sample 构造、fixture CLI。

- `test_models.py`
  - 职责：验证 GraphGPS、AE、diffusion 前向形状。

- `test_cli.py`
  - 职责：验证最小 CLI 训练/评估链路。

## 3. 测试结果深度解读

### 3.1 当前仓库已提交的真实 smoke 数据摘要

- 原始节点：
  - centroid: `4918`
  - population: `4918`
  - intersection: `4918`
- 原始 OD 边：`618316`
- sanitize 丢弃节点：`0`
- sanitize 丢弃边：`0`
- 零人口节点：`34`
- 样本数：
  - train: `16`
  - val: `4`
  - test: `3`
- 默认 split：
  - val county: `047`
  - test county: `061`

### 3.2 六个训练阶段的真实 smoke 结果

- Gravity
  - train MAE: `1.55496`
  - val MAE: `1.78350`
- Pair MLP
  - train loss: `20.27572`
  - val loss: `15.20376`
- GraphGPS regressor
  - train loss: `18.65365`
  - val loss: `13.67923`
- OD autoencoder
  - train loss: `15.41501`
  - val loss: `7.08304`
- Unconditional diffusion
  - train loss: `1.02274`
  - val loss: `1.00432`
- Conditional diffusion
  - train loss: `1.05243`
  - val loss: `1.00385`

### 3.3 评估指标的业务意义

- `AUROC / AUPRC / F1@tau`
  - 含义：判断一条 OD 边是否存在，即 `flow > 0`。
  - 业务意义：模型有没有学到“哪里有流动、哪里没有流动”。

- `MAE / RMSE`（log-space）
  - 含义：预测的 `log1p(flow)` 与真值偏差。
  - 业务意义：边存在以后，流量大小偏差有多大。

- `row_sum_mae / column_sum_mae / total_flow_error`
  - 含义：总流出、总流入、整图总流量守恒误差。
  - 业务意义：模型生成的 OD 图是否在总量上合理。

- `top_k_recall@k`
  - 含义：每个 origin 最重要的前 k 个目的地是否被找回。
  - 业务意义：模型是否抓住主导出行方向。

- `in_degree_distribution_error / out_degree_distribution_error`
  - 含义：阈值化后入度/出度分布与真值差距。
  - 业务意义：生成出来的稀疏拓扑是否像真实图。

- `distance_bin_curve_error / distance_decay_curve`
  - 含义：按距离分桶后的平均流量衰减误差。
  - 业务意义：模型是否遵守最基本的地理距离衰减规律。

### 3.4 当前真实 test 表现

#### Regressor 聚合结果

- AUROC: `0.6025`
- AUPRC: `0.5181`
- all_pairs MAE: `2.0210`
- positive_off_diagonal MAE: `1.9658`
- row_sum_mae: `7688.88`
- column_sum_mae: `7638.63`
- total_flow_error: `762877.46`
- top_k_recall@5: `0.0605`
- distance_bin_curve_error: `24.6780`

#### Conditional diffusion 聚合结果

- AUROC: `0.5139`
- AUPRC: `0.4598`
- all_pairs MAE: `1.8260`
- positive_off_diagonal MAE: `3.6924`
- row_sum_mae: `8231.03`
- column_sum_mae: `8229.70`
- total_flow_error: `822963.31`
- top_k_recall@5: `0.0552`
- distance_bin_curve_error: `30.7074`

### 3.5 如何解释当前结果

- 当前结果来自 `smoke.yaml`，不是长轮次 baseline。
- `smoke.yaml` 中所有 stage 都只跑 `1 epoch`。
- 因此当前结果的主要意义是：
  - 工程链路已打通
  - 数据协议正确
  - 六阶段训练与两条 test 评估命令已真实通过
- 当前 performance 不应被解读为模型上限。

### 3.6 需要特别注意的边缘情况

- 当前 `test` 只有 `3` 个样本，不是 `4` 或 `8`。这是当前配置与实际候选池/overlap 约束共同作用下的真实结果。
- 有 `34` 个零人口节点，当前 sanitize 不会删除它们。
- 不同 stage 的 loss 不能横向比较：
  - Gravity 是 MAE
  - PairMLP / Regressor / AE 是 masked MSE
  - Diffusion 是噪声预测 MSE
- 当前仓库没有同一 stage 的断点续训能力，只支持 stage 间接力。

## 4. 本地运行与继续训练指南

### 4.1 环境配置

当前代码要求：

- Python: `>=3.13`
- 安装方式：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./od_zero_shot[dev]
```

### 4.2 数据放置

把 Kang 的真实数据放到仓库根目录下：

```text
data/ny_state/
  centroid.pkl
  population.pkl
  od2flow.pkl
```

### 4.3 从头跑完整 smoke

在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli check_data --config od_zero_shot/configs/smoke.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli build_samples --config od_zero_shot/configs/smoke.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_gravity --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_pair_mlp --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_regressor --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_ae --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/smoke.yaml --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints --conditional --regressor-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/smoke.yaml --model-kind regressor --checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt --split test
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/smoke.yaml --model-kind conditional_diffusion --checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/conditional_diffusion.pt --regressor-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt --split test
```

### 4.4 跑更长训练的 baseline

如果你要继续训练，不要改代码入口，直接改配置文件：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli build_samples --config od_zero_shot/configs/baseline.yaml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src python3 -m od_zero_shot.cli train_regressor --config od_zero_shot/configs/baseline.yaml --checkpoint-dir od_zero_shot/artifacts/baseline_run/checkpoints
```

其余 stage 同理，把 `smoke.yaml` 替换成 `baseline.yaml`。

### 4.5 超参数修改入口

修改以下文件即可：

- 数据规模与 split：`od_zero_shot/configs/smoke.yaml` 或 `od_zero_shot/configs/baseline.yaml`
- `batch_size`：`dataset.batch_size`
- `epochs`：`train.epochs.*`
- 学习率：`train.lr_*`
- 模型宽度/深度：`model.gps_layers`、`model.hidden_dim`、`model.heads`、`model.pair_dim`、`model.latent_channels`、`model.diffusion_steps`

当前 baseline 配置要点：

- `batch_size = 4`
- `gravity epochs = 5`
- 其余 stage `epochs = 10`

### 4.6 模型保存位置

训练权重与日志由 `--checkpoint-dir` 决定。

例如 smoke 已验证产物在：

```text
od_zero_shot/artifacts/smoke_run/checkpoints/
```

其中包括：

- `gravity_model.json`
- `gravity_model_best.json`
- `gravity_model_final.json`
- `pair_mlp.pt / pair_mlp_best.pt / pair_mlp_final.pt`
- `graphgps_regressor.pt / *_best.pt / *_final.pt`
- `od_autoencoder.pt / *_best.pt / *_final.pt`
- `unconditional_diffusion.pt / *_best.pt / *_final.pt`
- `conditional_diffusion.pt / *_best.pt / *_final.pt`
- 各 stage `*_history.json`

### 4.7 关于断点续训

当前仓库**不支持同一 stage 的断点续训**。

证据：

- checkpoint 保存内容只有 `state_dict`
- 没有保存 optimizer state
- CLI 没有 `--resume-checkpoint`
- `runner.py` 每次都会重新初始化模型和优化器

因此当前能做的只有两类：

1. **stage 间接力**
   - 例如先完成 regressor 和 AE，再启动 diffusion。
   - 当前 diffusion 已显式要求：
     - `--ae-checkpoint`
     - `--regressor-checkpoint`（conditional 时）

2. **重新跑同一 stage**
   - 如果某个 stage 中途断了，当前只能重跑该 stage。

### 4.8 本地检查建议

你接手后先做这三步：

1. `check_data`
2. `build_samples`
3. `evaluate_infer` 复跑已提交的 smoke checkpoint

这样可以先验证：

- 数据路径正确
- 样本协议正确
- 环境和依赖正确
- 现有权重可正常加载

## 5. 当前交付的直接结论

- 真实 Kang 数据读取、sanitize、构样、六阶段 smoke 训练、两条 test 评估命令都已真实跑通。
- 当前 smoke 结果主要证明工程链路正确，不代表最终性能。
- 后续本地继续训练的最短路径是：
  - 使用 `baseline.yaml`
  - 指定新的 `--checkpoint-dir`
  - 按现有六阶段顺序继续跑
  - 不要假设当前仓库支持同 stage resume
