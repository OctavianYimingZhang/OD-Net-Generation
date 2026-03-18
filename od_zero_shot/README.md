# od_zero_shot

极简/样本高效的零样本城市 OD 网络生成基线工程。

当前仓库的实验叙事是 `county-held-out / region-held-out transfer`，不是跨城市 zero-shot。输入严格限制为人口分布与地理坐标；不使用卫星影像、路网图或历史 OD 图作为 encoder 输入。

详细交接说明见 `HANDOFF.md`。

## 环境

- 已实际测试：`Python 3.13`、`torch 2.8`
- 安装：

```bash
pip install -e ./od_zero_shot
```

## 真实数据放置

将 New York 三件套放到：

```text
data/ny_state/
  centroid.pkl
  population.pkl
  od2flow.pkl
```

本仓库不会提交这些真实 `.pkl` 文件。

## canonical pipeline

唯一公开入口：

```bash
python3 -m od_zero_shot.cli <subcommand> --config od_zero_shot/configs/smoke.yaml
```

- `od_zero_shot/configs/smoke.yaml`：smoke 配置，默认 `batch_size=1`、各 stage `epochs=1`
- `od_zero_shot/configs/baseline.yaml`：baseline 配置，适合更长训练与 `batch_size>=4`

主链是：

```text
cli.py
  -> data/raw.py
  -> data/sample_builder.py
  -> train/runner.py
  -> eval/inference.py
```

## 运行顺序

1. 校验原始数据并输出 sanitize 摘要：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli check_data --config od_zero_shot/configs/smoke.yaml
```

2. 构造 100 节点 train/val/test 子图样本：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli build_samples --config od_zero_shot/configs/smoke.yaml
```

3. 依次训练 6 个 smoke 阶段：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_gravity --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_pair_mlp --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_regressor --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_ae --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints \
  --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/smoke.yaml \
  --checkpoint-dir od_zero_shot/artifacts/smoke_run/checkpoints \
  --conditional \
  --regressor-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt \
  --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt
```

4. 评估 deterministic regressor：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/smoke.yaml \
  --model-kind regressor \
  --checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt \
  --split test
```

5. 评估 conditional diffusion：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/smoke.yaml \
  --model-kind conditional_diffusion \
  --checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/conditional_diffusion.pt \
  --regressor-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/graphgps_regressor.pt \
  --ae-checkpoint od_zero_shot/artifacts/smoke_run/checkpoints/od_autoencoder.pt \
  --split test
```

## 主要产物

- 样本清单：`od_zero_shot/artifacts/smoke_run/datasets/manifest.json`
- 数据摘要：`od_zero_shot/artifacts/smoke_run/datasets/dataset_summary.json`
- checkpoints：`od_zero_shot/artifacts/smoke_run/checkpoints/`
- metrics：`od_zero_shot/artifacts/smoke_run/metrics/`
- figures：`od_zero_shot/artifacts/smoke_run/figures/`

`dataset_summary.json` 会记录 sanitize 摘要、split 样本数、县代码划分、`ordering/knn_k/lap_pe_dim/rw_steps/neighbor_metric`、seed 尝试次数，以及样本重叠统计。

## 说明

- encoder 不接触真实 OD 邻接图，只使用坐标和人口派生出的几何图。
- 生成对象是完整 weighted directed OD matrix。
- topology 由 threshold / top-k 后处理得到，不单独生成。
