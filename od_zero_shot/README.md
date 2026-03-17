# od_zero_shot

极简/样本高效的零样本城市 OD 网络生成基线工程。

当前仓库的实验叙事是 `county-held-out / region-held-out transfer`，不是跨城市 zero-shot。输入严格限制为人口分布与地理坐标；不使用卫星影像、路网图或历史 OD 图作为 encoder 输入。

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
python3 -m od_zero_shot.cli <subcommand> --config od_zero_shot/configs/default.yaml
```

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
python3 -m od_zero_shot.cli check_data --config od_zero_shot/configs/default.yaml
```

2. 构造 100 节点 train/val/test 子图样本：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli build_samples --config od_zero_shot/configs/default.yaml
```

3. 依次训练 6 个 smoke 阶段：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_gravity --config od_zero_shot/configs/default.yaml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_pair_mlp --config od_zero_shot/configs/default.yaml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_regressor --config od_zero_shot/configs/default.yaml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_ae --config od_zero_shot/configs/default.yaml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/default.yaml \
  --ae-checkpoint od_zero_shot/artifacts/checkpoints/od_autoencoder.pt

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli train_diffusion --config od_zero_shot/configs/default.yaml \
  --conditional \
  --regressor-checkpoint od_zero_shot/artifacts/checkpoints/graphgps_regressor.pt \
  --ae-checkpoint od_zero_shot/artifacts/checkpoints/od_autoencoder.pt
```

4. 评估 deterministic regressor：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/default.yaml \
  --model-kind regressor \
  --checkpoint od_zero_shot/artifacts/checkpoints/graphgps_regressor.pt \
  --split test
```

5. 评估 conditional diffusion：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=od_zero_shot/src \
python3 -m od_zero_shot.cli evaluate_infer --config od_zero_shot/configs/default.yaml \
  --model-kind conditional_diffusion \
  --checkpoint od_zero_shot/artifacts/checkpoints/conditional_diffusion.pt \
  --regressor-checkpoint od_zero_shot/artifacts/checkpoints/graphgps_regressor.pt \
  --ae-checkpoint od_zero_shot/artifacts/checkpoints/od_autoencoder.pt \
  --split test
```

## 主要产物

- 样本清单：`od_zero_shot/artifacts/datasets/manifest.json`
- 数据摘要：`od_zero_shot/artifacts/datasets/dataset_summary.json`
- checkpoints：`od_zero_shot/artifacts/checkpoints/`
- metrics：`od_zero_shot/artifacts/metrics/`
- figures：`od_zero_shot/artifacts/figures/`

## 说明

- encoder 不接触真实 OD 邻接图，只使用坐标和人口派生出的几何图。
- 生成对象是完整 weighted directed OD matrix。
- topology 由 threshold / top-k 后处理得到，不单独生成。
