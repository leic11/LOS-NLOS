# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GNSS NLOS (Non-Line-of-Sight) signal detection using a Spatiotemporal Cross-Attention (STCA) model. Based on Zeng et al. (2024) paper. Achieves ~99% accuracy on static dataset.

## Quick Commands

```bash
# All commands run from the stca directory
cd stca

# Data preprocessing
python -m static_preprocess --generate-stca
python -m static_preprocess --split-mode outdomain --generate-stca

# Training
python -m train_static
python -m modules.train_static --split-mode outdomain

# Model evaluation
python -m eval_static

# Ablation study
python ablation_window_size.py
```

## Architecture Overview

```
DevLab/
├── stca/                        # 核心代码目录
│   ├── modules/                   # 模块化代码（新）
│   │   ├── constants.py           # 默认参数配置
│   │   ├── spatial_encoder.py     # 空间编码器 (AAM)
│   │   ├── temporal_encoder.py    # 时序编码器 (LSTM)
│   │   ├── cross_attention.py     # 交叉注意力模块
│   │   ├── sparse_representation.py # 稀疏表示模块
│   │   ├── trainer.py             # 训练器类
│   │   ├── train_static.py        # 训练脚本
│   │   └── eval_static.py         # 评估脚本
│   ├── static_preprocess.py       # 数据预处理
│   ├── plot_distribution.py     # Data distribution visualization
│   ├── stca_model.py            # Full STCA model
│   ├── spatial_encoder.py       # Spatial encoder (AAM)
│   ├── temporal_encoder.py      # Temporal encoder (LSTM)
│   ├── cross_attention.py       # Cross-attention fusion
│   ├── sparse_representation.py # Sparse representation layer
│   ├── dl_baselines.py          # MLP, LSTM baselines
│   ├── ablation_models.py       # Ablation study models
│   ├── experiment_tracker.py    # Auto-saves results to JSON
│   ├── trainer.py               # PyTorch trainer
│   ├── train_static.py          # Training script
│   ├── eval_static.py           # Evaluation script
│   ├── ablation_window_size.py  # Window size ablation script
│   └── seed_utils.py            # Random seed utilities
├── outputs/                     # Experiment outputs (kept at root)
└── utils/                       # Utility modules (kept at root)
    └── seed_utils.py
```

## Key Components

**STCA Model** (`stca/stca_model.py`):

- Dual input: spatial `(B, max_sats, 4)` + temporal `(B, window, 4)`
- Modules: Spatial Encoder → Temporal Encoder → Cross-Attention → Sparse Rep → Classifier
- Output: probability in [0, 1] via sigmoid

**Preprocessor** (`stca/static_preprocess.py`):

- 4 features: C/N0, Elevation, Azimuth, Pseudorange_residual
- Split modes: `indomain` (per-location) or `outdomain` (cross-location)
- Saves to `.npz` with X_train, X_train_spatial, y_train, etc.

**Trainer** (`stca/trainer.py`):

- BCELoss with sigmoid output
- Dual-input mode: `model(x_spatial=..., x_temporal=...)`
- Saves best model by validation accuracy

## Configuration

默认参数在 `stca/modules/constants.py` 中定义。关键参数：

- `SPATIAL_EMBED_DIM`: 64
- `TEMPORAL_EMBED_DIM`: 64
- `CROSS_ATTN_EMBED_DIM`: 64
- `SPARSE_EMBED_DIM`: 64
- `BATCH_SIZE`: 16
- `LEARNING_RATE`: 1e-3
- `SPATIAL_DROPOUT`: 0.5
- `TEMPORAL_DROPOUT`: 0.5

## Debugging

See `DEBUGGING_GUIDE.md` for detailed troubleshooting flowcharts covering:

- Data shape validation
- Module-by-module forward pass checks
- Gradient flow verification
- Loss convergence issues

## 注意

编辑文件时：强制指定每次只写入 100～200 行，然后使用 edits 自动接收模式完成编写。