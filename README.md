# GNSS NLOS 信号检测 - 时空交叉注意力 (STCA) 模型

本项目实现了一个基于 PyTorch 的时空交叉注意力（STCA）模型，用于 GNSS NLOS（非视距）信号检测。模型参考 Zeng 等人 (2024) 提出的架构，在静态数据集上达到 99% 以上准确率。

## ✨ 项目特点

- **双输入架构**：空间输入捕获多星联合分布特征，时序输入建模信号动态变化
- **交叉注意力融合**：以时序特征为查询、空间特征为键值，建立自适应关联
- **稀疏表示增强**：通过可学习激活函数提升特征判别性
- **完整实验流程**：支持数据预处理、模型训练、评估可视化和消融实验
- **域外泛化验证**：支持按地点划分数据，验证模型跨场景泛化能力

## 📁 项目结构

```
DevLab/
├── README.md                      # 项目说明文档
├── CLAUDE.md                      # 配置指南
├── requirements.txt               # Python 依赖
│
├── data for sharing_csv/          # 原始 GNSS 静态数据 (P2.csv ~ P8.csv)
│
├── stca/                          # 核心代码
│   ├── modules/                   # 模型模块
│   │   ├── constants.py           # 模型超参数
│   │   ├── stca_model.py          # STCA 完整模型（支持消融实验配置）
│   │   ├── spatial_encoder.py     # 空间编码器 (AAM Module)
│   │   ├── temporal_encoder.py    # 时序编码器 (LSTM-TFE Module)
│   │   ├── cross_attention.py     # 交叉注意力模块
│   │   └── sparse_representation.py # 稀疏表示模块
│   ├── data_loading/              # 数据预处理
│   │   ├── main.py                # 预处理器 (StaticPreprocessor)
│   │   ├── constants.py           # 预处理参数
│   │   ├── loaders.py             # CSV 加载器
│   │   ├── filters.py             # 数据过滤
│   │   ├── splitters.py           # 数据集划分 (Indomain/Outdomain)
│   │   ├── windowers.py           # 时间窗口生成
│   │   └── normalizers.py         # 特征归一化
│   ├── work/                      # 实验脚本
│   │   ├── train_static.py        # 训练与评估（无验证集模式）
│   │   ├── ablation_window_size.py    # 窗口长度消融 (6~32)
│   │   ├── ablation_modules.py        # 模块消融 (Baseline/CrossAttn/Both)
│   │   ├── ablation_hyperparam.py     # 超参数敏感性分析
│   │   └── ablation_baseline.py       # 基线对比实验
│   └── utils/                     # 工具模块
│
├── utils/                         # 通用工具
│   ├── logger_config.py           # 日志配置
│   └── seed_utils.py              # 随机种子设置
│
└── outputs/stca/                  # 实验输出
    ├── ablation/                  # 消融实验结果
    ├── figures/                   # 训练曲线与评估图
    └── final_model_*.pth          # 模型权重
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
cd stca

# 生成 STCA 格式数据（默认 outdomain 划分，窗口大小=8）
python -m data_loading.main --generate-stca

# 生成 indomain 划分数据
python -m data_loading.main --split-mode indomain --generate-stca
```

### 3. 训练模型

```bash
# 使用 outdomain 数据训练（默认）
python -m work.train_static

# 使用 indomain 数据训练
python -m work.train_static --split-mode indomain
```

### 4. 消融实验

```bash
# 窗口长度消融（6, 8, 10, ..., 32）
python -m work.ablation_window_size

# 模块消融（Baseline/CrossAttn/Both）
python -m work.ablation_modules

# 超参数敏感性分析
python -m work.ablation_hyperparam
```

## 📊 数据说明

### 原始数据文件

数据来源：`data for sharing_csv/` 目录，包含 7 个测站的静态 GNSS 观测数据：

| 文件 | 测站 | 记录数 |
|------|------|--------|
| P2.csv | P2 | ~6,500 |
| P3.csv | P3 | ~8,500 |
| P4.csv | P4 | ~12,500 |
| P5.csv | P5 | ~15,000 |
| P6.csv | P6 | ~10,000 |
| P7.csv | P7 | ~12,500 |
| P8.csv | P8 | ~17,000 |

### 原始数据字段（17 列）

| 字段 | 说明 | 单位 |
|------|------|------|
| GPS_Time | GPS 时间戳 | 秒 |
| PRN | 卫星伪随机噪声码编号 | - |
| nSV | 可见卫星数量 | - |
| pseudorange | 伪距观测值 | 米 |
| C/N0 | 载波噪声比 | dB-Hz |
| Elevation | 卫星高度角 | 度 (°) |
| Azimuth | 卫星方位角 | 度 (°) |
| Pseudorange_residual | 伪距残差 | 米 |
| LOS/NLOS_label | 标签：-1=NLOS, 1=LOS | - |

### 模型输入特征

**4 个原始特征**（论文推荐）：
| 特征 | 说明 |
|------|------|
| C/N0 | 载波噪声比 (dB-Hz) |
| Elevation | 高度角 (°) |
| Azimuth | 方位角 (°) |
| Pseudorange_residual | 伪距残差 (m) |

### 标签

- `0` → NLOS (非视距信号)
- `1` → LOS (视距信号)

## 🧠 模型架构

```
输入 → Spatial Encoder → Temporal Encoder → Cross-Attention → Sparse Rep → Classifier
       (AAM 模块)        (LSTM-TFE 模块)      (时空融合)        (ReLU)     (MLP+Sigmoid)
```

### 模型配置参数

所有默认参数在 `stca/modules/constants.py` 中定义：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| SPATIAL_EMBED_DIM | 空间嵌入维度 | 16 |
| SPATIAL_NUM_HEADS | 注意力头数 | 4 |
| SPATIAL_NUM_LAYERS | 编码器层数 | 2 |
| SPATIAL_D_FF | 前馈网络维度 | 32 |
| SPATIAL_DROPOUT | 空间编码器 Dropout | 0.5 |
| TEMPORAL_EMBED_DIM | 时间嵌入维度 | 16 |
| TEMPORAL_NUM_LAYERS | LSTM 层数 | 2 |
| TEMPORAL_DROPOUT | 时间编码器 Dropout | 0.5 |
| CROSS_ATTN_EMBED_DIM | 交叉注意力维度 | 32 |
| CROSS_ATTN_NUM_HEADS | 交叉注意力头数 | 1 |
| SPARSE_EMBED_DIM | 稀疏嵌入维度 | 32 |
| CLASSIFIER_HIDDEN_DIMS | 分类器隐藏层 | [16, 8] |
| BATCH_SIZE | 批大小 | 16 |
| EPOCHS | 训练轮数 | 50 |
| LEARNING_RATE | 学习率 | 1e-4 |

## 📈 消融实验配置

### 1. 窗口长度消融 (`ablation_window_size.py`)

测试不同时间窗口长度对模型性能的影响：
- 窗口范围：6, 8, 10, ..., 32
- 使用 4 特征输入
- 输出：性能折线图和 JSON 结果

### 2. 模块消融 (`ablation_modules.py`)

验证核心模块的贡献，支持三种配置：

| 配置 | use_cross_attention | use_sparse_representation | 说明 |
|------|---------------------|---------------------------|------|
| **Baseline** | `False` | `False` | 空间 + 时间特征直接拼接 |
| **+CrossAttn** | `True` | `False` | 添加交叉注意力融合 |
| **+Both** | `True` | `True` | 完整模型（交叉注意力 + 稀疏表示） |

### 3. 超参数敏感性分析 (`ablation_hyperparam.py`)

探究 8 个超参数对性能的影响：
- 空间/时间嵌入维度、层数、注意力头数
- Dropout 率、稀疏嵌入维度、分类器隐藏层
- 控制变量法：每次改变一个超参数，其他保持基准值

## 📁 数据预处理流程

```
原始 CSV → 加载合并 → 过滤 → 窗口化 → 数据集划分 → 归一化 → .npz
```

1. **加载合并**: 将所有测站 CSV 合并为一个 DataFrame，添加 `location` 列
2. **过滤**: 删除含 NaN 的行和无效数据（伪距残差 > 100m）
3. **窗口化**: 生成时空双通道输入
4. **数据集划分**:
   - Indomain：每个地点内随机划分 70% 训练 / 30% 测试
   - Outdomain：P2-P5、P8 训练，P6-P7 测试
5. **归一化**: 仅在训练集上拟合 scaler

### 时空双通道输入

**时间通道** `(N, window_size, features)`：
- 单颗卫星连续 window_size 个历元的观测历史

**空间通道** `(N, max_satellites, features)`：
- 同一时刻所有可见卫星的分布

**对齐方式**：
```
时间通道样本 i: [t-w+1, ..., t] 时刻的 PRN=X 观测
空间通道样本 i: t 时刻所有卫星的观测
标签 i:        t 时刻 PRN=X 的 LOS/NLOS 标志
```

## 📚 依赖

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 📖 引用

如果使用本代码，请引用原始论文：

```bibtex
@article{zeng2024stca,
  title={A Spatiotemporal Information-Driven Cross-Attention Model With Sparse Representation for GNSS NLOS Signal Detection},
  author={Zeng, et al.},
  year={2024}
}
```
