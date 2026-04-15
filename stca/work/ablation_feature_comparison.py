# ablation_feature_comparison.py
"""
消融实验：4 特征 vs 9 特征对比

4 特征（原始）:
    - C/N0
    - Elevation
    - Azimuth
    - Pseudorange_residual

9 特征（扩展）:
    - 4 个原始特征
    - Delta_CNR（CNR 一阶差分）
    - CNR_std（CNR 滑动标准差）
    - PrRes_std（伪距残差标准差）
    - Delta_Elevation（高度角变化率）
    - Delta_Azimuth（方位角变化率）

注意：默认配置为 9 特征，4 特征实验需要临时指定特征列
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 添加路径
ROOT_DIR = Path(__file__).parent.parent.parent  # DevLab 根目录
sys.path.insert(0, str(ROOT_DIR))

from utils.logger_config import setup_logger
from utils.seed_utils import set_seed

logger = setup_logger(__name__)

# 4 特征（原始特征）
BASE_FEATURES = [
    "C/N0", "Elevation", "Azimuth", "Pseudorange_residual"
]


def add_derived_features(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """添加衍生特征（9 特征模式）"""
    df = df.copy()

    # 按 location + PRN 排序
    df = df.sort_values(['location', 'PRN', 'GPS_Time(s)']).reset_index(drop=True)

    # 添加差分特征
    df["Delta_CNR"] = df.groupby(['location', 'PRN'])['C/N0'].diff().fillna(0)
    df["Delta_Elevation"] = df.groupby(['location', 'PRN'])['Elevation'].diff().fillna(0)
    df["Delta_Azimuth"] = df.groupby(['location', 'PRN'])['Azimuth'].diff().fillna(0)

    # 添加滑动标准差特征
    df["CNR_std"] = df.groupby(['location', 'PRN'])['C/N0'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    ).fillna(0)

    df["PrRes_std"] = df.groupby(['location', 'PRN'])['Pseudorange_residual'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).std()
    ).fillna(0)

    # 填充 NaN
    df = df.fillna(0)

    return df


def prepare_data(feature_cols: list, split_mode: str = None):
    """准备数据集

    Args:
        feature_cols: 特征列名列表
        split_mode: 划分模式 (indomain/outdomain)
    """
    # 添加 stca 目录到路径
    stca_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(stca_dir))

    from data_loading.windowers import WindowGenerator
    from data_loading.splitters import DataSplitter
    from data_loading.normalizers import UnifiedScaler
    from data_loading.constants import (
        DEFAULT_LOCATION_PREFIXES, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE,
        DEFAULT_RANDOM_SEED, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_SATELLITES,
        PRE_FILTER_THRESHOLD, PR_RATE_INVALID,
        OUTDOMAIN_TRAIN_LOCATIONS, OUTDOMAIN_VAL_LOCATIONS, OUTDOMAIN_TEST_LOCATIONS,
        DEFAULT_SPLIT_MODE,
    )

    # 使用配置文件中的默认 split_mode
    if split_mode is None:
        split_mode = DEFAULT_SPLIT_MODE

    script_dir = Path(__file__).parent
    # 数据目录在 stca 的父目录下
    data_dir = script_dir.parent.parent / "data for sharing_csv"

    # 加载数据
    from data_loading.loaders import CSVLoader
    from data_loading.filters import DataFilter

    loader = CSVLoader(str(data_dir), DEFAULT_LOCATION_PREFIXES)
    filter_obj = DataFilter(feature_cols, PRE_FILTER_THRESHOLD, PR_RATE_INVALID)

    df = loader.load_and_merge()
    df = filter_obj.filter_outliers(df)

    # 如果是 9 特征，添加衍生特征
    if len(feature_cols) == 9:
        logger.info("添加衍生特征（9 特征模式）")
        df = add_derived_features(df, window_size=5)

    # 生成窗口
    windower = WindowGenerator(feature_cols, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_SATELLITES)
    X_temporal, X_spatial, y, locations = windower.generate_stca_inputs(df)

    # 划分数据集
    splitter = DataSplitter(DEFAULT_RANDOM_SEED, DEFAULT_VAL_SIZE)

    # 根据 split_mode 选择划分方法
    if split_mode == "outdomain":
        # Outdomain：按地点划分
        result = splitter.split_outdomain(X_temporal, y, locations, X_spatial)
        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_spatial, X_val_spatial, X_test_spatial,
         y_train, y_val, y_test) = result
    else:
        # Indomain：每个地点内随机划分
        result = splitter.split_indomain(X_temporal, y, locations, DEFAULT_TEST_SIZE, X_spatial)
        (X_train_temporal, X_val_temporal, X_test_temporal,
         X_train_spatial, X_val_spatial, X_test_spatial,
         y_train, y_val, y_test) = result

    # 标准化（使用训练集拟合）
    scaler = UnifiedScaler.from_data(X_train_temporal, X_train_spatial)

    # 变换所有数据
    X_train_temporal = scaler.transform(X_train_temporal)
    X_val_temporal = scaler.transform(X_val_temporal)
    X_test_temporal = scaler.transform(X_test_temporal)
    X_train_spatial = scaler.transform(X_train_spatial)
    X_val_spatial = scaler.transform(X_val_spatial)
    X_test_spatial = scaler.transform(X_test_spatial)

    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)

    logger.info(f"STCA 划分 ({split_mode}) - 训练集：{n_train}, 验证集：{n_val}, 测试集：{n_test}")

    return {
        "X_train_temporal": X_train_temporal,
        "X_val_temporal": X_val_temporal,
        "X_test_temporal": X_test_temporal,
        "X_train_spatial": X_train_spatial,
        "X_val_spatial": X_val_spatial,
        "X_test_spatial": X_test_spatial,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "split_mode": split_mode,
    }


def plot_confusion_matrix(y_true, y_pred, feature_name, split_mode):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    # 计算百分比（按行归一化）
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))

    # 创建自定义注释：数值 + 百分比
    annot_labels = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f'{cm[i, j]:d}\n{cm_percent[i, j]:.1f}%'

    # 绘制热力图
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap='Blues',
                cbar_kws={'format': '%.0f%%'},
                xticklabels=['NLOS', 'LOS'], yticklabels=['NLOS', 'LOS'])

    plt.title(f'Confusion Matrix - {feature_name} ({split_mode})', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # 保存到 outputs/stca/ablation/feature 目录
    output_dir = Path(__file__).parent.parent.parent / "outputs" / "stca" / "ablation" / "feature"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"confusion_matrix_{feature_name}_{split_mode}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Confusion matrix saved to {plot_path}")
    return cm


def train_and_evaluate(feature_cols: list, use_default_config: bool = False):
    """训练并评估模型

    Args:
        feature_cols: 特征列名列表
        use_default_config: 是否使用默认配置（9 特征）
    """
    # 添加 stca 目录到路径
    stca_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(stca_dir))

    from modules.stca_model import STCAModel
    from modules.constants import (
        EPOCHS, BATCH_SIZE, LEARNING_RATE,
        SPATIAL_EMBED_DIM, SPATIAL_NUM_HEADS, SPATIAL_NUM_LAYERS, SPATIAL_DROPOUT,
        TEMPORAL_EMBED_DIM, TEMPORAL_NUM_LAYERS, TEMPORAL_DROPOUT,
        CROSS_ATTN_EMBED_DIM, CROSS_ATTN_NUM_HEADS, CROSS_ATTN_DROPOUT,
        CLASSIFIER_HIDDEN_DIMS, CLASSIFIER_DROPOUT,
    )
    from data_loading.constants import DEFAULT_SPLIT_MODE

    logger.info(f"\n{'='*60}")
    logger.info(f"使用特征：{feature_cols}")
    logger.info(f"特征数量：{len(feature_cols)}")
    logger.info(f"{'='*60}\n")

    # 准备数据
    data = prepare_data(feature_cols)

    X_train_spatial = data["X_train_spatial"]
    X_val_spatial = data["X_val_spatial"]
    X_test_spatial = data["X_test_spatial"]
    X_train_temporal = data["X_train_temporal"]
    X_val_temporal = data["X_val_temporal"]
    X_test_temporal = data["X_test_temporal"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    logger.info(f"训练集空间输入形状：{X_train_spatial.shape}")
    logger.info(f"训练集时间输入形状：{X_train_temporal.shape}")

    # 根据特征数量动态调整 LSTM 层数
    # 9 特征（默认配置）使用 2 层 LSTM，4 特征使用 1 层
    if use_default_config:
        temporal_num_layers = TEMPORAL_NUM_LAYERS  # 默认 2 层
    else:
        temporal_num_layers = 1  # 4 特征用 1 层
    logger.info(f"使用 LSTM 层数：{temporal_num_layers}")

    # 创建模型
    set_seed(42)
    model = STCAModel(
        input_dim=len(feature_cols),
        num_classes=2,
        spatial_embed_dim=SPATIAL_EMBED_DIM,
        spatial_num_heads=SPATIAL_NUM_HEADS,
        spatial_num_layers=SPATIAL_NUM_LAYERS,
        spatial_dropout=SPATIAL_DROPOUT,
        temporal_embed_dim=TEMPORAL_EMBED_DIM,
        temporal_num_layers=temporal_num_layers,
        temporal_dropout=TEMPORAL_DROPOUT,
        cross_attn_embed_dim=CROSS_ATTN_EMBED_DIM,
        cross_attn_num_heads=CROSS_ATTN_NUM_HEADS,
        cross_attn_dropout=CROSS_ATTN_DROPOUT,
        classifier_hidden_dims=CLASSIFIER_HIDDEN_DIMS,
        classifier_dropout=CLASSIFIER_DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量：{total_params:,}")

    # 训练
    history = model.fit(
        X_train_spatial, y_train,
        X_val_spatial=X_val_spatial, y_val=y_val,
        X_train_temporal=X_train_temporal,
        X_val_temporal=X_val_temporal,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device='cpu',
        verbose=True,
    )

    # 评估
    metrics = model.evaluate(
        X_test_spatial, y_test,
        X_test_3d=X_test_temporal,
        device='cpu',
    )

    logger.info(f"\n测试集结果:")
    logger.info(f"  Loss: {metrics.get('loss', 'N/A')}")
    logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"  F1: {metrics['f1_score']*100:.2f}%")

    # 绘制混淆矩阵
    feature_name = "4feat" if len(feature_cols) == 4 else "9feat"
    y_true = y_test
    y_pred = metrics['y_pred']

    plot_confusion_matrix(
        y_true, y_pred,
        feature_name=feature_name,
        split_mode=data.get('split_mode', 'outdomain')
    )

    return {
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "history": history,
        "test_metrics": metrics,
    }


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("消融实验：4 特征 vs 9 特征对比")
    logger.info("="*60)

    # 添加 stca 目录到路径
    stca_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(stca_dir))

    # 从 data_loading.constants 导入 9 特征配置
    from data_loading.constants import DEFAULT_FEATURE_COLS as FEATURES_9

    results = {}

    # 实验 1: 4 特征（原始）
    logger.info("\n\n>>> 实验 1: 4 特征（原始）")
    results["4_features"] = train_and_evaluate(
        feature_cols=BASE_FEATURES,
        use_default_config=False,
    )

    # 实验 2: 9 特征（默认配置）
    logger.info("\n\n>>> 实验 2: 9 特征（默认配置）")
    results["9_features"] = train_and_evaluate(
        feature_cols=FEATURES_9,
        use_default_config=True,
    )

    # 输出对比结果
    logger.info("\n" + "="*60)
    logger.info("消融实验结果对比")
    logger.info("="*60)

    for name, result in results.items():
        metrics = result["test_metrics"]
        logger.info(f"\n{name}:")
        logger.info(f"  特征数量：{result['num_features']}")
        logger.info(f"  测试集 Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f"  测试集 F1: {metrics['f1_score']*100:.2f}%")

    # 计算提升
    if "4_features" in results and "9_features" in results:
        acc_4 = results["4_features"]["test_metrics"]["accuracy"]
        acc_9 = results["9_features"]["test_metrics"]["accuracy"]
        improvement = (acc_9 - acc_4) / acc_4 * 100

        f1_4 = results["4_features"]["test_metrics"]["f1_score"]
        f1_9 = results["9_features"]["test_metrics"]["f1_score"]
        f1_improvement = (f1_9 - f1_4) / f1_4 * 100

        logger.info(f"\n性能提升:")
        logger.info(f"  Accuracy 提升：{improvement:.2f}%")
        logger.info(f"  F1 提升：{f1_improvement:.2f}%")

        if improvement > 0:
            logger.info(f"\n结论：扩展特征有效，推荐使用 9 特征")
        else:
            logger.info(f"\n结论：扩展特征无明显提升，使用 4 特征即可")

    logger.info("\n" + "="*60)
    logger.info("消融实验完成")
    logger.info("="*60)

    return results


if __name__ == "__main__":
    main()
