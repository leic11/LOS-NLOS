# ablation_hyperparam.py
"""
超参数敏感性分析实验
============================

用途：
    探究空间模块和时间模块关键超参数对 GNSS NLOS 检测模型性能的影响。

实验设置：
    - 使用 outdomain 数据划分模式
    - 控制变量法：每次只改变一个超参数，其他保持基准值不变
    - 基准配置只测试 1 次，每个超参数测试 2 个变化值
    - 总计测试 11 次 = 1 次基准 + 5 个超参数 × 2 个变化值

基准配置（对标论文 Table III 基础模型）：
    空间模块（AAM）：
    - SPATIAL_EMBED_DIM = 32       (空间嵌入维度)
    - SPATIAL_NUM_HEADS = 1        (注意力头数)
    - SPATIAL_NUM_LAYERS = 1       (空间编码器层数)
    - DROPOUT = 0.1                (统一 Dropout，应用到所有模块)

    时间模块（LSTM-TFE）：
    - TEMPORAL_EMBED_DIM = 32      (时间嵌入维度)

    固定参数（不测试）：
    - SPATIAL_D_FF = 64            (前馈网络维度，embed_dim * 2)
    - TEMPORAL_NUM_LAYERS = 4      (LSTM 层数)
    - CROSS_ATTN_EMBED_DIM = 16    (交叉注意力维度)
    - CROSS_ATTN_NUM_HEADS = 4     (交叉注意力头数)
    - SPARSE_EMBED_DIM = 64        (稀疏嵌入维度)
    - CLASSIFIER_HIDDEN_DIMS = [16, 8]  (分类器隐藏层)

    训练参数（固定，不测试）：
    - LEARNING_RATE = 1e-4
    - BATCH_SIZE = 16
    - EPOCHS = 50

测试的超参数及取值（5 个）：
    1. 空间嵌入维度：16, 64（基准 32）
    2. 注意力头数：2, 4（基准 1）
    3. 空间层数：2, 4（基准 1）
    4. Dropout 率：0.3, 0.5（基准 0.1）
    5. 时间嵌入维度：16, 64（基准 32）

输出表格格式（7 行）：
    - 6 列：5 个超参数列 + 5 个性能指标列（Loss/Acc/Pre/Rec/F1）
    - 第 1 行：基准配置 + 5 大指标 + 参数量
    - 第 2-11 行：5 个超参数 × 2 个变化值 = 10 行
    - 第 11 行：最优组合汇总（F1 提升）

输出：
    - 终端：汇总表格
    - 文件：hyperparam_study_results.csv/json（详细结果）

使用方式：
    python -m work.ablation_hyperparam
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODULES_DIR = Path(__file__).parent / "modules"
sys.path.insert(0, str(MODULES_DIR))

STCA_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(STCA_DIR))

from modules.constants import DEVICE
from utils.logger_config import setup_logger
from utils.seed_utils import set_seed
from modules.stca_model import STCAModel
from data_loading.main import StaticPreprocessor
from data_loading.constants import DEFAULT_MAX_SATELLITES, DEFAULT_WINDOW_SIZE

logger = setup_logger(__name__)

# ============================================================================
# 基准配置（对标论文 Table III 基础模型 M1）
# ============================================================================

# 基准配置（硬编码，对标论文设置）
BASE_CONFIG = {
    "spatial_embed_dim": 32,         # 空间嵌入维度
    "spatial_num_heads": 1,          # 注意力头数
    "spatial_num_layers": 1,         # 空间编码器层数
    "spatial_d_ff": 64,              # 前馈网络维度，embed_dim * 2
    "dropout": 0.1,                  # 统一 Dropout
    "temporal_embed_dim": 32,        # 时间嵌入维度
    "temporal_num_layers": 1,        # LSTM 层数
    "cross_attn_embed_dim": 16,      # 交叉注意力维度
    "cross_attn_num_heads": 4,       # 交叉注意力头数
    "sparse_embed_dim": 64,          # 稀疏嵌入维度
    "classifier_hidden_dims": [16, 8],  # 分类器隐藏层
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 50,
    "random_seed": 42,
    "max_satellites": DEFAULT_MAX_SATELLITES,
    "window_size": DEFAULT_WINDOW_SIZE,
    "split_mode": "outdomain",
    "input_dim": 4,  # 4 特征
    "num_classes": 2,
}

# ============================================================================
# 超参数测试配置（每组 2 个测试值，基准值单独测试 1 次）
# ============================================================================

HYPERPARAM_CONFIGS = {
    "spatial_embed_dim": {
        "name": "空间嵌入维度",
        "values": [16, 64],  # 基准 32
    },
    "spatial_num_heads": {
        "name": "注意力头数",
        "values": [2, 4],  # 基准 1
    },
    "spatial_num_layers": {
        "name": "空间层数",
        "values": [2, 4],  # 基准 1
    },
    "dropout": {
        "name": "Dropout 率",
        "values": [0.3, 0.5],  # 基准 0.1
    },
    "temporal_embed_dim": {
        "name": "时间嵌入维度",
        "values": [16, 64],  # 基准 32
    },
}

# 5 个测试的超参数键（用于表格列显示）
HYPERPARAM_KEYS = list(HYPERPARAM_CONFIGS.keys())

# ============================================================================
# 输出目录
# ============================================================================

OUTPUT_DIR = ROOT_DIR / "outputs" / "stca" / "ablation" / "hyperparam"


def load_or_preprocess_data(split_mode, window_size):
    """加载或预处理数据"""
    npz_filename = f"static_processed_{split_mode}.npz"
    npz_path = STCA_DIR / npz_filename

    if npz_path.exists():
        logger.info(f"Loading data from {npz_path}")
        data = StaticPreprocessor.load_processed(str(npz_path))
    else:
        logger.info(f"Preprocessing data with split_mode={split_mode}...")
        data_dir = STCA_DIR.parent / "data for sharing_csv"
        preprocessor = StaticPreprocessor(data_dir=str(data_dir))
        data = preprocessor.process_stca(
            window_size=window_size,
            split_mode=split_mode,
        )
        preprocessor.save_processed(data, str(npz_path))
        logger.info(f"Data saved to {npz_path}")

    return data


def train_and_evaluate(config, param_name, param_value):
    """训练并评估单个超参数配置"""
    logger.info(f"\n{'='*60}")
    logger.info(f"超参数：{param_name} | 测试值：{param_value}")
    logger.info(f"{'='*60}")

    # 设置随机种子
    set_seed(config["random_seed"])

    # 加载数据（只加载一次，复用）
    data = load_or_preprocess_data(config["split_mode"], config["window_size"])

    X_train_spatial = data["X_train_spatial"]
    X_train_temporal = data["X_train_temporal"]
    y_train = data["y_train"]
    X_test_spatial = data["X_test_spatial"]
    X_test_temporal = data["X_test_temporal"]
    y_test = data["y_test"]

    logger.info(f"Data: Train={len(y_train)}, Test={len(y_test)}")

    # 始终保持 spatial_d_ff = spatial_embed_dim * 2
    config["spatial_d_ff"] = config["spatial_embed_dim"] * 2

    # 构建模型
    model = STCAModel(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        spatial_embed_dim=config["spatial_embed_dim"],
        spatial_num_heads=config["spatial_num_heads"],
        spatial_num_layers=config["spatial_num_layers"],
        spatial_d_ff=config["spatial_d_ff"],
        spatial_dropout=config["dropout"],        # 统一 dropout
        temporal_embed_dim=config["temporal_embed_dim"],
        temporal_num_layers=config["temporal_num_layers"],
        temporal_dropout=config["dropout"],        # 统一 dropout
        cross_attn_embed_dim=config["cross_attn_embed_dim"],
        cross_attn_num_heads=config["cross_attn_num_heads"],
        cross_attn_dropout=config["dropout"],      # 统一 dropout
        sparse_embed_dim=config["sparse_embed_dim"],
        classifier_hidden_dims=config["classifier_hidden_dims"],
        classifier_dropout=config["dropout"],      # 统一 dropout
    )

    logger.info(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    logger.info(f"Using device: {DEVICE}")

    # 训练（使用 config 中的学习率、batch_size、epochs）
    logger.info(f"Training for {config['epochs']} epochs...")
    history = model.fit(
        X_train_spatial, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        device=DEVICE,
        verbose=False,
        X_train_temporal=X_train_temporal,
    )

    # 评估
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(
        X_test_spatial, y_test,
        device=DEVICE,
        X_test_3d=X_test_temporal,
    )

    # 获取最终 epoch 的训练损失
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0

    result = {
        "param_value": str(param_value),
        "test_loss": float(metrics.get("loss", final_train_loss)),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        # 保存 10 个超参数的值
        "spatial_embed_dim": config["spatial_embed_dim"],
        "spatial_num_heads": config["spatial_num_heads"],
        "spatial_num_layers": config["spatial_num_layers"],
        "dropout": config["dropout"],
        "temporal_embed_dim": config["temporal_embed_dim"],
        "temporal_num_layers": config["temporal_num_layers"],
        "cross_attn_embed_dim": config["cross_attn_embed_dim"],
        "cross_attn_num_heads": config["cross_attn_num_heads"],
        "sparse_embed_dim": config["sparse_embed_dim"],
        "classifier_hidden_dims": str(config["classifier_hidden_dims"]),
    }

    # 保存 10 个超参数的值（用于表格显示）
    # 1. spatial_embed_dim, 2. temporal_embed_dim, 3. spatial_num_layers
    # 4. dropout, 5. temporal_num_layers, 6. sparse_embed_dim
    # 7. classifier_hidden_dims, 8. spatial_num_heads

    logger.info(f"Result: ACC={result['accuracy']:.4f}, PRE={result['precision']:.4f}, "
               f"REC={result['recall']:.4f}, F1={result['f1_score']:.4f}")

    return result


def try_load_results_from_cache():
    """尝试从缓存文件加载所有结果"""
    all_results = []

    # 基准结果
    baseline_path = OUTPUT_DIR / "result_baseline.json"
    if baseline_path.exists():
        with open(baseline_path, 'r', encoding='utf-8-sig') as f:
            baseline = json.load(f)
        # 强制设置正确的中文名称
        baseline["param_name"] = "基准配置"
        baseline["param_key"] = "baseline"
        baseline["is_baseline"] = True
        all_results.append(baseline)
        logger.info(f"已加载基准结果：ACC={baseline['accuracy']:.4f}")
    else:
        return None

    # 其他超参数结果
    for param_key, param_config in HYPERPARAM_CONFIGS.items():
        for value in param_config["values"]:
            result_path = OUTPUT_DIR / f"result_{param_key}_{str(value)}.json"
            if result_path.exists():
                with open(result_path, 'r', encoding='utf-8-sig') as f:
                    result = json.load(f)
                # 强制从配置中获取正确的中文名称（覆盖可能损坏的值）
                result["param_name"] = param_config["name"]
                result["param_key"] = param_key
                result["is_baseline"] = False
                all_results.append(result)
                logger.info(f"已加载 {param_config['name']}={value}: ACC={result['accuracy']:.4f}")
            else:
                return None  # 有任何一个缺失就返回 None

    if len(all_results) == 1 + sum(len(p["values"]) for p in HYPERPARAM_CONFIGS.values()):
        logger.info(f"\n检测到 {len(all_results)} 个缓存结果文件，直接从缓存加载！")
        return all_results
    return None


def run_hyperparam_study():
    """执行超参数敏感性分析"""
    logger.info("="*60)
    logger.info("超参数敏感性分析实验")
    logger.info("="*60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先尝试从缓存加载
    all_results = try_load_results_from_cache()

    if all_results is not None:
        logger.info("\n使用缓存结果生成汇总表格，无需重新训练！\n")
        baseline_result = next(r for r in all_results if r["is_baseline"])
    else:
        logger.info("\n缓存结果不完整，开始执行训练实验...\n")
        all_results = []

        # 1. 先测试一次完整基准配置（只测试 1 次）
        logger.info(f"\n{'#'*60}")
        logger.info("# 测试基准配置（所有超参数默认值）")
        logger.info(f"{'#'*60}")

        logger.info(f"\n>>> 测试 基准配置 ...")
        baseline_result = train_and_evaluate(BASE_CONFIG, "基准配置", "baseline")
        baseline_result["param_name"] = "基准配置"
        baseline_result["param_key"] = "baseline"
        baseline_result["is_baseline"] = True
        all_results.append(baseline_result)

        # 保存基准结果
        result_path = OUTPUT_DIR / "result_baseline.json"
        with open(result_path, 'w') as f:
            json.dump(baseline_result, f, indent=2, ensure_ascii=False)

        # 2. 遍历每个超参数，只测试变化值
        for param_key, param_config in HYPERPARAM_CONFIGS.items():
            param_name = param_config["name"]
            test_values = param_config["values"]

            logger.info(f"\n{'#'*60}")
            logger.info(f"# 测试超参数：{param_name} ({param_key})")
            logger.info(f"# 测试值：{test_values}")
            logger.info(f"# 基准值：{BASE_CONFIG[param_key]}（已测试，复用上方结果）")
            logger.info(f"{'#'*60}")

            for value in test_values:
                # 复制基准配置并修改当前超参数
                config = BASE_CONFIG.copy()
                config[param_key] = value

                logger.info(f"\n>>> 测试 {param_name} = {value} ...")

                result = train_and_evaluate(config, param_name, value)
                result["param_name"] = param_name
                result["param_key"] = param_key
                result["is_baseline"] = False

                all_results.append(result)

                # 保存单个结果
                result_path = OUTPUT_DIR / f"result_{param_key}_{str(value)}.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

    # 生成汇总表格
    logger.info("\n" + "="*60)
    logger.info("汇总表格")
    logger.info("="*60)

    # 创建 DataFrame
    df = pd.DataFrame(all_results)

    # 打印表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    # 输出到终端
    print("\n" + "="*160)
    print("超参数敏感性分析结果汇总")
    print("="*160)

    # 获取基准结果
    baseline_f1 = baseline_result["f1_score"]
    baseline_acc = baseline_result["accuracy"]
    baseline_pre = baseline_result["precision"]
    baseline_rec = baseline_result["recall"]
    baseline_loss = baseline_result["test_loss"]

    # 计算基准参数量
    baseline_params = sum(p.numel() for p in STCAModel(
        input_dim=BASE_CONFIG["input_dim"],
        num_classes=BASE_CONFIG["num_classes"],
        spatial_embed_dim=BASE_CONFIG["spatial_embed_dim"],
        spatial_num_heads=BASE_CONFIG["spatial_num_heads"],
        spatial_num_layers=BASE_CONFIG["spatial_num_layers"],
        spatial_d_ff=BASE_CONFIG["spatial_d_ff"],
        spatial_dropout=BASE_CONFIG["dropout"],
        temporal_embed_dim=BASE_CONFIG["temporal_embed_dim"],
        temporal_num_layers=BASE_CONFIG["temporal_num_layers"],
        temporal_dropout=BASE_CONFIG["dropout"],
        cross_attn_embed_dim=BASE_CONFIG["cross_attn_embed_dim"],
        cross_attn_num_heads=BASE_CONFIG["cross_attn_num_heads"],
        cross_attn_dropout=BASE_CONFIG["dropout"],
        classifier_hidden_dims=BASE_CONFIG["classifier_hidden_dims"],
        classifier_dropout=BASE_CONFIG["dropout"],
    ).parameters())

    # 表格列宽定义
    COL_SPATIAL_EMBED = 8
    COL_SPATIAL_HEADS = 6
    COL_SPATIAL_LAYERS = 6
    COL_DROPOUT = 6
    COL_TEMPORAL_EMBED = 8

    # 表格头部（5 个超参数列 + 5 个性能指标列 = 10 列）
    header = (f"{'模型配置':<15} | "
              f"{'s_embed':>{COL_SPATIAL_EMBED}} | "
              f"{'s_head':>{COL_SPATIAL_HEADS}} | "
              f"{'s_lyr':>{COL_SPATIAL_LAYERS}} | "
              f"{'drop':>{COL_DROPOUT}} | "
              f"{'t_embed':>{COL_TEMPORAL_EMBED}} | "
              f"Loss   | Acc    | Pre    | Rec    | F1")
    print(header)
    print("-"*130)

    # 辅助函数：格式化参数值（不变的显示"-"）
    def fmt_param(value, baseline_value, width=6):
        """格式化参数值，与基准值相同则显示 '-'，否则右对齐显示值"""
        if str(value) == str(baseline_value):
            return "-" * width
        else:
            return str(value).rjust(width)

    # 第 1 行：基准配置（显示所有基准值）
    baseline_row = (f"{'[基准配置]':<15} | "
                   f"{str(BASE_CONFIG['spatial_embed_dim']):>{COL_SPATIAL_EMBED}} | "
                   f"{str(BASE_CONFIG['spatial_num_heads']):>{COL_SPATIAL_HEADS}} | "
                   f"{str(BASE_CONFIG['spatial_num_layers']):>{COL_SPATIAL_LAYERS}} | "
                   f"{str(BASE_CONFIG['dropout']):>{COL_DROPOUT}} | "
                   f"{str(BASE_CONFIG['temporal_embed_dim']):>{COL_TEMPORAL_EMBED}} | "
                   f"{baseline_loss:<6.4f} {baseline_acc:<6.4f} {baseline_pre:<6.4f} {baseline_rec:<6.4f} {baseline_f1:<6.4f}")
    print(baseline_row)
    print(f"  参数量：{baseline_params:,}")
    print("-"*130)

    # 第 2-11 行：每个超参数的变化值（5 个超参数 × 2 个值 = 10 行）
    for param_key, param_config in HYPERPARAM_CONFIGS.items():
        param_results = df[df["param_key"] == param_key]
        for _, row in param_results.iterrows():
            spatial_embed = fmt_param(row['spatial_embed_dim'], BASE_CONFIG['spatial_embed_dim'], COL_SPATIAL_EMBED)
            spatial_heads = fmt_param(row['spatial_num_heads'], BASE_CONFIG['spatial_num_heads'], COL_SPATIAL_HEADS)
            spatial_layers = fmt_param(row['spatial_num_layers'], BASE_CONFIG['spatial_num_layers'], COL_SPATIAL_LAYERS)
            dropout = fmt_param(row['dropout'], BASE_CONFIG['dropout'], COL_DROPOUT)
            temporal_embed = fmt_param(row['temporal_embed_dim'], BASE_CONFIG['temporal_embed_dim'], COL_TEMPORAL_EMBED)

            row_text = (f"{param_config['name']:<15} | "
                       f"{spatial_embed} | "
                       f"{spatial_heads} | "
                       f"{spatial_layers} | "
                       f"{dropout} | "
                       f"{temporal_embed} | "
                       f"{row['test_loss']:<6.4f} {row['accuracy']:<6.4f} {row['precision']:<6.4f} {row['recall']:<6.4f} {row['f1_score']:<6.4f}")
            print(row_text)

    # 第 11 行：最优配置汇总（找出 F1 最高的配置，包括基准配置）
    print("-"*130)

    # 找出 F1 最高的配置（包括基准配置）
    best_idx = df["f1_score"].idxmax()
    best_row = df.loc[best_idx]
    improvement = best_row["f1_score"] - baseline_f1

    # 判断是否是基准配置
    is_best_baseline = best_row.get("is_baseline", False)

    if is_best_baseline:
        # 最优就是基准配置，显示基准值
        best_spatial_embed = BASE_CONFIG["spatial_embed_dim"]
        best_spatial_heads = BASE_CONFIG["spatial_num_heads"]
        best_spatial_layers = BASE_CONFIG["spatial_num_layers"]
        best_dropout = BASE_CONFIG["dropout"]
        best_temporal_embed = BASE_CONFIG["temporal_embed_dim"]
        print(f"{'[最优组合]':<15} | {str(best_spatial_embed):>{COL_SPATIAL_EMBED}} | {str(best_spatial_heads):>{COL_SPATIAL_HEADS}} | {str(best_spatial_layers):>{COL_SPATIAL_LAYERS}} | {str(best_dropout):>{COL_DROPOUT}} | {str(best_temporal_embed):>{COL_TEMPORAL_EMBED}} | "
              f"{baseline_loss:<6.4f} {baseline_acc:<6.4f} {baseline_pre:<6.4f} {baseline_rec:<6.4f} {baseline_f1:<6.4f}")
        print(f"  基准配置即为最优，无提升 ({improvement:+.4f})")
    else:
        # 最优是某个测试配置，显示其参数值
        best_spatial_embed = int(best_row['spatial_embed_dim'])
        best_spatial_heads = int(best_row['spatial_num_heads'])
        best_spatial_layers = int(best_row['spatial_num_layers'])
        best_dropout = float(best_row['dropout'])
        best_temporal_embed = int(best_row['temporal_embed_dim'])

        print(f"{'[最优组合]':<15} | {str(best_spatial_embed):>{COL_SPATIAL_EMBED}} | {str(best_spatial_heads):>{COL_SPATIAL_HEADS}} | {str(best_spatial_layers):>{COL_SPATIAL_LAYERS}} | {str(best_dropout):>{COL_DROPOUT}} | {str(best_temporal_embed):>{COL_TEMPORAL_EMBED}} | "
              f"{best_row['test_loss']:<6.4f} {best_row['accuracy']:<6.4f} {best_row['precision']:<6.4f} {best_row['recall']:<6.4f} {best_row['f1_score']:<6.4f}")
        print(f"  相比基准：F1 提升 {improvement:+.4f} ({improvement/baseline_f1*100:+.2f}%)")

    # 保存为 CSV
    csv_path = OUTPUT_DIR / "hyperparam_study_results.csv"
    try:
        df.to_csv(csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        logger.info(f"Results saved to {csv_path}")
    except PermissionError:
        logger.warning(f"无法保存 {csv_path}，文件可能已被 Excel 或其他程序打开，跳过保存。")

    # 保存为 JSON
    json_path = OUTPUT_DIR / "hyperparam_study_results.json"
    try:
        with open(json_path, 'w', encoding='utf-8-sig') as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Results saved to {json_path}")
    except PermissionError:
        logger.warning(f"无法保存 {json_path}，文件可能已被打开，跳过保存。")

    # 保存汇总表（每个超参数最佳值）添加到 hyperparam_study_results.csv 的最后一行
    # 注意：基准值也参与比较，因为基准值可能本身就是最优的
    summary_rows = []
    for param_key, param_config in HYPERPARAM_CONFIGS.items():
        param_results = df[df["param_key"] == param_key]

        # 添加基准值到比较中
        baseline_row_for_param = {
            "param_value": str(BASE_CONFIG[param_key]),
            "f1_score": baseline_f1,
            "is_baseline": True,
        }

        # 合并测试值和基准值，找出最优
        all_values_for_param = [baseline_row_for_param]
        for _, row in param_results.iterrows():
            all_values_for_param.append({
                "param_value": row["param_value"],
                "f1_score": row["f1_score"],
                "is_baseline": False,
            })

        # 找出 F1 最高的配置
        best = max(all_values_for_param, key=lambda x: x["f1_score"])
        improvement = best["f1_score"] - baseline_f1

        summary_rows.append({
            "param_key": f"[最优] {param_config['name']}",
            "param_value": best["param_value"],
            "test_loss": float('nan'),
            "accuracy": float('nan'),
            "precision": float('nan'),
            "recall": float('nan'),
            "f1_score": best["f1_score"],
            "spatial_embed_dim": float('nan'),
            "spatial_num_heads": float('nan'),
            "spatial_num_layers": float('nan'),
            "dropout": float('nan'),
            "temporal_embed_dim": float('nan'),
            "temporal_num_layers": float('nan'),
            "cross_attn_embed_dim": float('nan'),
            "cross_attn_num_heads": float('nan'),
            "sparse_embed_dim": float('nan'),
            "classifier_hidden_dims": best["param_value"] if param_key == "classifier_hidden_dims" else float('nan'),
            "is_baseline": False,
        })

    # 将最优值汇总添加到主 DataFrame
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        df = pd.concat([df, summary_df], ignore_index=True)

    # 保存为 CSV（包含最优值汇总行）
    csv_path = OUTPUT_DIR / "hyperparam_study_results.csv"
    try:
        df.to_csv(csv_path, index=False, float_format="%.4f", encoding='utf-8-sig')
        logger.info(f"Results saved to {csv_path}")
    except PermissionError:
        logger.warning(f"无法保存 {csv_path}，文件可能已被 Excel 或其他程序打开，跳过保存。")

    logger.info("\n超参数敏感性分析实验完成！")

    return all_results


if __name__ == "__main__":
    run_hyperparam_study()
