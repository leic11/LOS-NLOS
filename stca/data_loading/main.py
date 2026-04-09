# main.py
"""
STCA 数据预处理器 - 重构版（外观模式）

本模块通过组合更小、更专一的功能模块来编排数据预处理流程。
"""
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# 将项目根目录（DevLab）添加到 sys.path，支持导入 utils 模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from .constants import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_LOCATION_PREFIXES, DEFAULT_SPLIT_MODE, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE,
    DEFAULT_RANDOM_SEED, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_SATELLITES,
    PRE_FILTER_THRESHOLD, PR_RATE_INVALID,
    OUTDOMAIN_TRAIN_LOCATIONS, OUTDOMAIN_VAL_LOCATIONS, OUTDOMAIN_TEST_LOCATIONS,
)
from .loaders import CSVLoader
from .filters import DataFilter
from .windowers import WindowGenerator
from .splitters import DataSplitter
from .normalizers import UnifiedScaler
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class StaticPreprocessor:
    """STCA 数据预处理器（外观模式）"""

    def __init__(
        self,
        data_dir: str,
        feature_cols: List[str] = DEFAULT_FEATURE_COLS,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.data_dir = data_dir
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        # 组合各模块
        self.loader = CSVLoader(data_dir, DEFAULT_LOCATION_PREFIXES)
        self.filter = DataFilter(self.feature_cols, PRE_FILTER_THRESHOLD, PR_RATE_INVALID)
        self.windower = WindowGenerator(
            self.feature_cols,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_MAX_SATELLITES
        )
        self.splitter = DataSplitter(random_seed, val_size)
        self.scaler: Optional[UnifiedScaler] = None

    def load_and_merge_csv(self) -> pd.DataFrame:
        """委托给 CSVLoader 加载并合并数据"""
        return self.loader.load_and_merge()

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """委托给 DataFilter 过滤异常值"""
        return self.filter.filter_outliers(df)

    def _get_split_indices(
        self,
        locations: np.ndarray,
        y: np.ndarray,
        split_mode: str
    ) -> Dict[str, np.ndarray]:
        """获取训练集/验证集/测试集的划分索引"""
        if split_mode == "indomain":
            # Indomain：每个地点内随机划分
            # 使用全 1 作为虚拟特征，因为 split_indomain 只需要索引划分
            X_dummy = np.ones((len(locations), 1))

            X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split_indomain(
                X_dummy, y, locations, self.test_size
            )

            # 需要从 locations 和 y 中获取实际索引
            # 重新执行划分逻辑来获取索引
            unique_locations = np.unique(locations)
            train_indices = []
            test_indices = []

            for loc in unique_locations:
                loc_mask = locations == loc
                loc_indices = np.where(loc_mask)[0]
                n_samples = len(loc_indices)
                n_test = int(n_samples * self.test_size)

                np.random.seed(self.splitter.random_seed)
                shuffled = np.random.permutation(n_samples)

                test_idx = loc_indices[shuffled[:n_test]]
                train_val_idx = loc_indices[shuffled[n_test:]]

                train_indices.extend(train_val_idx)
                test_indices.extend(test_idx)

            # 从训练集中划分验证集
            train_indices = np.array(train_indices)
            n_val = int(len(train_indices) * self.splitter.val_size)
            np.random.seed(self.splitter.random_seed)
            shuffled = np.random.permutation(len(train_indices))

            val_idx = train_indices[shuffled[:n_val]]
            train_idx = train_indices[shuffled[n_val:]]

            return {
                "train": train_idx,
                "val": val_idx,
                "test": np.array(test_indices)
            }
        else:
            # Outdomain：按地点划分，返回布尔掩码
            return {
                "train": np.isin(locations, OUTDOMAIN_TRAIN_LOCATIONS),
                "val": np.isin(locations, OUTDOMAIN_VAL_LOCATIONS),
                "test": np.isin(locations, OUTDOMAIN_TEST_LOCATIONS)
            }

    def process_stca(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_satellites: int = DEFAULT_MAX_SATELLITES,
        split_mode: str = DEFAULT_SPLIT_MODE,
    ) -> Dict:
        """执行完整的预处理流程"""
        # 1. 加载数据
        df = self.loader.load_and_merge()

        # 2. 过滤异常值
        df = self.filter.filter_outliers(df)

        # 3. 生成 STCA 输入（时间通道 + 空间通道）
        X_temporal, X_spatial, y, locations = self.windower.generate_stca_inputs(df)

        # 4. 划分数据集
        indices = self._get_split_indices(locations, y, split_mode)

        # 5. 标准化（仅使用训练集拟合）
        self.scaler = UnifiedScaler.from_data(
            X_temporal[indices["train"]],
            X_spatial[indices["train"]]
        )

        # 6. 变换所有数据
        X_train_temporal = self.scaler.transform(X_temporal[indices["train"]])
        X_val_temporal = self.scaler.transform(X_temporal[indices["val"]])
        X_test_temporal = self.scaler.transform(X_temporal[indices["test"]])

        X_train_spatial = self.scaler.transform(X_spatial[indices["train"]])
        X_val_spatial = self.scaler.transform(X_spatial[indices["val"]])
        X_test_spatial = self.scaler.transform(X_spatial[indices["test"]])

        y_train = y[indices["train"]]
        y_val = y[indices["val"]]
        y_test = y[indices["test"]]

        logger.info(
            f"STCA 划分 - 训练集：{len(indices['train'])}, "
            f"验证集：{len(indices['val'])}, 测试集：{len(indices['test'])}"
        )

        # 7. 确定地点信息
        if split_mode == "indomain":
            train_locations = val_locations = test_locations = "indomain"
        else:
            train_locations = OUTDOMAIN_TRAIN_LOCATIONS
            val_locations = OUTDOMAIN_VAL_LOCATIONS
            test_locations = OUTDOMAIN_TEST_LOCATIONS

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
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "split_mode": split_mode,
            "train_locations": train_locations,
            "val_locations": val_locations,
            "test_locations": test_locations,
            "window_size": window_size,
            "max_satellites": max_satellites,
        }

    def save_processed(self, data: Dict, output_path: str) -> None:
        """将处理后的数据保存为 .npz 文件"""
        scaler = data.get("scaler")

        save_dict = {
            "X_train": data.get("X_train_temporal"),
            "X_val": data.get("X_val_temporal"),
            "X_test": data.get("X_test_temporal"),
            "y_train": data["y_train"],
            "y_val": data["y_val"],
            "y_test": data["y_test"],
            "scaler_mean": scaler.mean_ if scaler else None,
            "scaler_scale": scaler.scale_ if scaler else None,
            "feature_cols": np.array(data["feature_cols"]),
            "split_mode": data.get("split_mode", "random"),
            "window_size": data.get("window_size", DEFAULT_WINDOW_SIZE),
            "max_satellites": data.get("max_satellites", DEFAULT_MAX_SATELLITES),
            "X_train_spatial": data.get("X_train_spatial"),
            "X_val_spatial": data.get("X_val_spatial"),
            "X_test_spatial": data.get("X_test_spatial"),
        }

        np.savez(output_path, **save_dict)
        logger.info(f"已保存处理后的数据到 {output_path}")

    @staticmethod
    def load_processed(npz_path: str) -> Dict:
        """从 .npz 文件加载处理后的数据"""
        loader = np.load(npz_path, allow_pickle=True)

        scaler_mean = loader.get("scaler_mean")
        scaler = None
        if scaler_mean is not None:
            scaler = UnifiedScaler(
                means=scaler_mean,
                stds=loader["scaler_scale"],
            )

        return {
            "X_train_temporal": loader["X_train"],
            "X_val_temporal": loader["X_val"],
            "X_test_temporal": loader["X_test"],
            "y_train": loader["y_train"],
            "y_val": loader["y_val"],
            "y_test": loader["y_test"],
            "scaler": scaler,
            "feature_cols": list(loader["feature_cols"]),
            "split_mode": str(loader.get("split_mode", "random")),
            "window_size": int(loader.get("window_size", DEFAULT_WINDOW_SIZE)),
            "max_satellites": int(loader.get("max_satellites", DEFAULT_MAX_SATELLITES)),
            "X_train_spatial": loader.get("X_train_spatial"),
            "X_val_spatial": loader.get("X_val_spatial"),
            "X_test_spatial": loader.get("X_test_spatial"),
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="STCA 数据预处理器")
    parser.add_argument(
        "--split-mode",
        type=str,
        default=DEFAULT_SPLIT_MODE,
        choices=["indomain", "outdomain"],
        help="数据划分模式：indomain（内域）或 outdomain（跨域）"
    )
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="时间窗口大小"
    )
    parser.add_argument(
        "--max-satellites",
        type=int,
        default=DEFAULT_MAX_SATELLITES,
        help="最大卫星数"
    )
    args = parser.parse_args()

    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 数据目录在 stca 的父目录下
    data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data for sharing_csv")
    # 输出目录在 stca 目录下
    output_path = args.output or os.path.join(
        os.path.dirname(script_dir),
        f"static_processed_{args.split_mode}.npz"
    )

    # 执行预处理
    preprocessor = StaticPreprocessor(data_dir=data_dir)
    data = preprocessor.process_stca(
        window_size=args.window_size,
        max_satellites=args.max_satellites,
        split_mode=args.split_mode,
    )

    preprocessor.save_processed(data, output_path)
    print(f"\n处理后的数据已保存到 {output_path}")
    print(f"时间通道形状 - 训练集：{data['X_train_temporal'].shape}")
    print(f"空间通道形状 - 训练集：{data['X_train_spatial'].shape}")


if __name__ == "__main__":
    main()
