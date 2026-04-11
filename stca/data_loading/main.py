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
_stca_dir = os.path.dirname(_current_dir)  # stca 目录
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))
if _stca_dir not in sys.path:
    sys.path.insert(0, str(_stca_dir))
if _current_dir not in sys.path:
    sys.path.insert(0, str(_current_dir))

# 支持两种运行方式：模块导入 或 直接运行
if __name__ == "__main__" or "data_loading" not in __name__:
    # 直接运行时（python main.py 或 python data_loading/main.py）
    from constants import (
        DEFAULT_FEATURE_COLS,
        DEFAULT_LOCATION_PREFIXES, DEFAULT_SPLIT_MODE, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE,
        DEFAULT_RANDOM_SEED, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_SATELLITES,
        PRE_FILTER_THRESHOLD, PR_RATE_INVALID,
        OUTDOMAIN_TRAIN_LOCATIONS, OUTDOMAIN_VAL_LOCATIONS, OUTDOMAIN_TEST_LOCATIONS,
    )
    from loaders import CSVLoader
    from filters import DataFilter
    from windowers import WindowGenerator
    from splitters import DataSplitter
    from normalizers import UnifiedScaler
else:
    # 作为模块导入时（python -m data_loading.main 或 from data_loading.main import ...）
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
        split_mode: str,
        window_end_times: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """获取训练集/验证集/测试集的划分索引

        Args:
            locations: 地点数组
            y: 标签数组
            split_mode: 划分模式
            window_end_times: 窗口结束时间数组（用于按时间划分）

        Returns:
            包含 train/val/test 索引或布尔掩码的字典
        """
        if split_mode == "indomain":
            # Indomain：每个地点内按时间顺序划分（前 70% 训练，后 30% 测试）
            # 避免随机划分导致的数据泄露问题
            unique_locations = np.unique(locations)
            train_indices = []
            test_indices = []

            for loc in unique_locations:
                loc_mask = locations == loc
                loc_indices = np.where(loc_mask)[0]

                # 获取该地点的时间信息
                loc_times = window_end_times[loc_mask] if window_end_times is not None else np.arange(len(loc_indices))

                # 按时间排序
                time_order = np.argsort(loc_times)
                sorted_loc_indices = loc_indices[time_order]

                n_samples = len(sorted_loc_indices)
                n_test = int(n_samples * self.test_size)

                # 按时间顺序划分：前 70% 训练/验证，后 30% 测试
                test_idx = sorted_loc_indices[-n_test:] if n_test > 0 else np.array([], dtype=int)
                train_val_idx = sorted_loc_indices[:-n_test] if n_test > 0 else sorted_loc_indices

                train_indices.extend(train_val_idx)
                test_indices.extend(test_idx)

                logger.info(
                    f"  {loc}: train_val={len(train_val_idx)}, test={len(test_idx)}"
                )

            # 从训练集中按时间划分验证集（前 20% 作为验证）
            train_indices = np.array(train_indices)
            if len(train_indices) > 0:
                train_times = window_end_times[train_indices] if window_end_times is not None else np.arange(len(train_indices))
                train_time_order = np.argsort(train_times)
                sorted_train_indices = train_indices[train_time_order]

                n_val = int(len(sorted_train_indices) * self.val_size)

                # 按时间顺序划分：前 20% 验证，后 80% 训练
                val_idx = sorted_train_indices[:n_val] if n_val > 0 else np.array([], dtype=int)
                train_idx = sorted_train_indices[n_val:] if n_val > 0 else sorted_train_indices
            else:
                train_idx = np.array([], dtype=int)
                val_idx = np.array([], dtype=int)

            logger.info(
                f"Indomain (temporal split) - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_indices)}"
            )

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

        # 3. 生成 STCA 输入（时间通道 + 空间通道）- 使用传入的 window_size
        windower = WindowGenerator(self.feature_cols, window_size, max_satellites)
        X_temporal, X_spatial_list, y, locations, window_end_times = windower.generate_stca_inputs(df)

        # 4. 划分数据集（传递时间信息用于按时间划分）
        indices = self._get_split_indices(locations, y, split_mode, window_end_times)

        # 5. 标准化（仅使用训练集拟合，应用到全部数据）
        # 类似 BaseModel：训练集计算 mean/std，应用到所有数据
        # 处理索引类型：indomain 返回索引数组，outdomain 返回布尔掩码
        train_idx = indices["train"]
        val_idx = indices["val"]
        test_idx = indices["test"]
        
        # 如果是索引数组（indomain），直接索引；如果是布尔掩码（outdomain），也直接索引
        X_train_temporal_raw = X_temporal[train_idx]
        X_val_temporal_raw = X_temporal[val_idx]
        X_test_temporal_raw = X_temporal[test_idx]
        
        # 变长空间输入：使用索引数组或布尔掩码提取
        if train_idx.dtype == bool:
            X_train_spatial_raw = [X_spatial_list[i] for i in range(len(X_spatial_list)) if train_idx[i]]
        else:
            X_train_spatial_raw = [X_spatial_list[i] for i in train_idx]
        if val_idx.dtype == bool:
            X_val_spatial_raw = [X_spatial_list[i] for i in range(len(X_spatial_list)) if val_idx[i]]
        else:
            X_val_spatial_raw = [X_spatial_list[i] for i in val_idx]
        if test_idx.dtype == bool:
            X_test_spatial_raw = [X_spatial_list[i] for i in range(len(X_spatial_list)) if test_idx[i]]
        else:
            X_test_spatial_raw = [X_spatial_list[i] for i in test_idx]
        
        self.scaler = UnifiedScaler.from_data(X_train_temporal_raw, X_train_spatial_raw)

        # 6. 变换所有数据
        X_train_temporal = self.scaler.transform(X_train_temporal_raw)
        X_val_temporal = self.scaler.transform(X_val_temporal_raw)
        X_test_temporal = self.scaler.transform(X_test_temporal_raw)

        # 变长空间输入：使用 transform_spatial
        X_train_spatial = self.scaler.transform_spatial(X_train_spatial_raw)
        X_val_spatial = self.scaler.transform_spatial(X_val_spatial_raw)
        X_test_spatial = self.scaler.transform_spatial(X_test_spatial_raw)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        # 计算实际样本数（outdomain 模式下 indices 是布尔掩码）
        n_train = np.sum(indices["train"]) if indices["train"].dtype == bool else len(indices["train"])
        n_val = np.sum(indices["val"]) if indices["val"].dtype == bool else len(indices["val"])
        n_test = np.sum(indices["test"]) if indices["test"].dtype == bool else len(indices["test"])

        logger.info(
            f"STCA 划分 - 训练集：{n_train}, "
            f"验证集：{n_val}, 测试集：{n_test}"
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
        """将处理后的数据保存为 .npz 文件（变长空间输入）"""
        scaler = data.get("scaler")
        import pickle

        # 变长空间数据用 pickle 保存
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
            # 变长空间数据用 pickle
            "X_train_spatial_pkl": pickle.dumps(data.get("X_train_spatial")),
            "X_val_spatial_pkl": pickle.dumps(data.get("X_val_spatial")),
            "X_test_spatial_pkl": pickle.dumps(data.get("X_test_spatial")),
        }

        np.savez(output_path, **save_dict)
        logger.info(f"已保存处理后的数据到 {output_path}")

    @staticmethod
    def load_processed(npz_path: str) -> Dict:
        """从 .npz 文件加载处理后的数据（变长空间输入）"""
        import pickle
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
            # 变长空间数据从 pickle 加载
            "X_train_spatial": pickle.loads(loader["X_train_spatial_pkl"]) if "X_train_spatial_pkl" in loader else None,
            "X_val_spatial": pickle.loads(loader["X_val_spatial_pkl"]) if "X_val_spatial_pkl" in loader else None,
            "X_test_spatial": pickle.loads(loader["X_test_spatial_pkl"]) if "X_test_spatial_pkl" in loader else None,
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
    print(f"空间通道 - 训练集：{len(data['X_train_spatial'])} samples, 每个样本卫星数不同")


if __name__ == "__main__":
    main()
