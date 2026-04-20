# splitters.py
"""
数据划分器模块 - 负责 indomain/outdomain 数据集划分
"""
import numpy as np
from typing import Tuple
import os, sys

# 支持直接运行
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, str(_current_dir))

if __name__ == "__main__" or "data_loading" not in __name__:
    from constants import (
        OUTDOMAIN_TRAIN_LOCATIONS,
        OUTDOMAIN_TEST_LOCATIONS,
    )
else:
    from .constants import (
        OUTDOMAIN_TRAIN_LOCATIONS,
        OUTDOMAIN_TEST_LOCATIONS,
    )

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class DataSplitter:
    """数据划分器"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def split_indomain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
        test_size: float = 0.3,
        X_spatial: np.ndarray = None,
    ) -> Tuple:
        """
        内域数据集划分：每个地点内 70% 训练 / 30% 测试，无验证集

        Args:
            X: 时间特征数组 (N, window_size, 4)
            y: 标签数组
            locations: 地点数组
            test_size: 测试集比例
            X_spatial: 空间特征数组 (N, max_satellites, 4)，可选

        Returns:
            如果 X_spatial 为 None: X_train, X_test, y_train, y_test
            如果 X_spatial 提供：返回 (X_train, X_test, X_spatial_train, X_spatial_test, y_train, y_test)
        """
        unique_locations = np.unique(locations)
        logger.info(f"In-domain split: {len(unique_locations)} locations")

        X_train_list, X_test_list = [], []
        X_spatial_train_list, X_spatial_test_list = [], [] if X_spatial is not None else None
        y_train_list, y_test_list = [], []

        for loc in unique_locations:
            loc_mask = locations == loc
            X_loc = X[loc_mask]
            y_loc = y[loc_mask]

            # 该地点内随机划分
            loc_indices = np.where(loc_mask)[0]
            n_samples = len(loc_indices)
            n_test = int(n_samples * test_size)

            np.random.seed(self.random_seed)
            shuffled = np.random.permutation(n_samples)

            test_idx = shuffled[:n_test]
            train_idx = shuffled[n_test:]

            X_train_list.append(X_loc[train_idx])
            X_test_list.append(X_loc[test_idx])
            y_train_list.append(y_loc[train_idx])
            y_test_list.append(y_loc[test_idx])

            if X_spatial is not None:
                X_spatial_loc = X_spatial[loc_mask]
                X_spatial_train_list.append(X_spatial_loc[train_idx])
                X_spatial_test_list.append(X_spatial_loc[test_idx])

            logger.info(
                f"  {loc}: train={len(train_idx)}, test={len(test_idx)}"
            )

        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        X_spatial_train = np.concatenate(X_spatial_train_list, axis=0) if X_spatial is not None else None
        X_spatial_test = np.concatenate(X_spatial_test_list, axis=0) if X_spatial is not None else None

        logger.info(
            f"In-domain - Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
        )

        if X_spatial is not None:
            return (X_train, X_test, X_spatial_train, X_spatial_test, y_train, y_test)
        return X_train, X_test, y_train, y_test

    def split_outdomain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
        X_spatial: np.ndarray = None,
    ) -> Tuple:
        """
        外域数据集划分：5 个地点训练，2 个地点测试（无验证集）

        Args:
            X: 时间特征数组 (N, window_size, 4)
            y: 标签数组
            locations: 地点数组
            X_spatial: 空间特征数组 (N, max_satellites, 4)，可选

        Returns:
            如果 X_spatial 为 None: X_train, X_test, y_train, y_test
            如果 X_spatial 提供：返回 (X_train, X_test, X_spatial_train, X_spatial_test, y_train, y_test)
        """
        unique_locations = sorted(np.unique(locations))
        logger.info(f"所有地点：{unique_locations}")

        train_locations = OUTDOMAIN_TRAIN_LOCATIONS
        test_locations = OUTDOMAIN_TEST_LOCATIONS

        logger.info(
            f"Out-domain split: train={train_locations}, test={test_locations}"
        )

        train_mask = np.isin(locations, train_locations)
        test_mask = np.isin(locations, test_locations)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        X_spatial_train = X_spatial[train_mask] if X_spatial is not None else None
        X_spatial_test = X_spatial[test_mask] if X_spatial is not None else None

        # 详细统计
        for loc in train_locations:
            loc_mask = locations == loc
            logger.info(
                f"  Train {loc}: {np.sum(loc_mask)} records, "
                f"NLOS={np.sum(y[loc_mask]==0)}, LOS={np.sum(y[loc_mask]==1)}"
            )

        for loc in test_locations:
            loc_mask = locations == loc
            logger.info(
                f"  Test {loc}: {np.sum(loc_mask)} records, "
                f"NLOS={np.sum(y[loc_mask]==0)}, LOS={np.sum(y[loc_mask]==1)}"
            )

        logger.info(
            f"Out-domain - Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
        )

        if X_spatial is not None:
            return (X_train, X_test, X_spatial_train, X_spatial_test, y_train, y_test)
        return X_train, X_test, y_train, y_test
