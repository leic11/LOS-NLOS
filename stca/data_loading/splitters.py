# splitters.py
"""
数据划分器模块 - 负责 indomain/outdomain 数据集划分
"""
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class DataSplitter:
    """数据划分器"""

    def __init__(self, random_seed: int = 42, val_size: float = 0.2):
        self.random_seed = random_seed
        self.val_size = val_size

    def split_indomain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
        test_size: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        内域数据集划分：每个地点内 70% 训练 / 30% 测试，再从训练集划分 20% 验证

        Args:
            X: 特征数组
            y: 标签数组
            locations: 地点数组
            test_size: 测试集比例

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        unique_locations = np.unique(locations)
        logger.info(f"In-domain split: {len(unique_locations)} locations")

        X_train_list, X_test_list = [], []
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
            train_val_idx = shuffled[n_test:]

            X_train_list.append(X_loc[train_val_idx])
            X_test_list.append(X_loc[test_idx])
            y_train_list.append(y_loc[train_val_idx])
            y_test_list.append(y_loc[test_idx])

            logger.info(
                f"  {loc}: train={len(train_val_idx)}, test={len(test_idx)}"
            )

        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # 从训练集中划分 20% 作为验证集
        n_val = int(len(X_train) * self.val_size)
        np.random.seed(self.random_seed)
        shuffled = np.random.permutation(len(X_train))

        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]

        logger.info(
            f"In-domain - Train: {X_train.shape[0]} | "
            f"Val: {X_val.shape[0]} | Test: {X_test.shape[0]}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_outdomain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        locations: np.ndarray,
        train_locations: List[str] = None,
        val_locations: List[str] = None,
        test_locations: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        外域数据集划分：P2-P5 训练，P6 验证，P7-P8 测试

        Args:
            X: 特征数组
            y: 标签数组
            locations: 地点数组
            train_locations: 训练地点列表
            val_locations: 验证地点列表
            test_locations: 测试地点列表

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        unique_locations = sorted(np.unique(locations))

        if train_locations is None:
            train_locations = unique_locations[:4]  # P2-P5
        if val_locations is None:
            val_locations = [unique_locations[4]]  # P6
        if test_locations is None:
            test_locations = unique_locations[5:]  # P7-P8

        logger.info(
            f"Out-domain split: train={train_locations}, "
            f"val={val_locations}, test={test_locations}"
        )

        train_mask = np.isin(locations, train_locations)
        val_mask = np.isin(locations, val_locations)
        test_mask = np.isin(locations, test_locations)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # 详细统计
        for loc in train_locations:
            loc_mask = locations == loc
            logger.info(
                f"  Train {loc}: {np.sum(loc_mask)} records, "
                f"NLOS={np.sum(y[loc_mask]==0)}, LOS={np.sum(y[loc_mask]==1)}"
            )

        for loc in val_locations:
            loc_mask = locations == loc
            logger.info(
                f"  Val {loc}: {np.sum(loc_mask)} records, "
                f"NLOS={np.sum(y[loc_mask]==0)}, LOS={np.sum(y[loc_mask]==1)}"
            )

        for loc in test_locations:
            loc_mask = locations == loc
            logger.info(
                f"  Test {loc}: {np.sum(loc_mask)} records, "
                f"NLOS={np.sum(y[loc_mask]==0)}, LOS={np.sum(y[loc_mask]==1)}"
            )

        logger.info(
            f"Out-domain - Train: {X_train.shape[0]} | "
            f"Val: {X_val.shape[0]} | Test: {X_test.shape[0]}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
