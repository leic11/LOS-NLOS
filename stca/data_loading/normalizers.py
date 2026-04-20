# normalizers.py
"""
标准化器模块 - 负责特征标准化
"""
import numpy as np
from typing import Dict
import os, sys

# 支持直接运行
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class UnifiedScaler:
    """
    统一标准化器

    时间特征和空间特征使用同一个标准化器，因为：
    - 两者是同一组特征（4 个原始特征）
    - 交叉注意力融合时需要数值范围完全一致
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        """
        Args:
            means: 每个特征的均值 (num_features,)
            stds: 每个特征的标准差 (num_features,)
        """
        assert means.shape == stds.shape, f"means shape {means.shape} != stds shape {stds.shape}"
        self.means = means
        self.stds = stds
        # 兼容 save/load 代码
        self.mean_ = means
        self.scale_ = stds

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标准化变换

        Args:
            X: 输入特征 (N, num_features) 或 (N, window_size, num_features) 或 (N, max_satellites, num_features)

        Returns:
            标准化后的特征
        """
        if X.ndim == 2:
            # (N, num_features) - 直接应用标准化
            return (X - self.means) / self.stds
        elif X.ndim == 3:
            # (N, seq_len, num_features)
            return (X - self.means) / self.stds
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

    def fit(self, X_temporal: np.ndarray, X_spatial: np.ndarray) -> "UnifiedScaler":
        """
        拟合标准化器（仅使用训练数据）

        Args:
            X_temporal: 时间通道输入 (N, window_size, num_features)
            X_spatial: 空间通道输入 (N, max_satellites, num_features)

        Returns:
            self
        """
        # 动态获取特征数量
        num_features = X_temporal.shape[-1]

        # 对每个特征单独计算均值和标准差
        means = []
        stds = []

        for feat_idx in range(num_features):
            # 提取时间输入和空间输入中该特征的所有值
            temporal_feat = X_temporal[:, :, feat_idx].reshape(-1)
            spatial_feat = X_spatial[:, :, feat_idx].reshape(-1)

            # 合并计算
            combined_feat = np.concatenate([temporal_feat, spatial_feat])
            means.append(combined_feat.mean())
            stds.append(combined_feat.std())

        self.means = np.array(means)
        self.stds = np.array(stds)
        self.mean_ = self.means
        self.scale_ = self.stds

        return self

    def fit_transform(
        self, X_temporal: np.ndarray, X_spatial: np.ndarray
    ) -> np.ndarray:
        """
        拟合并应用标准化变换

        Args:
            X_temporal: 时间通道输入 (N, window_size, 4)
            X_spatial: 空间通道输入 (N, max_satellites, 4)

        Returns:
            标准化后的 X_temporal
        """
        self.fit(X_temporal, X_spatial)
        return self.transform(X_temporal)

    @classmethod
    def from_data(cls, X_temporal: np.ndarray, X_spatial: np.ndarray) -> "UnifiedScaler":
        """
        从数据创建标准化器

        Args:
            X_temporal: 时间通道输入 (N, window_size, num_features)
            X_spatial: 空间通道输入 (N, max_satellites, num_features)

        Returns:
            UnifiedScaler 实例
        """
        # 根据数据形状动态确定特征数量
        num_features = X_temporal.shape[-1]
        scaler = cls(means=np.zeros(num_features), stds=np.ones(num_features))
        scaler.fit_transform(X_temporal, X_spatial)
        return scaler
