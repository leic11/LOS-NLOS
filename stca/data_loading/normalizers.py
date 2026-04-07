# normalizers.py
"""
标准化器模块 - 负责特征标准化
"""
import numpy as np
from typing import Dict

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class UnifiedScaler:
    """
    统一标准化器

    时间特征和空间特征使用同一个标准化器，因为：
    - 两者是同一组 4 个特征（C/N0, Elevation, Azimuth, Pseudorange_residual）
    - 交叉注意力融合时需要数值范围完全一致
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        """
        Args:
            means: 每个特征的均值 (4,)
            stds: 每个特征的标准差 (4,)
        """
        self.means = means
        self.stds = stds
        # 兼容 save/load 代码
        self.mean_ = means
        self.scale_ = stds

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        标准化变换

        Args:
            X: 输入特征 (N, num_features * 4) 或 (N, window_size, 4) 或 (N, max_satellites, 4)

        Returns:
            标准化后的特征
        """
        if X.ndim == 2:
            # (N, num_features * 4) -> 重塑为 (N, num_features, 4)
            n_samples = X.shape[0]
            num_features = X.shape[1] // 4
            X_reshaped = X.reshape(n_samples, num_features, 4)
            return ((X_reshaped - self.means) / self.stds).reshape(
                n_samples, num_features * 4
            )
        elif X.ndim == 3:
            # (N, seq_len, 4)
            return (X - self.means) / self.stds
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

    def fit(self, X_temporal: np.ndarray, X_spatial: np.ndarray) -> "UnifiedScaler":
        """
        拟合标准化器（仅使用训练数据）

        Args:
            X_temporal: 时间通道输入 (N, window_size, 4)
            X_spatial: 空间通道输入 (N, max_satellites, 4)

        Returns:
            self
        """
        # 对每个特征单独计算均值和标准差
        means = []
        stds = []

        for feat_idx in range(4):
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
            X_temporal: 时间通道输入 (N, window_size, 4)
            X_spatial: 空间通道输入 (N, max_satellites, 4)

        Returns:
            UnifiedScaler 实例
        """
        scaler = cls(means=np.zeros(4), stds=np.ones(4))
        scaler.fit_transform(X_temporal, X_spatial)
        return scaler
