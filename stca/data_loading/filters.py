# filters.py
"""
数据过滤器模块 - 负责异常值过滤和标签映射
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from .constants import (
    DEFAULT_FEATURE_COLS,
    LABEL_COL,
    LABEL_MAP,
    PRE_FILTER_THRESHOLD,
    PR_RATE_INVALID,
)
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class DataFilter:
    """数据过滤器"""

    def __init__(
        self,
        feature_cols: List[str] = None,
        pre_threshold: float = PRE_FILTER_THRESHOLD,
        pr_rate_invalid: float = PR_RATE_INVALID,
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.pre_threshold = pre_threshold
        self.pr_rate_invalid = pr_rate_invalid

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤异常值

        Args:
            df: 原始 DataFrame

        Returns:
            过滤后的 DataFrame
        """
        n_before = len(df)

        # 丢弃含 NaN 的行
        df = df.dropna()
        n_after_dropna = len(df)
        logger.info(
            f"Dropped rows with NaN: {n_before} -> {n_after_dropna} (removed {n_before - n_after_dropna})"
        )

        # 过滤 Pr_rate_consitency
        df = df[df["Pr_rate_consitency"] != self.pr_rate_invalid].copy()
        n_after_pr = len(df)

        logger.info(
            f"Filtered Pr_rate_consitency (value == {self.pr_rate_invalid}): "
            f"{n_after_dropna} -> {n_after_pr} (removed {n_after_dropna - n_after_pr})"
        )

        logger.info(
            f"Total filtering: {n_before} -> {n_after_pr} records "
            f"({n_after_pr/n_before*100:.1f}%), removed {n_before - n_after_pr} outliers"
        )

        return df

    def map_labels(self, df: pd.DataFrame, label_col: str = LABEL_COL) -> pd.Series:
        """
        标签映射：-1→0, 1→1

        Args:
            df: DataFrame
            label_col: 标签列名

        Returns:
            映射后的标签 Series
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in data")

        return df[label_col].map(LABEL_MAP)

    def extract_features(
        self,
        df: pd.DataFrame,
        handle_missing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征和标签

        Args:
            df: DataFrame
            handle_missing: 是否处理缺失值

        Returns:
            (X, y): 特征数组和标签数组
        """
        # 检查必需列
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # 提取特征
        X = df[self.feature_cols].values.astype(np.float32)

        # 提取并映射标签
        y = df[LABEL_COL].values
        y = np.array([LABEL_MAP.get(v, v) for v in y], dtype=np.int32)

        # 处理缺失值
        if handle_missing:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Raw feature shape: {X.shape}, Labels shape: {y.shape}")
        logger.info(f"Label distribution: NLOS={np.sum(y==0)}, LOS={np.sum(y==1)}")

        return X, y
