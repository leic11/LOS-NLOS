# windowers.py
"""
窗口生成器模块 - 负责生成时间通道和空间通道输入
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from .constants import (
    DEFAULT_FEATURE_COLS,
    LABEL_COL,
    LABEL_MAP,
    DEFAULT_MAX_SATELLITES,
    DEFAULT_WINDOW_SIZE,
)
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class WindowGenerator:
    """STCA 窗口生成器"""

    def __init__(
        self,
        feature_cols: List[str] = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_satellites: int = DEFAULT_MAX_SATELLITES,
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.window_size = window_size
        self.max_satellites = max_satellites

    def generate_temporal_input(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成时间通道输入（按 location+PRN 分组滑动窗口）

        Args:
            df: 预处理后的 DataFrame

        Returns:
            X_temporal: (N, window_size, 4)
            y: (N,)
            locations: (N,)
            window_end_times: (N,)
        """
        # 排序
        df_sorted = df.sort_values(by=["GPS_Time(s)", "PRN"]).reset_index(drop=True)

        # 提取数据
        feature_data = df_sorted[self.feature_cols].values.astype(np.float32)
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        gps_times = df_sorted["GPS_Time(s)"].values
        labels = df_sorted[LABEL_COL].map(LABEL_MAP).values
        prn_values = df_sorted["PRN"].values
        loc_values = df_sorted["location"].values

        X_temporal_list = []
        y_list = []
        locations_list = []
        window_end_times = []

        # 按 (location, PRN) 组合分组 - 不同地点的相同 PRN 视为独立卫星
        unique_locations = np.unique(loc_values)
        total_windows = 0

        for loc in unique_locations:
            loc_mask = loc_values == loc
            loc_prn_values = prn_values[loc_mask]
            unique_prns_in_loc = np.unique(loc_prn_values)

            for prn in unique_prns_in_loc:
                # 同时匹配地点和 PRN
                combined_mask = (loc_values == loc) & (prn_values == prn)
                prn_indices = np.where(combined_mask)[0]

                if len(prn_indices) < self.window_size:
                    continue

                # 提取该 (location, PRN) 组合的数据
                prn_gps_times = gps_times[prn_indices]
                prn_features = feature_data[prn_indices]
                prn_labels = labels[prn_indices]
                prn_locations = loc_values[prn_indices]  # 应该全是同一个 loc

                # 按时间排序
                sorted_order = np.argsort(prn_gps_times)
                prn_gps_times = prn_gps_times[sorted_order]
                prn_features = prn_features[sorted_order]
                prn_labels = prn_labels[sorted_order]
                prn_locations = prn_locations[sorted_order]

                # 滑动窗口
                for start_idx in range(len(prn_indices) - self.window_size + 1):
                    window_X = prn_features[start_idx : start_idx + self.window_size]
                    window_y = prn_labels[start_idx + self.window_size - 1]
                    window_end_time = prn_gps_times[start_idx + self.window_size - 1]

                    X_temporal_list.append(window_X)
                    y_list.append(window_y)
                    locations_list.append(prn_locations[start_idx + self.window_size - 1])
                    window_end_times.append(window_end_time)

                total_windows += max(0, len(prn_indices) - self.window_size + 1)

        X_temporal = np.array(X_temporal_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        locations = np.array(locations_list)
        window_end_times = np.array(window_end_times)

        logger.info(f"Temporal input shape: {X_temporal.shape} (按 location+PRN 分组，共 {total_windows} 个窗口)")
        return X_temporal, y, locations, window_end_times

    def generate_spatial_input(
        self,
        df: pd.DataFrame,
        window_end_times: np.ndarray,
    ) -> np.ndarray:
        """
        生成空间通道输入（同一时刻所有卫星）

        Args:
            df: 预处理后的 DataFrame
            window_end_times: 窗口结束时间数组

        Returns:
            X_spatial: (N, max_satellites, 4)
        """
        # 按 GPS_Time 分组
        time_groups = df.groupby("GPS_Time(s)")
        time_to_sat_features = {}

        for t, group in time_groups:
            sat_features = group[self.feature_cols].values.astype(np.float32)
            sat_features = np.nan_to_num(sat_features, nan=0.0, posinf=0.0, neginf=0.0)
            time_to_sat_features[t] = sat_features

        X_spatial_list = []

        for end_time in window_end_times:
            if end_time in time_to_sat_features:
                sat_features = np.array(time_to_sat_features[end_time], dtype=np.float32)
                n_sat = len(sat_features)

                if n_sat >= self.max_satellites:
                    sat_features = sat_features[: self.max_satellites]
                else:
                    padding = np.zeros(
                        (self.max_satellites - n_sat, sat_features.shape[1]),
                        dtype=np.float32,
                    )
                    sat_features = np.vstack([sat_features, padding])
            else:
                sat_features = np.zeros(
                    (self.max_satellites, len(self.feature_cols)), dtype=np.float32
                )

            X_spatial_list.append(sat_features)

        X_spatial = np.array(X_spatial_list, dtype=np.float32)
        logger.info(f"Spatial input shape: {X_spatial.shape}")
        return X_spatial

    def generate_stca_inputs(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        生成 STCA 双通道输入

        Args:
            df: 预处理后的 DataFrame

        Returns:
            X_temporal, X_spatial, y, locations
        """
        X_temporal, y, locations, window_end_times = self.generate_temporal_input(df)
        X_spatial = self.generate_spatial_input(df, window_end_times)

        # 验证对齐
        assert len(X_temporal) == len(X_spatial) == len(y), (
            f"数据对齐错误：temporal={len(X_temporal)}, "
            f"spatial={len(X_spatial)}, labels={len(y)}"
        )

        return X_temporal, X_spatial, y, locations
