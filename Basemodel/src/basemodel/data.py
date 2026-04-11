from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class GNSSCombinedDataset(Dataset):
    """
    GNSS 数据集 - 从 CSV 文件加载（保留原有 9 个特征工程）

    支持两种划分模式：
    - indomain: 每个地点内按时间划分（前 70% 训练，后 30% 测试）
    - outdomain: 按地点划分（训练地点和测试地点完全不重叠）

    特征：
    - 4 个原始特征：C/N0, Elevation, Azimuth, Pseudorange_residual
    - 5 个衍生特征：Delta_CNR, CNR_std, PrRes_std, Delta_Elevation, Delta_Azimuth
    """

    def __init__(
        self,
        data_dir: str,
        history_len: int = 10,
        feature_cols: List[str] = None,
        label_col: str = "LOS/NLOS_label",
        split_mode: str = "outdomain",
        train_locations: List[str] = None,
        test_locations: List[str] = None,
        split_by_point: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        if feature_cols is None:
            feature_cols = [
                "CNR",
                "Elevation",
                "Azimuth",
                "Pseudorange_residual",
                "Delta_CNR",
                "CNR_std",
                "PrRes_std",
                "Delta_Elevation",
                "Delta_Azimuth",
            ]

        self.samples = []
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.split_mode = split_mode

        # 从 CSV 目录加载所有 CSV 文件
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise RuntimeError(f"No CSV files found in {data_dir}")

        # 根据划分模式确定训练/测试地点
        if split_mode == "outdomain":
            # Outdomain: 按地点划分
            is_train_split = train_locations is not None
            location_set = set(train_locations) if is_train_split else set(test_locations)
        else:
            # Indomain: 每个地点内按时间划分
            is_train_split = mean is None  # 第一次加载是训练集
            location_set = None  # indomain 模式下使用所有地点

        for fp in csv_files:
            point = os.path.splitext(os.path.basename(fp))[0]

            # Outdomain 模式：跳过不属于当前数据集的地点
            if split_mode == "outdomain" and point not in location_set:
                continue

            df = pd.read_csv(fp)

            # 先重命名列，再检查
            # 重命名 C/N0 为 CNR 以便与原有特征名兼容
            if "C/N0" in df.columns:
                df = df.rename(columns={"C/N0": "CNR"})

            # 重命名 LOS/NLOS_label 为 LOS
            if "LOS/NLOS_label" in df.columns and "LOS" not in df.columns:
                df = df.rename(columns={"LOS/NLOS_label": "LOS"})

            # 检查必需列
            required_cols = ["GPS_Time(s)", "PRN", "CNR", "Elevation", "Azimuth", "Pseudorange_residual", label_col]
            missing = set(required_cols) - set(df.columns)
            if missing:
                print(f"Warning: Missing columns in {fp}: {missing}")
                continue

            # 标签转换：-1 -> 0 (NLOS), 1 -> 1 (LOS)
            if label_col in df.columns:
                df[label_col] = df[label_col].apply(lambda x: 0 if x == -1 else 1)
            df[label_col] = df[label_col].astype(int)

            if split_by_point:
                df["__PointName__"] = point

            # 特征工程 - 衍生特征
            df["Delta_CNR"] = df.groupby("PRN")["CNR"].diff().fillna(0)
            df["Delta_Pr_Residual"] = df.groupby("PRN")["Pseudorange_residual"].diff().fillna(0)
            df["Delta_Elevation"] = df.groupby("PRN")["Elevation"].diff().fillna(0)
            df["Delta_Azimuth"] = df.groupby("PRN")["Azimuth"].diff().fillna(0)

            df["CNR_std"] = (
                df.groupby("PRN")["CNR"].rolling(window=3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
            )
            df["PrRes_std"] = (
                df.groupby("PRN")["Pseudorange_residual"]
                .rolling(window=3, min_periods=1)
                .std()
                .reset_index(0, drop=True)
                .fillna(0)
            )

            time_samples = []
            group_keys = ["__PointName__", "PRN"] if split_by_point else ["PRN"]

            for _, grp in df.groupby(group_keys):
                grp = grp.sort_values("GPS_Time(s)").reset_index(drop=True)

                # 重新计算组内衍生特征（确保准确性）
                grp["Delta_CNR"] = grp["CNR"].diff().fillna(0)
                grp["Delta_Pr_Residual"] = grp["Pseudorange_residual"].diff().fillna(0)
                grp["Delta_Elevation"] = grp["Elevation"].diff().fillna(0)
                grp["Delta_Azimuth"] = grp["Azimuth"].diff().fillna(0)

                grp["CNR_std"] = grp["CNR"].rolling(window=3, min_periods=1).std().fillna(0)
                grp["PrRes_std"] = grp["Pseudorange_residual"].rolling(window=3, min_periods=1).std().fillna(0)

                feature_values = grp[feature_cols].values.astype(np.float32)
                if len(grp) < history_len + 1:
                    continue

                # Indomain 模式：按时间划分
                if split_mode == "indomain":
                    split_idx = int(len(grp) * 0.7)
                    if is_train_split:
                        grp = grp.iloc[:split_idx]
                    else:
                        grp = grp.iloc[split_idx:]

                    # 重新提取特征
                    if len(grp) < history_len + 1:
                        continue
                    feature_values = grp[feature_cols].values.astype(np.float32)

                for i in range(history_len, len(grp)):
                    hist = feature_values[i - history_len : i]
                    t = grp.iloc[i]["GPS_Time(s)"]
                    lbl = int(grp.iloc[i][label_col])
                    prn = grp.iloc[i]["PRN"]
                    time_samples.append(
                        {
                            "prn": prn,
                            "time_history": hist,
                            "gps_time": t,
                            "label": lbl,
                            "station": point,
                            "__PointName__": point,
                        }
                    )

            # 构建空间输入（同一时刻所有卫星）
            space_dict = {}
            for t, grp in df.groupby("GPS_Time(s)"):
                grp = grp.sort_values("PRN")
                space_dict[t] = grp[feature_cols].values.astype(np.float32)

            for sample in time_samples:
                t = sample["gps_time"]
                if t in space_dict:
                    self.samples.append({**sample, "space_global": space_dict[t]})

        if len(self.samples) == 0:
            raise RuntimeError("Dataset is empty. Check CSV paths, columns, and history length.")

        # 标准化
        all_features = np.vstack([s["time_history"] for s in self.samples])
        if mean is None or std is None:
            self.mean = all_features.mean(axis=0)
            self.std = all_features.std(axis=0) + 1e-5
        else:
            self.mean = mean
            self.std = std

        for sample in self.samples:
            sample["time_history"] = torch.tensor((sample["time_history"] - self.mean) / self.std, dtype=torch.float32)
            sample["space_global"] = torch.tensor((sample["space_global"] - self.mean) / self.std, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def combined_collate_fn(batch):
    times = torch.stack([b["time_history"] for b in batch], dim=0)
    spaces = [b["space_global"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    spaces_p = pad_sequence(spaces, batch_first=True)
    return {
        "time_history": times,
        "space_global": spaces_p,
        "labels": labels,
        "prns": [b["prn"] for b in batch],
        "gps_times": [b["gps_time"] for b in batch],
        "stations": [b["station"] for b in batch],
    }
