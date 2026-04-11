from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class GNSSCombinedDataset(Dataset):
    def __init__(
        self,
        data_files,
        history_len=10,
        feature_cols=None,
        label_col="LOS",
        split_by_point=True,
        mean=None,
        std=None,
    ):
        if feature_cols is None:
            feature_cols = [
                "CNR",
                "Elevation",
                "Azimuth",
                "Pr_Residual",
                "Delta_CNR",
                "CNR_std",
                "PrRes_std",
                "Delta_Elevation",
                "Delta_Azimuth",
            ]

        self.samples = []

        for fp in data_files:
            fp = Path(fp)
            df = pd.read_excel(fp).dropna()
            station = os.path.basename(fp).split(".")[0]
            df[label_col] = df[label_col].astype(int)
            point = os.path.splitext(os.path.basename(fp))[0]
            if split_by_point:
                df["__PointName__"] = point

            df["Delta_CNR"] = df.groupby("PRN")["CNR"].diff().fillna(0)
            df["Delta_Pr_Residual"] = df.groupby("PRN")["Pr_Residual"].diff().fillna(0)
            df["Delta_Elevation"] = df.groupby("PRN")["Elevation"].diff().fillna(0)
            df["Delta_Azimuth"] = df.groupby("PRN")["Azimuth"].diff().fillna(0)

            df["CNR_std"] = (
                df.groupby("PRN")["CNR"].rolling(window=3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
            )
            df["PrRes_std"] = (
                df.groupby("PRN")["Pr_Residual"]
                .rolling(window=3, min_periods=1)
                .std()
                .reset_index(0, drop=True)
                .fillna(0)
            )

            time_samples = []
            group_keys = ["__PointName__", "PRN"] if split_by_point else ["PRN"]

            for _, grp in df.groupby(group_keys):
                grp = grp.sort_values("GPS_Time(s)").reset_index(drop=True)

                grp["Delta_CNR"] = grp["CNR"].diff().fillna(0)
                grp["Delta_Pr_Residual"] = grp["Pr_Residual"].diff().fillna(0)
                grp["Delta_Elevation"] = grp["Elevation"].diff().fillna(0)
                grp["Delta_Azimuth"] = grp["Azimuth"].diff().fillna(0)

                grp["CNR_std"] = grp["CNR"].rolling(window=3, min_periods=1).std().fillna(0)
                grp["PrRes_std"] = grp["Pr_Residual"].rolling(window=3, min_periods=1).std().fillna(0)

                feature_values = grp[feature_cols].values.astype(np.float32)
                if len(grp) < history_len + 1:
                    continue

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
                            "station": station,
                            "__PointName__": point,
                        }
                    )

            space_dict = {}
            for t, grp in df.groupby("GPS_Time(s)"):
                grp = grp.sort_values("PRN")
                space_dict[t] = grp[feature_cols].values.astype(np.float32)

            for sample in time_samples:
                t = sample["gps_time"]
                if t in space_dict:
                    self.samples.append({**sample, "space_global": space_dict[t]})

        if len(self.samples) == 0:
            raise RuntimeError("Dataset is empty. Check Excel paths, columns, and history length.")

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
