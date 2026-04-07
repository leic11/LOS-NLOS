# main.py
"""
STCA Data Preprocessor - Refactored Version (Facade Pattern)

This module orchestrates the data preprocessing pipeline by composing
smaller, single-responsibility modules.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from constants import (
    DEFAULT_FEATURE_COLS, LABEL_COL, LABEL_MAP,
    DEFAULT_LOCATION_PREFIXES, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE,
    DEFAULT_RANDOM_SEED, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_SATELLITES,
    PRE_FILTER_THRESHOLD, PR_RATE_INVALID,
)
from loaders import CSVLoader
from filters import DataFilter
from windowers import WindowGenerator
from splitters import DataSplitter
from normalizers import UnifiedScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StaticPreprocessor:
    """STCA Data Preprocessor using Facade Pattern."""

    def __init__(
        self,
        data_dir: str,
        feature_cols: Optional[List[str]] = None,
        test_size: float = DEFAULT_TEST_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ):
        self.data_dir = data_dir
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        # Compose modules
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
        """Delegate to CSVLoader."""
        return self.loader.load_and_merge()

    def filter_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to DataFilter."""
        return self.filter.filter_outliers(df)

    def _get_split_masks(
        self,
        locations: np.ndarray,
        split_mode: str
    ) -> Dict[str, np.ndarray]:
        """Get boolean masks for train/val/test split."""
        if split_mode == "indomain":
            # Indomain: 每个地点内随机划分，无法直接用 mask 表示
            # 返回索引数组
            y = np.zeros(len(locations), dtype=np.int32)
            X_dummy = np.arange(len(locations))
            X_train, X_val, X_test, *_ = self.splitter.split_indomain(
                X_dummy.reshape(-1, 1), y, locations, self.test_size
            )
            return {
                "train": X_train.flatten(),
                "val": X_val.flatten(),
                "test": X_test.flatten()
            }
        else:
            # Outdomain: 按地点划分，可以直接用 mask
            train_locs = ["P2", "P3", "P4", "P8"]
            val_locs = ["P7"]
            test_locs = ["P5", "P6"]

            return {
                "train": np.isin(locations, train_locs),
                "val": np.isin(locations, val_locs),
                "test": np.isin(locations, test_locs)
            }

    def process_stca(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_satellites: int = DEFAULT_MAX_SATELLITES,
        split_mode: str = "indomain",
    ) -> Dict:
        """Orchestrate the complete preprocessing pipeline."""
        # 1. Load
        df = self.loader.load_and_merge()

        # 2. Filter
        df = self.filter.filter_outliers(df)

        # 3. Generate STCA inputs
        X_temporal, X_spatial, y, locations = self.windower.generate_stca_inputs(df)

        # 4. Split dataset
        indices = self._get_split_masks(locations, split_mode)

        # 5. Normalize (fit on training set only)
        self.scaler = UnifiedScaler.from_data(
            X_temporal[indices["train"]],
            X_spatial[indices["train"]]
        )

        # 6. Transform all data
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
            f"STCA Split - Train: {len(indices['train'])}, "
            f"Val: {len(indices['val'])}, Test: {len(indices['test'])}"
        )

        # 7. Determine location info
        if split_mode == "indomain":
            train_locations = val_locations = test_locations = "indomain"
        else:
            unique_locs = sorted(np.unique(locations))
            train_locations = unique_locs[:4]
            val_locations = [unique_locs[4]]
            test_locations = unique_locs[5:]

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
        """Save processed data to .npz file."""
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
        logger.info(f"Saved processed data to {output_path}")

    @staticmethod
    def load_processed(npz_path: str) -> Dict:
        """Load processed data from .npz file."""
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
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="STCA Data Preprocessor")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="indomain",
        choices=["indomain", "outdomain"]
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE
    )
    parser.add_argument(
        "--max-satellites",
        type=int,
        default=DEFAULT_MAX_SATELLITES
    )
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data for sharing_csv")
    output_path = args.output or os.path.join(
        os.path.dirname(script_dir),
        f"static_processed_{args.split_mode}.npz"
    )

    # Execute preprocessing
    preprocessor = StaticPreprocessor(data_dir=data_dir)
    data = preprocessor.process_stca(
        window_size=args.window_size,
        max_satellites=args.max_satellites,
        split_mode=args.split_mode,
    )

    preprocessor.save_processed(data, output_path)
    print(f"\nProcessed data saved to {output_path}")
    print(f"Temporal shapes - Train: {data['X_train_temporal'].shape}")
    print(f"Spatial shapes - Train: {data['X_train_spatial'].shape}")


if __name__ == "__main__":
    main()
