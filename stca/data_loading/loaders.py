# loaders.py
"""
数据加载器模块 - 负责 CSV 文件发现和加载
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class CSVLoader:
    """CSV 文件加载器"""

    def __init__(self, data_dir: str, location_prefixes: List[str]):
        self.data_dir = Path(data_dir)
        self.location_prefixes = location_prefixes
        self.location_files: Dict[str, Path] = {}
        self._discover_location_files()

    def _discover_location_files(self) -> None:
        """发现数据目录中的所有地点 CSV 文件"""
        csv_files = sorted(self.data_dir.glob("*.csv"))
        for csv_path in csv_files:
            stem = csv_path.stem
            if stem in self.location_prefixes:
                self.location_files[stem] = csv_path

        logger.info(f"Discovered {len(self.location_files)} location files: {list(self.location_files.keys())}")

    def load_location(self, location: str) -> pd.DataFrame:
        """
        加载指定地点的 CSV 文件

        Args:
            location: 地点名称 (如 "P2")

        Returns:
            包含 location 列的 DataFrame

        Raises:
            ValueError: 地点不存在
        """
        if location not in self.location_files:
            available = list(self.location_files.keys())
            raise ValueError(f"Location '{location}' not found. Available: {available}")

        df = pd.read_csv(self.location_files[location], skipinitialspace=True)
        df["location"] = location
        logger.info(f"Loaded {location}: {len(df)} records")
        return df

    def load_and_merge(self) -> pd.DataFrame:
        """
        加载并合并所有 CSV 文件

        Returns:
            合并后的 DataFrame，包含 location 列

        Raises:
            FileNotFoundError: 未找到 CSV 文件
        """
        if not self.location_files:
            self._discover_location_files()

        if not self.location_files:
            raise FileNotFoundError(f"No location CSV files found in {self.data_dir}")

        dfs = []
        for location, csv_path in sorted(self.location_files.items()):
            logger.info(f"Loading {csv_path.name}...")
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df["location"] = location
            dfs.append(df)

        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
