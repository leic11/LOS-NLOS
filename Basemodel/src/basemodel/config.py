from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_FEATURE_COLS = [
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

# 地点划分配置（与 STCA 保持一致）
OUTDOMAIN_TRAIN_LOCATIONS = ["P2", "P3", "P4", "P8"]
OUTDOMAIN_VAL_LOCATIONS = ["P5"]
OUTDOMAIN_TEST_LOCATIONS = ["P6", "P7"]


@dataclass
class TrainConfig:
    # 输出目录
    result_root: Path = Path(__file__).resolve().parents[3] / "outputs" / "basemodel"
    exp_name: str = "Basemodel"

    # 数据路径 - 使用 CSV 文件目录（和 STCA 相同）
    data_dir: Path = Path(__file__).resolve().parents[3] / "data for sharing_csv"

    # 数据划分模式："indomain" 或 "outdomain"
    split_mode: str = "outdomain"

    feature_cols: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLS.copy())
    label_col: str = "LOS"
    history_len: int = 10
    split_by_point: bool = True

    # 训练超参数
    epochs: int = 20
    batch_size: int = 32
    lr: float = 2e-4
    train_lambda_reg: float = 0.008
    eval_lambda_reg: float = 0.005
    dropout: float = 0.6

    # 模型架构
    lstm_hidden: int = 64
    lstm_layers: int = 2
    attn_hidden: int = 64
    attn_heads: int = 1
    ff_hidden: int = 128

    @property
    def outdir(self) -> Path:
        return self.result_root / self.exp_name

    @property
    def figdir(self) -> Path:
        return self.outdir / "figs"

    def get_train_locations(self) -> list[str]:
        """获取训练集地点列表"""
        if self.split_mode == "outdomain":
            return OUTDOMAIN_TRAIN_LOCATIONS
        else:  # indomain
            return ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]

    def get_test_locations(self) -> list[str]:
        """获取测试集地点列表"""
        if self.split_mode == "outdomain":
            return OUTDOMAIN_TEST_LOCATIONS
        else:  # indomain - indomain 模式下测试集从各地点后 30% 划分
            return ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]

    def get_val_locations(self) -> list[str]:
        """获取验证集地点列表（outdomain 模式专用）"""
        if self.split_mode == "outdomain":
            return OUTDOMAIN_VAL_LOCATIONS
        else:
            return []
