from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_FEATURE_COLS = [
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


@dataclass
class TrainConfig:
    result_root: Path = Path(r"D:\AAA_Doctoral_Materails\Read_Paper\Basemodel\outputs")
    exp_name: str = "Basemodel"
    data_dir: Path = Path(r"D:\guyansong\DATA")
    train_files: list[str] = field(
        default_factory=lambda: ["Wampo_Data.xlsx", "MongKok_Data.xlsx", "TsimShaTsui_Data.xlsx"]
    )
    test_files: list[str] = field(default_factory=lambda: ["Test.xlsx"])
    feature_cols: list[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLS.copy())
    label_col: str = "LOS"
    history_len: int = 10
    split_by_point: bool = True
    epochs: int = 20
    batch_size: int = 32
    lr: float = 2e-4
    train_lambda_reg: float = 0.008
    eval_lambda_reg: float = 0.005
    dropout: float = 0.6
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

    def resolve_train_files(self) -> list[Path]:
        return [self.data_dir / name for name in self.train_files]

    def resolve_test_files(self) -> list[Path]:
        return [self.data_dir / name for name in self.test_files]
