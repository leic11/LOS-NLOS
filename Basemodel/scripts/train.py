from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from basemodel.config import TrainConfig
from basemodel.io_utils import RedirectStdStreams
from basemodel.train import run_training


def main() -> None:
    config = TrainConfig()
    log_path = config.outdir / f"{config.exp_name}.txt"
    with RedirectStdStreams(log_path):
        run_training(config)


if __name__ == "__main__":
    main()
