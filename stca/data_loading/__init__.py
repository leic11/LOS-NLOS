"""
STCA 数据预处理包

提供用于 STCA 模型的模块化数据预处理流程。

示例:
    >>> from data_loading import StaticPreprocessor
    >>> preprocessor = StaticPreprocessor(data_dir="path/to/data")
    >>> data = preprocessor.process_stca(split_mode="outdomain")
    >>> preprocessor.save_processed(data, "output.npz")
"""
from .main import StaticPreprocessor
from .constants import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MAX_SATELLITES,
    DEFAULT_TEST_SIZE,
    DEFAULT_SPLIT_MODE,
    INPUT_DIM,
    NUM_CLASSES,
)

__all__ = [
    "StaticPreprocessor",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_MAX_SATELLITES",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_SPLIT_MODE",
    "INPUT_DIM",
    "NUM_CLASSES",
]
