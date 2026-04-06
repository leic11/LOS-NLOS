"""
STCA Data Preprocessing Package

This package provides a modular, testable data preprocessing pipeline
for GNSS NLOS signal classification using the STCA model.

Modules:
    - constants: Configuration constants
    - loaders: CSV file loading and merging
    - filters: Outlier filtering and label mapping
    - windowers: Temporal and spatial window generation
    - splitters: Dataset splitting (indomain/outdomain)
    - normalizers: Unified feature normalization
    - main: StaticPreprocessor facade class

Example:
    >>> from data_loading import StaticPreprocessor
    >>> preprocessor = StaticPreprocessor(data_dir="path/to/data")
    >>> data = preprocessor.process_stca(split_mode="outdomain")
    >>> preprocessor.save_processed(data, "output.npz")
"""
from main import StaticPreprocessor
from constants import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MAX_SATELLITES,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
)

__all__ = [
    "StaticPreprocessor",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_MAX_SATELLITES",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_VAL_SIZE",
]
