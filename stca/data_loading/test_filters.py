# test_filters.py
"""
DataFilter 单元测试 - 测试空值处理功能

运行方式:
    cd stca/data_loading
    python test_filters.py
"""
import sys
import os

# 确保只从当前目录导入，避免上层包干扰
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pytest
import pandas as pd
import numpy as np

# 直接导入 filters 模块，不经过包导入链
import filters
from filters import DataFilter


class TestDataFilterNaNHandling:
    """测试 DataFilter 的空值处理能力"""

    @pytest.fixture
    def sample_df(self):
        """含空值的测试数据"""
        return pd.DataFrame({
            "C/N0": [35.0, 40.0, np.nan, 38.0, 36.0],
            "Elevation": [45.0, np.nan, 30.0, 60.0, 50.0],
            "Azimuth": [120.0, 90.0, 180.0, np.nan, 150.0],
            "Pseudorange_residual": [5.0, -3.0, 8.0, 12.0, np.nan],
            "Pr_rate_consitency": [1.0, 2.0, 3.0, 4.0, 5.0],
            "LOS/NLOS_label": [1, -1, 1, -1, 1],
        })

    @pytest.fixture
    def data_filter(self):
        """创建 DataFilter 实例"""
        return DataFilter()

    def test_dropna_removes_all_nan_rows(self, sample_df, data_filter):
        """测试 dropna 能删除所有含空值的行"""
        result = data_filter.filter_outliers(sample_df)

        # 原始 5 行，每行都有 1 个 NaN，应全部被删除
        assert len(result) == 0, "所有含 NaN 的行都应被删除"

    def test_dropna_preserves_complete_rows(self, sample_df, data_filter):
        """测试 dropna 保留完整数据行"""
        # 添加一行完整数据
        complete_row = pd.DataFrame({
            "C/N0": [35.0],
            "Elevation": [45.0],
            "Azimuth": [120.0],
            "Pseudorange_residual": [5.0],
            "Pr_rate_consitency": [1.0],
            "LOS/NLOS_label": [1],
        })
        df_with_complete = pd.concat([sample_df, complete_row], ignore_index=True)

        result = data_filter.filter_outliers(df_with_complete)

        assert len(result) == 1, "应保留唯一完整的那行"
        assert result.iloc[0]["C/N0"] == 35.0

    def test_dropna_before_outlier_filtering(self, sample_df, data_filter):
        """测试空值过滤在异常值过滤之前执行"""
        # 添加一行含 NaN 但伪距残差超限的数据
        nan_with_outlier = pd.DataFrame({
            "C/N0": [35.0],
            "Elevation": [45.0],
            "Azimuth": [120.0],
            "Pseudorange_residual": [200.0],  # 超出阈值 100
            "Pr_rate_consitency": [1.0],
            "LOS/NLOS_label": [1],
        })
        df = pd.concat([sample_df, nan_with_outlier], ignore_index=True)

        result = data_filter.filter_outliers(df)

        # 含 NaN 的行被删除，伪距超限的行也被删除
        assert len(result) == 0


class TestDataFilterOutlierFiltering:
    """测试异常值过滤功能（不含空值情况）"""

    @pytest.fixture
    def clean_df(self):
        """无空值的测试数据"""
        return pd.DataFrame({
            "C/N0": [35.0, 40.0, 38.0, 36.0],
            "Elevation": [45.0, 30.0, 60.0, 50.0],
            "Azimuth": [120.0, 90.0, 180.0, 150.0],
            "Pseudorange_residual": [5.0, -3.0, 150.0, 12.0],  # 第 3 行超限
            "Pr_rate_consitency": [1.0, 9999.0, 3.0, 4.0],    # 第 2 行无效
            "LOS/NLOS_label": [1, -1, 1, -1],
        })

    @pytest.fixture
    def data_filter(self):
        return DataFilter()

    def test_pseudorange_residual_filtering(self, clean_df, data_filter):
        """测试伪距残差异常值过滤"""
        result = data_filter.filter_outliers(clean_df)

        # 第 3 行 (|150| > 100) 应被删除
        assert len(result) == 3
        assert 150.0 not in result["Pseudorange_residual"].values

    def test_pr_rate_consitency_filtering(self, clean_df, data_filter):
        """测试 Pr_rate_consitency 无效值过滤"""
        result = data_filter.filter_outliers(clean_df)

        # 第 2 行 (9999.0) 应被删除
        assert len(result) == 3
        assert 9999.0 not in result["Pr_rate_consitency"].values

    def test_combined_filtering(self, clean_df, data_filter):
        """测试组合过滤"""
        result = data_filter.filter_outliers(clean_df)

        # 原始 4 行，删除 2 行异常值，保留 2 行
        assert len(result) == 2


class TestDataFilterLabelMapping:
    """测试标签映射功能"""

    @pytest.fixture
    def data_filter(self):
        return DataFilter()

    def test_label_mapping(self, data_filter):
        """测试标签映射 -1→0, 1→1"""
        df = pd.DataFrame({
            "LOS/NLOS_label": [-1, 1, -1, 1, 1],
        })

        y = data_filter.map_labels(df)

        assert list(y) == [0, 1, 0, 1, 1]

    def test_label_mapping_missing_column(self, data_filter):
        """测试缺失标签列时的错误处理"""
        df = pd.DataFrame({"other_col": [1, 2, 3]})

        with pytest.raises(ValueError, match="Label column"):
            data_filter.map_labels(df)


class TestDataFeatureExtraction:
    """测试特征提取功能"""

    @pytest.fixture
    def data_filter(self):
        return DataFilter(feature_cols=["C/N0", "Elevation"])

    def test_extract_features(self, data_filter):
        """测试特征和标签提取"""
        df = pd.DataFrame({
            "C/N0": [35.0, 40.0, 38.0],
            "Elevation": [45.0, 30.0, 60.0],
            "LOS/NLOS_label": [-1, 1, -1],
        })

        X, y = data_filter.extract_features(df)

        assert X.shape == (3, 2)
        assert list(y) == [0, 1, 0]

    def test_extract_features_with_nan_handling(self, data_filter):
        """测试含 NaN 数据的处理"""
        df = pd.DataFrame({
            "C/N0": [35.0, np.nan, 38.0],
            "Elevation": [45.0, 30.0, np.nan],
            "LOS/NLOS_label": [-1, 1, -1],
        })

        X, y = data_filter.extract_features(df, handle_missing=True)

        # NaN 应被替换为 0
        assert X[0, 0] == 35.0
        assert X[1, 0] == 0.0
        assert X[2, 1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
