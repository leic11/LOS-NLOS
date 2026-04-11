# constants.py
"""
STCA 数据预处理常量配置
"""

# 论文规定的 4 个关键特征 (Section IV-A)
DEFAULT_FEATURE_COLS = [
    "C/N0",                    # 载噪比 (dB-Hz)
    "Elevation",               # 高度角 (度)
    "Azimuth",                 # 方位角 (度)
    "Pseudorange_residual",    # 伪距残差 (m)
]

# 标签列名和映射
LABEL_COL = "LOS/NLOS_label"
LABEL_MAP = {-1: 0, 1: 1}  # NLOS→0, LOS→1

# 地点文件前缀
DEFAULT_LOCATION_PREFIXES = ["P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# 数据划分默认值
DEFAULT_SPLIT_MODE = "outdomain"  # 默认划分模式：indomain 或 outdomain
DEFAULT_TEST_SIZE = 0.3
DEFAULT_VAL_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42

# STCA 模型输入参数
DEFAULT_WINDOW_SIZE = 16      # 论文最优值 (Section IV-F)
DEFAULT_MAX_SATELLITES = 20   # 单历元最大卫星数（实际数据通常 ≤ 20）

# 基础数据参数
INPUT_DIM = 4                 # 输入特征维度 (C/N0, Elevation, Azimuth, Pseudorange_residual)
NUM_CLASSES = 2               # 类别数量 (二分类输出，1 个神经元 + sigmoid)

# 异常值过滤阈值
PRE_FILTER_THRESHOLD = 100    # Pseudorange_residual 阈值
PR_RATE_INVALID = 9999.0      # Pr_rate_consitency 无效值

# Outdomain 划分地点配置
# 注意：P7 地点 LOS 比例极低 (11%)，不适合单独作为验证集
# 使用 P5 作为验证集 (LOS 约 65%)，P6+P7 作为测试集
OUTDOMAIN_TRAIN_LOCATIONS = ["P2", "P3", "P4", "P8"]
OUTDOMAIN_VAL_LOCATIONS = ["P5"]
OUTDOMAIN_TEST_LOCATIONS = ["P6", "P7"]
