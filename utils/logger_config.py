# logger_config.py
"""
统一日志配置模块

所有模块导入此模块配置的 logger 即可，日志统一输出到 outputs/ 目录
"""
import os
import logging
from datetime import datetime

# 项目根目录（utils 的父目录，即 DevLab）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 日志输出目录（项目根目录/outputs/logs）
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名（按日期命名）
log_filename = datetime.now().strftime("stca_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)


def setup_logger(name: str) -> logging.Logger:
    """
    配置并返回 logger

    Args:
        name: logger 名称（通常使用 __name__）

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 创建文件 handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# 初始化根 logger
root_logger = setup_logger("stca")
