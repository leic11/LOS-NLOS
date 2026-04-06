# trainer.py
"""
Trainer Class - PyTorch Version
训练器类
=============================================================

本模块实现模型训练的核心功能，包括训练循环、验证、模型保存等。

══════════════════════════════════════════════════════════════════════════════
                              功能列表
══════════════════════════════════════════════════════════════════════════════

    1. 训练循环: 支持多轮训练，包含前向传播、损失计算、反向传播
    2. 验证评估: 在验证集上评估模型性能
    3. 模型保存: 保存最佳模型权重和完整模型
    4. 指标记录: 记录训练过程中的损失和指标
    5. 早停机制: 防止过拟合
    6. 学习率调度: 支持学习率衰减
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path
import logging
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer Class for PyTorch Model.
    
    负责模型训练、验证和模型保存。
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        criterion: nn.Module = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs",
        use_sparse_loss: bool = False,
        sparse_weight: float = 1e-4,
        use_dual_input: bool = False,
        split_mode: str = "indomain",
    ):
        """
        初始化训练器

        Args:
            model: PyTorch模型
            optimizer: 优化器 (默认为Adam)
            criterion: 损失函数 (默认为CrossEntropyLoss)
            device: 训练设备 ("cuda" 或 "cpu")
            output_dir: 输出目录
            use_sparse_loss: 是否使用稀疏损失
            sparse_weight: 稀疏损失权重
            use_dual_input: 是否使用双输入模式 (2D for AAM, 3D for LSTM)
            split_mode: 数据分割模式 ("indomain" 或 "outdomain")，用于模型文件名
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_dual_input = use_dual_input
        self.split_mode = split_mode
        
        # 默认优化器
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
        else:
            self.optimizer = optimizer
        
        # 默认损失函数: 二元交叉熵（论文IV.E节）
        # 使用BCELoss因为模型输出层已经包含sigmoid
        if criterion is None:
            self.criterion = nn.BCELoss()
        else:
            self.criterion = criterion
        
        self.use_sparse_loss = use_sparse_loss
        self.sparse_weight = sparse_weight
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }
        
        # 最佳验证准确率
        self.best_val_acc = 0.0
        
        # 早停参数
        self.patience = 10
        self.patience_counter = 0
        
        # 学习率调度器
        self.scheduler = None
    
    def compute_loss(self, outputs, targets, fusion_features=None):
        """
        计算损失。论文 3.3 节：L = L_BCE + λ·||h_fusion||₁

        Args:
            outputs: 模型输出（sigmoid 后的概率值）
            targets: 目标标签
            fusion_features: 交叉注意力输出的融合特征（用于 L1 正则化）

        Returns:
            总损失 = BCE 损失 + L1 正则化损失
        """
        # BCE 损失
        bce_loss = self.criterion(outputs.squeeze(-1), targets)

        # L1 正则化损失 (论文 3.3 节)
        l1_loss = torch.tensor(0.0, device=self.device)
        if fusion_features is not None and self.use_sparse_loss:
            # ||h_fusion||₁ - L1 范数
            l1_loss = torch.mean(torch.abs(fusion_features))

        # 总损失
        total_loss = bce_loss + self.sparse_weight * l1_loss
        return total_loss

    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
                - 普通模式: TensorDataset containing (data, targets)
                - 双输入模式: list of tuples ((data_2d, data_3d), targets)
            epoch: 当前epoch编号

        Returns:
            平均损失、准确率和F1分数
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        if self.use_dual_input:
            # 双输入模式: train_loader 是 ((loader_2d, loader_3d), ...) 格式的列表
            for batch_idx, batch_data in enumerate(train_loader):
                # batch_data 是 ((data_2d, data_3d), targets) 格式
                (data_2d, data_3d), targets = batch_data
                data_2d = data_2d.to(self.device)
                data_3d = data_3d.to(self.device)
                targets = targets.to(self.device)

                # 前向传播 - 使用双输入
                self.optimizer.zero_grad()
                # return_features=True 获取融合特征用于 L1 正则化
                result = self.model(x_spatial=data_2d, x_temporal=data_3d, return_features=True)
                if isinstance(result, tuple):
                    outputs, fusion_features = result
                else:
                    outputs = result
                    fusion_features = None
                loss = self.compute_loss(outputs, targets, fusion_features)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                # sigmoid输出概率值，使用阈值0.5判断类别
                preds = (outputs.squeeze(-1) >= 0.5).float()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        else:
            # 普通模式
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                # return_features=True 获取融合特征用于 L1 正则化
                result = self.model(data, return_features=True)
                if isinstance(result, tuple):
                    outputs, fusion_features = result
                else:
                    outputs = result
                    fusion_features = None
                loss = self.compute_loss(outputs, targets, fusion_features)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                # sigmoid输出概率值，使用阈值0.5判断类别
                preds = (outputs.squeeze(-1) >= 0.5).float()
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        f1 = 100. * f1_score(all_targets, all_preds, average='binary', zero_division=0)

        return avg_loss, accuracy, f1
    
    def validate(self, val_loader):
        """
        在验证集上评估模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            平均损失、准确率和F1分数
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            if self.use_dual_input:
                for batch_data in val_loader:
                    (data_2d, data_3d), targets = batch_data
                    data_2d = data_2d.to(self.device)
                    data_3d = data_3d.to(self.device)
                    targets = targets.to(self.device)

                    # return_features=True 获取融合特征用于 L1 正则化
                    result = self.model(x_spatial=data_2d, x_temporal=data_3d, return_features=True)
                    if isinstance(result, tuple):
                        outputs, fusion_features = result
                    else:
                        outputs = result
                        fusion_features = None
                    loss = self.compute_loss(outputs, targets, fusion_features)

                    total_loss += loss.item()
                    # sigmoid输出概率值，使用阈值0.5判断类别
                    preds = (outputs.squeeze(-1) >= 0.5).float()
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            else:
                for data, targets in val_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(data)
                    # BCELoss需要squeeze
                    loss = self.criterion(outputs.squeeze(-1), targets)

                    total_loss += loss.item()
                    # sigmoid输出概率值，使用阈值0.5判断类别
                    preds = (outputs.squeeze(-1) >= 0.5).float()
                    total += targets.size(0)
                    correct += preds.eq(targets).sum().item()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        f1 = 100. * f1_score(all_targets, all_preds, average='binary', zero_division=0)

        return avg_loss, accuracy, f1
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        early_stopping: bool = False,
        verbose: bool = True,
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            early_stopping: 是否使用早停
            verbose: 是否打印训练信息
            
        Returns:
            训练历史
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")

        # 记录训练开始时间
        train_start_time = time.time()

        # 格式化时间函数
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.1f}秒"
            elif seconds < 3600:
                mins = int(seconds // 60)
                secs = int(seconds % 60)
                return f"{mins}分{secs}秒"
            else:
                hours = int(seconds // 3600)
                mins = int((seconds % 3600) // 60)
                return f"{hours}小时{mins}分"

        for epoch in range(epochs):
            # 记录当前 epoch 开始时间
            epoch_start_time = time.time()

            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            # 计算当前 epoch 耗时
            epoch_elapsed = time.time() - epoch_start_time
            # 累计耗时
            total_elapsed = time.time() - train_start_time
            # 预估剩余时间
            avg_epoch_time = total_elapsed / (epoch + 1)
            remaining_epochs = epochs - epoch - 1
            estimated_remaining = avg_epoch_time * remaining_epochs

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            # 打印信息
            if verbose:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}% | "
                    f"本轮耗时: {format_time(epoch_elapsed)}, 已运行: {format_time(total_elapsed)}, 预估剩余: {format_time(estimated_remaining)}"
                )
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_name = f"best_model_{self.split_mode}.pth"
                self.save_model(best_model_name)
                logger.info(f"Best model saved with Val Acc: {val_acc:.2f}%")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停检查
            if early_stopping and self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
        
        # 计算总训练时间
        total_train_time = time.time() - train_start_time
        logger.info(f"Training completed. Best Val Acc: {self.best_val_acc:.2f}%")
        logger.info(f"总训练时长: {format_time(total_train_time)}")

        return self.history
    
    def evaluate(self, test_loader):
        """
        在测试集上评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            包含评估指标的字典
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            if self.use_dual_input:
                for batch_data in test_loader:
                    (data_2d, data_3d), targets = batch_data
                    data_2d = data_2d.to(self.device)
                    data_3d = data_3d.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(x_spatial=data_2d, x_temporal=data_3d)
                    # sigmoid输出概率值
                    probs = outputs.squeeze(-1)
                    # 阈值0.5判断类别
                    preds = (probs >= 0.5).long()

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            else:
                for data, targets in test_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(data)
                    # sigmoid输出概率值
                    probs = outputs.squeeze(-1)
                    # 阈值0.5判断类别
                    preds = (probs >= 0.5).long()

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
        # 转换为numpy数组
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        auc = roc_auc_score(all_targets, all_probs)
        
        metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "roc_auc": auc,
        }
        
        logger.info("Test Set Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filename: str = "best_model.pth"):
        """
        保存模型权重
        
        Args:
            filename: 保存的文件名
        """
        save_path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str = "best_model.pth"):
        """
        加载模型权重
        
        Args:
            filename: 文件名
        """
        load_path = self.output_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]
        
        logger.info(f"Model loaded from {load_path}")
    
    def save_metrics(self, metrics: dict, filename: str = "metrics.json"):
        """
        保存评估指标
        
        Args:
            metrics: 指标字典
            filename: 文件名
        """
        save_path = self.output_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {save_path}")
