#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
联邦学习防御框架
====================
这个框架实现了带有小波防御机制的联邦学习系统，能够抵抗多种类型的模型中毒攻击。

主要特点:
- 支持多种数据集 (MNIST, Fashion-MNIST, CIFAR-10)
- 支持多种攻击类型 (LIE, Fang, AGR-tailored, MIN-MAX, MIN-SUM)
- 基于小波变换的防御机制
- 详细的实验评估工具
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import seaborn as sns
from datetime import datetime
import logging
import json
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field

from scipy.stats import entropy
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import torchvision.transforms as transforms

# 假设小波防御模块是外部导入的
from wavelet_defense import WaveletDefense


# =========================================================
# 配置和常量
# =========================================================

@dataclass
class FLConfig:
    """联邦学习配置"""
    # 数据集配置
    dataset: str = "cifar10"  # 'mnist', 'fashion-mnist', 'cifar10'
    data_dir: str = "./data"

    # 系统配置
    num_clients: int = 20
    num_attackers: int = 1

    # 训练配置
    batch_size: int = 128
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_epochs: int = 100
    num_local_epochs: int = 1  # 每个客户端本地训练轮数

    # 攻击配置
    attack_type: str = "fang"  # 'lie', 'fang', 'agr', 'min-max', 'min-sum', 'adaptive'
    deviation_type: str = "unit_vec"  # 'unit_vec', 'sign', 'std'
    adaptive_attack_weight: float = 0.5
    adaptive_scale_candidates: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
    attack_debug_enabled: bool = True

    # 防御配置
    defense_enabled: bool = True

    # 数据分布配置
    is_iid: bool = True  # 是否使用IID数据分布
    dirichlet_alpha: float = 1.0  # Dirichlet分布参数，控制non-IID程度

    # 系统配置
    seed: int = 42
    output_dir: str = "results"
    use_gpu: bool = True

    # z值字典，用于LIE攻击
    z_values: Dict[int, float] = field(default_factory=lambda: {
        1: 0.65, 2: 0.67, 3: 0.69847, 5: 0.7054, 8: 0.71904, 10: 0.72575, 12: 0.73891,
        15: 0.75, 20: 0.76, 25: 0.77, 30: 0.78, 35: 0.79, 40: 0.80, 45: 0.81, 50: 0.82
    })

    def __post_init__(self):
        """执行验证并设置派生参数"""
        # 验证攻击类型
        valid_attack_types = ['lie', 'fang', 'agr', 'min-max', 'min-sum', 'adaptive']
        if self.attack_type not in valid_attack_types:
            raise ValueError(f"无效的攻击类型: {self.attack_type}. 有效类型: {valid_attack_types}")

        # 验证偏移类型
        valid_dev_types = ['unit_vec', 'sign', 'std']
        if self.deviation_type not in valid_dev_types:
            raise ValueError(f"无效的偏移类型: {self.deviation_type}. 有效类型: {valid_dev_types}")

        # 验证数据集类型
        valid_datasets = ['mnist', 'fashion-mnist', 'cifar10']
        if self.dataset not in valid_datasets:
            raise ValueError(f"无效的数据集类型: {self.dataset}. 有效类型: {valid_datasets}")

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")


@dataclass
class AttackContext:
    """攻击构造所需的上下文，便于复用与调试。"""
    defender: Optional[WaveletDefense] = None
    benign_feature_prototype: Optional[np.ndarray] = None
    epoch_num: int = 0
    debug_store: Optional[List[Dict[str, Any]]] = None


# =========================================================
# 日志工具
# =========================================================

class LoggerFactory:
    """日志工厂类，用于创建和管理日志记录器"""

    @staticmethod
    def setup_logger(name: str, log_dir: str = 'logs') -> logging.Logger:
        """设置日志记录器

        Args:
            name: 日志记录器的名称
            log_dir: 日志文件保存目录

        Returns:
            配置好的日志记录器
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        # 创建记录器
        logger = logging.getLogger(name)

        # 避免重复配置
        if not logger.handlers:
            logger.setLevel(logging.INFO)

            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(console_handler)

        return logger


# =========================================================
# 模型定义
# =========================================================

class ModelFactory:
    """模型工厂类，用于创建不同任务的模型"""

    @staticmethod
    def create_model(dataset_type: str, config: FLConfig) -> Tuple[nn.Module, optim.Optimizer]:
        """创建适合特定数据集的模型

        Args:
            dataset_type: 数据集类型 ('mnist', 'fashion-mnist', 'cifar10')
            config: 联邦学习配置

        Returns:
            (model, optimizer): 创建的模型和优化器
        """
        device = config.device

        if dataset_type == 'mnist':
            # MNIST简单模型
            model = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
                nn.Linear(9216, 128),
                nn.ReLU(),
                nn.Dropout2d(0.5),
                nn.Linear(128, 10)
            ).to(device)

            weight_decay = 5e-4

        elif dataset_type == 'fashion-mnist':
            # Fashion-MNIST优化模型
            model = SimplifiedFashionMNISTNet(drop_rate=0.3).to(device)
            weight_decay = 1e-4

        elif dataset_type == 'cifar10':
            # CIFAR-10优化的ResNet
            model = ModelFactory.create_resnet34(drop_rate=0.2).to(device)
            weight_decay = 1e-4

        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")

        # 创建优化器
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=weight_decay
        )

        return model, optimizer

    @staticmethod
    def create_resnet18(drop_rate: float = 0.2) -> nn.Module:
        """创建ResNet-18模型"""
        return CIFAR10ResNet(BasicBlock, [2, 2, 2, 2], drop_rate=drop_rate)

    @staticmethod
    def create_resnet34(drop_rate: float = 0.2) -> nn.Module:
        """创建ResNet-34模型"""
        return CIFAR10ResNet(BasicBlock, [3, 4, 6, 3], drop_rate=drop_rate)


# CIFAR-10专用的ResNet模型
class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, drop_rate: float = 0.0):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Dropout层
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)

        out += self.shortcut(identity)
        out = F.relu(out)

        return out


class CIFAR10ResNet(nn.Module):
    """为CIFAR-10优化的ResNet架构"""

    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 10, drop_rate: float = 0.2):
        super(CIFAR10ResNet, self).__init__()
        self.in_planes = 64

        # 初始层 - 适合32x32图像的小卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 主要残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, drop_rate=drop_rate)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int,
                    drop_rate: float) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_rate))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始处理
        out = F.relu(self.bn1(self.conv1(x)))

        # 通过残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 全局池化和分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        avg_pool = self.avg_pool(x).view(batch_size, channels)
        max_pool = self.max_pool(x).view(batch_size, channels)

        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)

        out = avg_out + max_out
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)

        return out


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成空间注意力图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        out = self.sigmoid(out)

        return out


class SimplifiedFashionMNISTNet(nn.Module):
    """为Fashion-MNIST优化的简化网络架构"""

    def __init__(self, drop_rate: float = 0.3):
        super(SimplifiedFashionMNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 轻量级通道注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        # 应用轻量级注意力
        attn = self.attention(x)
        x = x * attn

        x = self.classifier(x)
        return x


# =========================================================
# 学习率调度器
# =========================================================

class StepLRScheduler:
    """阶梯式学习率调度器"""

    def __init__(self, optimizer: optim.Optimizer, init_lr: float = 0.02,
                 gamma: float = 0.5, milestones: List[int] = [40, 80, 120]):
        """初始化学习率调度器

        Args:
            optimizer: 优化器
            init_lr: 初始学习率
            gamma: 学习率衰减因子
            milestones: 学习率下降的轮次列表
        """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.milestones = milestones
        self.current_lr = init_lr

        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.init_lr

    def step(self, epoch: int) -> bool:
        """更新学习率

        Args:
            epoch: 当前轮次

        Returns:
            bool: 学习率是否发生变化
        """
        if epoch in self.milestones:
            self.current_lr *= self.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            return True
        return False

    def get_last_lr(self) -> List[float]:
        """获取当前学习率"""
        return [self.current_lr]


# =========================================================
# 防御评估工具
# =========================================================

class DefenseEvaluator:
    """评估防御机制性能的工具类"""

    def __init__(self, n_attackers: int, n_clients: int):
        """初始化防御评估器

        Args:
            n_attackers: 攻击者数量
            n_clients: 客户端总数
        """
        self.n_attackers = n_attackers
        self.n_clients = n_clients
        self.stats = []

    def evaluate_round(self, selected_indices: List[int], malicious_indices: List[int], epoch: int) -> Dict[str, float]:
        """评估一轮防御的效果

        Args:
            selected_indices: 选中参与聚合的客户端索引
            malicious_indices: 检测到的恶意客户端索引
            epoch: 当前轮次

        Returns:
            Dict: 各项性能指标
        """
        # 假设攻击者总是前n_attackers个客户端
        true_malicious = set(range(self.n_attackers))
        selected = set(selected_indices)
        detected = set(malicious_indices)

        # 计算各种指标
        total_selected = len(selected)
        detected_malicious = len(true_malicious & detected)  # 真阳性：正确检测的恶意客户端
        missed_malicious = len(true_malicious & selected)  # 假阴性：逃过检测的恶意客户端
        false_positives = len(detected - true_malicious)  # 假阳性：误判的良性客户端
        true_negatives = self.n_clients - self.n_attackers - false_positives  # 真阴性：正确识别的良性客户端

        # 计算检测率和误报率
        detection_rate = detected_malicious / self.n_attackers if self.n_attackers > 0 else 1.0
        false_positive_rate = false_positives / (self.n_clients - self.n_attackers) if (
                                                                                               self.n_clients - self.n_attackers) > 0 else 0

        # 计算精确度和召回率
        precision = detected_malicious / len(detected) if len(detected) > 0 else 1.0
        recall = detection_rate  # 召回率等同于检测率

        # 计算F1分数
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 计算准确率
        accuracy = (detected_malicious + true_negatives) / self.n_clients

        round_stats = {
            'epoch': epoch,
            'total_selected': total_selected,
            'detected_malicious': detected_malicious,
            'missed_malicious': missed_malicious,
            'false_positives': false_positives,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }

        self.stats.append(round_stats)
        return round_stats

    def print_round_stats(self, round_stats: Dict[str, float]) -> None:
        """打印单轮统计信息"""
        print("\n=== 防御性能 ===")
        print(f"选中的客户端总数: {round_stats['total_selected']}")
        print(f"检测到的恶意客户端: {round_stats['detected_malicious']}/{self.n_attackers}")
        print(f"逃过检测的恶意客户端: {round_stats['missed_malicious']}")
        print(f"误判的良性客户端: {round_stats['false_positives']}")
        print(f"检测率: {round_stats['detection_rate']:.2%}")
        print(f"误报率: {round_stats['false_positive_rate']:.2%}")
        print(f"精确度: {round_stats['precision']:.2%}")
        print(f"召回率: {round_stats['recall']:.2%}")
        print(f"F1分数: {round_stats['f1_score']:.2%}")
        print(f"准确率: {round_stats['accuracy']:.2%}")

    def get_summary(self) -> Optional[Dict[str, float]]:
        """获取整体统计信息"""
        if not self.stats:
            return None

        df = pd.DataFrame(self.stats)
        summary = {
            'avg_detection_rate': df['detection_rate'].mean(),
            'avg_false_positive_rate': df['false_positive_rate'].mean(),
            'avg_precision': df['precision'].mean(),
            'avg_recall': df['recall'].mean(),
            'avg_f1_score': df['f1_score'].mean(),
            'avg_accuracy': df['accuracy'].mean(),
            'total_missed_malicious': df['missed_malicious'].sum(),
            'avg_selected_clients': df['total_selected'].mean()
        }
        return summary

    def plot_performance(self, save_path: str = 'defense_performance.png') -> None:
        """绘制性能指标随时间的变化

        Args:
            save_path: 图表保存路径
        """
        if not self.stats:
            print("没有统计数据可供绘制")
            return

        df = pd.DataFrame(self.stats)

        plt.figure(figsize=(15, 12))

        # 检测率、误报率、精确度和召回率
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['detection_rate'], 'g-', label='检测率')
        plt.plot(df['epoch'], df['false_positive_rate'], 'r-', label='误报率')
        plt.plot(df['epoch'], df['precision'], 'b--', label='精确度')
        plt.plot(df['epoch'], df['recall'], 'm--', label='召回率')
        plt.xlabel('轮次')
        plt.ylabel('比率')
        plt.title('检测性能')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # F1分数和准确率
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['f1_score'], 'g-', label='F1分数')
        plt.plot(df['epoch'], df['accuracy'], 'b-', label='准确率')
        plt.xlabel('轮次')
        plt.ylabel('分数')
        plt.title('综合性能指标')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 检测到和未检测到的恶意客户端数量
        plt.subplot(2, 2, 3)
        plt.stackplot(df['epoch'],
                      df['detected_malicious'],
                      df['missed_malicious'],
                      labels=['检测到的恶意客户端', '未检测到的恶意客户端'],
                      colors=['g', 'r'])
        plt.plot(df['epoch'], [self.n_attackers] * len(df), 'k--', label='恶意客户端总数')
        plt.xlabel('轮次')
        plt.ylabel('客户端数量')
        plt.title('恶意客户端检测情况')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 选中的客户端数量
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['total_selected'], 'b-', label='选中的客户端总数')
        plt.plot(df['epoch'], df['total_selected'] - df['missed_malicious'], 'g-', label='良性客户端数量')
        plt.plot(df['epoch'], [self.n_clients] * len(df), 'k--', label='客户端总数')
        plt.plot(df['epoch'], [self.n_clients - self.n_attackers] * len(df), 'k:', label='良性客户端总数')
        plt.xlabel('轮次')
        plt.ylabel('客户端数量')
        plt.title('客户端选择情况')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_stats(self, filename: str = 'defense_stats.csv') -> None:
        """保存防御统计数据到CSV文件

        Args:
            filename: 要保存的CSV文件名
        """
        if not self.stats:
            print("没有统计数据可供保存")
            return

        df = pd.DataFrame(self.stats)
        df.to_csv(filename, index=False)
        print(f"统计数据已保存到 {filename}")


# =========================================================
# 攻击策略实现
# =========================================================

class AttackFactory:
    """攻击策略工厂类"""

    @staticmethod
    def create_attack(attack_type: str, config: FLConfig, attack_context: Optional[AttackContext] = None) -> Callable:
        """创建指定类型的攻击函数

        Args:
            attack_type: 攻击类型
            config: 联邦学习配置

        Returns:
            攻击函数
        """
        if attack_type == 'lie':
            return lambda all_updates, model_re, n_attackers: AttackFactory.lie_attack(
                all_updates, config.z_values[n_attackers], config.device
            )
        elif attack_type == 'fang':
            return lambda all_updates, model_re, n_attackers: AttackFactory.fang_attack(
                all_updates, model_re, n_attackers, config.device
            )
        elif attack_type == 'agr':
            return lambda all_updates, model_re, n_attackers: AttackFactory.agr_attack(
                all_updates, model_re, n_attackers, config.deviation_type, config.device
            )
        elif attack_type == 'min-max':
            return lambda all_updates, model_re, n_attackers: AttackFactory.min_max_attack(
                all_updates, model_re, n_attackers, config.deviation_type, config.device
            )
        elif attack_type == 'min-sum':
            return lambda all_updates, model_re, n_attackers: AttackFactory.min_sum_attack(
                all_updates, model_re, n_attackers, config.deviation_type, config.device
            )
        elif attack_type == 'adaptive':
            return lambda all_updates, model_re, n_attackers: AttackFactory.adaptive_attack(
                all_updates, model_re, n_attackers, config, attack_context
            )
        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}")

    @staticmethod
    def lie_attack(all_updates: torch.Tensor, z_value: float, device: torch.device) -> torch.Tensor:
        """LIE攻击 - 向均值的相反方向添加扰动

        Args:
            all_updates: 客户端更新张量 [num_clients, update_dim]
            z_value: 攻击强度参数
            device: 计算设备

        Returns:
            恶意更新张量 [update_dim]
        """
        # 使用detach()创建新的张量，避免视图操作
        updates = all_updates.detach().clone()
        avg = torch.mean(updates, dim=0)
        std = torch.std(updates, dim=0)
        return avg + z_value * std

    @staticmethod
    def compute_fang_lambda(all_updates: torch.Tensor, model_re: torch.Tensor,
                            n_attackers: int, device: torch.device) -> torch.Tensor:
        """计算Fang攻击的lambda参数 - 内存优化版

        Args:
            all_updates: 良性客户端更新
            model_re: 模型更新
            n_attackers: 攻击者数量
            device: 计算设备

        Returns:
            lambda参数
        """
        try:
            n_benign, d = all_updates.shape

            # 极端情况处理：当几乎所有客户端都是攻击者
            if n_benign <= n_attackers + 2:
                print(f"警告: 良性客户端数量太少 ({n_benign} <= {n_attackers + 2}), 使用默认lambda值")
                return torch.tensor(0.5, device=device)

            # 使用 CPU 计算距离矩阵以减少 GPU 内存使用
            cpu_updates = all_updates.detach().cpu()
            distances = torch.zeros((n_benign, n_benign), device="cpu")

            # 分批计算距离以减少内存使用
            batch_size = 10  # 可以根据可用内存调整
            for i in range(0, n_benign, batch_size):
                i_end = min(i + batch_size, n_benign)
                for j in range(n_benign):
                    if i <= j < i_end:  # 只计算上三角部分
                        continue  # 跳过对角线
                    for k in range(i, i_end):
                        if j != k:
                            distances[j, k] = torch.norm(cpu_updates[j] - cpu_updates[k]).item()
                            distances[k, j] = distances[j, k]  # 利用对称性

            # 对每个样本，填充自己到自己的距离为无穷大
            for i in range(n_benign):
                distances[i, i] = float('inf')

            # 将距离矩阵移回 GPU (如果需要)
            # distances = distances.to(device)  # 如果后续计算需要在 GPU 上进行

            # 对每一行排序 - 这一步在 CPU 上进行以节省 GPU 内存
            sorted_distances, _ = torch.sort(distances, dim=1)

            # 计算安全的k值
            k = max(1, min(n_benign - n_attackers - 2, n_benign - 1))

            # 计算得分
            scores = torch.sum(sorted_distances[:, 1:k + 1], dim=1)
            min_score = torch.min(scores)

            # 计算lambda值
            term_1 = min_score / (k * torch.sqrt(torch.tensor(float(d))))

            # 计算到模型更新的最大距离 - 在 CPU 上计算
            model_re_cpu = model_re.detach().cpu()
            dists_to_mean = torch.norm(cpu_updates - model_re_cpu, dim=1)
            max_dist = torch.max(dists_to_mean) / torch.sqrt(torch.tensor(float(d)))

            # 计算最终结果并返回
            lambda_val = term_1 + max_dist
            return lambda_val.to(device)  # 将结果移回 GPU

        except Exception as e:
            print(f"计算lambda参数时出错: {str(e)}")
            return torch.tensor(0.5, device=device)  # 默认值

    @staticmethod
    def fang_attack(all_updates: torch.Tensor, model_re: torch.Tensor,
                    n_attackers: int, device: torch.device) -> torch.Tensor:
        """生成Fang攻击的恶意更新 - 强化版

        Args:
            all_updates: 良性客户端更新
            model_re: 模型更新
            n_attackers: 攻击者数量
            device: 计算设备

        Returns:
            恶意更新张量
        """
        try:
            # 安全检查
            if len(all_updates) == 0:
                print("警告: 没有可用的更新进行Fang攻击")
                return -1.0 * model_re  # 直接返回相反方向的更新

            # 计算偏移方向
            deviation = torch.sign(model_re)

            # 计算lambda参数
            lamda = AttackFactory.compute_fang_lambda(all_updates, model_re, n_attackers, device)

            # 生成恶意更新
            mal_update = model_re - lamda * deviation

            # 检查生成的恶意更新是否有效
            if torch.isnan(mal_update).any() or torch.isinf(mal_update).any():
                print("警告: 恶意更新包含NaN或无穷大值，使用简单的反向更新")
                return -0.5 * model_re

            # 检查恶意更新的范数
            update_norm = torch.norm(mal_update)
            if update_norm > 10 * torch.norm(model_re):
                print("警告: 恶意更新范数过大，进行缩放")
                mal_update = mal_update * (torch.norm(model_re) / update_norm)

            return mal_update

        except Exception as e:
            print(f"Fang攻击生成失败: {str(e)}")
            # 出错时返回一个保守的恶意更新
            return -0.2 * model_re

    @staticmethod
    def get_deviation_vector(model_re: torch.Tensor, all_updates: torch.Tensor,
                             dev_type: str) -> torch.Tensor:
        """获取偏移向量

        Args:
            model_re: 模型更新
            all_updates: 所有客户端更新
            dev_type: 偏移类型

        Returns:
            偏移向量
        """
        if dev_type == 'unit_vec':
            return model_re / torch.norm(model_re)  # 单位向量，方向与良性更新相反
        elif dev_type == 'sign':
            return torch.sign(model_re)
        elif dev_type == 'std':
            return torch.std(all_updates, 0)
        else:
            raise ValueError(f"未知的偏移类型: {dev_type}")

    @staticmethod
    def agr_attack(all_updates: torch.Tensor, model_re: torch.Tensor,
                   n_attackers: int, dev_type: str,
                   device: torch.device) -> torch.Tensor:
        """AGR-tailored攻击 - 针对聚合规则的攻击 (内存优化版)

        Args:
            all_updates: 客户端更新张量
            model_re: 模型更新
            n_attackers: 攻击者数量
            dev_type: 偏移类型
            device: 计算设备

        Returns:
            恶意更新张量
        """
        try:
            # 将计算转移到CPU以减少GPU内存使用
            all_updates_cpu = all_updates.detach().cpu()
            model_re_cpu = model_re.detach().cpu()

            # 获取偏移向量（在CPU上）
            if dev_type == 'unit_vec':
                deviation = model_re_cpu / torch.norm(model_re_cpu)
            elif dev_type == 'sign':
                deviation = torch.sign(model_re_cpu)
            elif dev_type == 'std':
                deviation = torch.std(all_updates_cpu, 0)
            else:
                raise ValueError(f"未知的偏移类型: {dev_type}")

            lamda = torch.tensor([10.0]).float()
            threshold_diff = 1e-5
            lamda_fail = lamda.clone()
            lamda_succ = torch.tensor([0.0]).float()

            # 计算良性更新的统计特征
            benign_updates = all_updates_cpu[n_attackers:]
            if benign_updates.shape[0] == 0:
                print("警告: 没有良性更新可用")
                return model_re  # 如果没有良性更新，返回原始模型更新

            mean_update = torch.mean(benign_updates, dim=0)
            std_update = torch.std(benign_updates, dim=0)

            # 计算良性更新之间的最大距离（分批计算以减少内存使用）
            n_benign = benign_updates.shape[0]
            max_distance = 0.0
            batch_size = 10  # 可以根据可用内存调整

            for i in range(0, n_benign, batch_size):
                i_end = min(i + batch_size, n_benign)
                batch_i = benign_updates[i:i_end]

                for j in range(0, n_benign, batch_size):
                    j_end = min(j + batch_size, n_benign)
                    batch_j = benign_updates[j:j_end]

                    # 计算批次间的距离矩阵
                    for idx_i, update_i in enumerate(batch_i):
                        for idx_j, update_j in enumerate(batch_j):
                            if i + idx_i != j + idx_j:  # 避免计算自己到自己的距离
                                dist = torch.norm(update_i - update_j) ** 2
                                max_distance = max(max_distance, dist.item())

            # 二分搜索找到合适的lambda值
            while torch.abs(lamda_succ - lamda) > threshold_diff:
                # 生成恶意更新
                mal_update = (model_re_cpu - lamda * deviation)

                # 计算与良性更新的距离（分批计算）
                max_d = 0.0
                within_bounds = True

                for i in range(0, n_benign, batch_size):
                    i_end = min(i + batch_size, n_benign)
                    batch = benign_updates[i:i_end]

                    for idx, update in enumerate(batch):
                        # 计算距离
                        dist = torch.norm(update - mal_update) ** 2
                        max_d = max(max_d, dist.item())

                        # 检查统计边界
                        if not torch.all(torch.abs(mal_update - mean_update) <= 3 * std_update):
                            within_bounds = False
                            break

                    if not within_bounds:
                        break

                # 更新lambda值
                if max_d <= max_distance and within_bounds:
                    lamda_succ = lamda.clone()
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2

            # 生成最终的恶意更新
            mal_update = (model_re_cpu - lamda_succ * deviation)

            # 将结果移回GPU
            return mal_update.to(device)

        except Exception as e:
            print(f"AGR攻击生成失败: {str(e)}")
            # 出错时生成一个安全的恶意更新
            return -0.2 * model_re.detach().clone()

    @staticmethod
    def min_max_attack(all_updates: torch.Tensor, model_re: torch.Tensor,
                       n_attackers: int, dev_type: str,
                       device: torch.device) -> torch.Tensor:
        """MIN-MAX攻击 - 最小化最大距离 (内存优化版)

        Args:
            all_updates: 客户端更新张量
            model_re: 模型更新
            n_attackers: 攻击者数量
            dev_type: 偏移类型
            device: 计算设备

        Returns:
            恶意更新张量
        """
        try:
            # 将计算转移到CPU以减少GPU内存使用
            all_updates_cpu = all_updates.detach().cpu()
            model_re_cpu = model_re.detach().cpu()

            # 获取偏移向量（在CPU上）
            if dev_type == 'unit_vec':
                deviation = model_re_cpu / torch.norm(model_re_cpu)
            elif dev_type == 'sign':
                deviation = torch.sign(model_re_cpu)
            elif dev_type == 'std':
                deviation = torch.std(all_updates_cpu, 0)
            else:
                raise ValueError(f"未知的偏移类型: {dev_type}")

            lamda = torch.tensor([10.0]).float()
            threshold_diff = 1e-5
            lamda_fail = lamda.clone()
            lamda_succ = torch.tensor([0.0]).float()

            # 计算所有更新之间的最大距离（分批计算以减少内存使用）
            n_clients = all_updates_cpu.shape[0]
            max_distance = 0.0
            batch_size = 10  # 可以根据可用内存调整

            for i in range(0, n_clients, batch_size):
                i_end = min(i + batch_size, n_clients)
                batch_i = all_updates_cpu[i:i_end]

                for update in all_updates_cpu:
                    for batch_update in batch_i:
                        dist = torch.norm(batch_update - update) ** 2
                        max_distance = max(max_distance, dist.item())

            # 二分搜索找到合适的lambda值
            while torch.abs(lamda_succ - lamda) > threshold_diff:
                # 生成恶意更新
                mal_update = (model_re_cpu - lamda * deviation)

                # 计算与所有更新的最大距离
                max_d = 0.0
                for update in all_updates_cpu:
                    dist = torch.norm(update - mal_update) ** 2
                    max_d = max(max_d, dist.item())

                # 更新lambda值
                if max_d <= max_distance:
                    lamda_succ = lamda.clone()
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2

            # 生成最终的恶意更新
            mal_update = (model_re_cpu - lamda_succ * deviation)

            # 将结果移回GPU
            return mal_update.to(device)

        except Exception as e:
            print(f"MIN-MAX攻击生成失败: {str(e)}")
            # 出错时生成一个安全的恶意更新
            return -0.2 * model_re.detach().clone()

    @staticmethod
    def min_sum_attack(all_updates: torch.Tensor, model_re: torch.Tensor,
                       n_attackers: int, dev_type: str,
                       device: torch.device) -> torch.Tensor:
        """MIN-SUM攻击 - 最小化距离之和 (内存优化版)

        Args:
            all_updates: 客户端更新张量
            model_re: 模型更新
            n_attackers: 攻击者数量
            dev_type: 偏移类型
            device: 计算设备

        Returns:
            恶意更新张量
        """
        try:
            # 将计算转移到CPU以减少GPU内存使用
            all_updates_cpu = all_updates.detach().cpu()
            model_re_cpu = model_re.detach().cpu()

            # 获取偏移向量（在CPU上）
            if dev_type == 'unit_vec':
                deviation = model_re_cpu / torch.norm(model_re_cpu)
            elif dev_type == 'sign':
                deviation = torch.sign(model_re_cpu)
            elif dev_type == 'std':
                deviation = torch.std(all_updates_cpu, 0)
            else:
                raise ValueError(f"未知的偏移类型: {dev_type}")

            lamda = torch.tensor([10.0]).float()
            threshold_diff = 1e-5
            lamda_fail = lamda.clone()
            lamda_succ = torch.tensor([0.0]).float()

            # 计算所有更新之间的距离和得分（分批计算以减少内存使用）
            n_clients = all_updates_cpu.shape[0]
            scores = torch.zeros(n_clients)
            batch_size = 10  # 可以根据可用内存调整

            for i in range(n_clients):
                update_i = all_updates_cpu[i]
                total_dist = 0.0

                for j in range(0, n_clients, batch_size):
                    j_end = min(j + batch_size, n_clients)
                    batch_j = all_updates_cpu[j:j_end]

                    for update_j in batch_j:
                        dist = torch.norm(update_i - update_j) ** 2
                        total_dist += dist.item()

                scores[i] = total_dist

            min_score = torch.min(scores).item()

            # 二分搜索找到合适的lambda值
            while torch.abs(lamda_succ - lamda) > threshold_diff:
                # 生成恶意更新
                mal_update = (model_re_cpu - lamda * deviation)

                # 计算与所有更新的距离之和
                score = 0.0
                for update in all_updates_cpu:
                    dist = torch.norm(update - mal_update) ** 2
                    score += dist.item()

                # 更新lambda值
                if score <= min_score:
                    lamda_succ = lamda.clone()
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2

            # 生成最终的恶意更新
            mal_update = (model_re_cpu - lamda_succ * deviation)

            # 将结果移回GPU
            return mal_update.to(device)

        except Exception as e:
            print(f"MIN-SUM攻击生成失败: {str(e)}")
            # 出错时生成一个安全的恶意更新
            return -0.2 * model_re.detach().clone()

    @staticmethod
    def adaptive_attack(all_updates: torch.Tensor, model_re: torch.Tensor,
                        n_attackers: int, config: FLConfig,
                        attack_context: Optional[AttackContext]) -> torch.Tensor:
        """
        防御感知自适应攻击：
        在原有投毒目标与防御特征伪装目标之间做折中，便于后续比较实验。
        """
        try:
            if len(all_updates) == 0:
                return -0.2 * model_re.detach().clone()

            deviation = AttackFactory.get_deviation_vector(model_re, all_updates, config.deviation_type)
            deviation_norm = torch.norm(deviation)
            if deviation_norm.item() == 0:
                deviation = torch.sign(model_re)
                deviation_norm = torch.norm(deviation) + 1e-12
            deviation = deviation / (deviation_norm + 1e-12)

            benign_mean = torch.mean(all_updates, dim=0)
            baseline_distance = torch.norm(model_re - benign_mean) + 1e-12
            best_candidate = None
            best_score = None
            candidate_debug = []

            for scale in config.adaptive_scale_candidates:
                candidate = model_re - float(scale) * deviation * baseline_distance
                poison_score = torch.norm(candidate - benign_mean).item()
                evade_distance = 0.0

                if attack_context and attack_context.defender and attack_context.benign_feature_prototype is not None:
                    defender = attack_context.defender
                    with torch.no_grad():
                        candidate_updates = torch.stack([candidate.detach()])
                        candidate_probed = defender.generate_probing_parameters(candidate_updates)
                        candidate_features = defender.extract_feature_differences(
                            candidate_updates, candidate_probed
                        )

                    if len(candidate_features) > 0 and len(attack_context.benign_feature_prototype) > 0:
                        evade_distance = float(np.linalg.norm(
                            candidate_features[0] - attack_context.benign_feature_prototype
                        ))

                total_score = poison_score - config.adaptive_attack_weight * evade_distance
                candidate_debug.append({
                    "epoch": attack_context.epoch_num if attack_context else -1,
                    "scale": float(scale),
                    "poison_score": float(poison_score),
                    "evasion_distance": float(evade_distance),
                    "total_score": float(total_score)
                })

                if best_score is None or total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate

            if attack_context and attack_context.debug_store is not None and config.attack_debug_enabled:
                attack_context.debug_store.extend(candidate_debug)

            return best_candidate if best_candidate is not None else (-0.2 * model_re.detach().clone())
        except Exception as e:
            print(f"自适应攻击生成失败: {str(e)}")
            return -0.2 * model_re.detach().clone()


# =========================================================
# 数据加载和处理
# =========================================================

class DataManager:
    """数据管理类 - 负责加载和分割数据集"""

    # 添加到DataManager类中的方法

    @staticmethod
    def create_non_iid_partition(labels, num_clients, alpha, seed=42):
        """
        使用Dirichlet分布创建non-IID数据分区

        Args:
            labels: 标签张量
            num_clients: 客户端数量
            alpha: Dirichlet分布的alpha参数 (alpha越小，分布越不均衡)
            seed: 随机种子

        Returns:
            client_partitions: 列表，包含每个客户端的样本索引
        """
        # 设置随机种子确保可重现性
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 确保labels在CPU上且为numpy数组
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        # 获取类别数和样本总数
        n_classes = len(np.unique(labels_np))
        n_samples = len(labels_np)

        # 创建客户端分区
        client_partitions = [[] for _ in range(num_clients)]

        # 按类别将索引分组
        class_indices = []
        for k in range(n_classes):
            idx_k = np.where(labels_np == k)[0]
            # 打乱每个类别的索引
            np.random.shuffle(idx_k)
            class_indices.append(idx_k)

        # 为每个类别生成Dirichlet分布
        # 这决定了每个客户端获得每个类别的比例
        class_proportions = np.random.dirichlet(np.repeat(alpha, num_clients), n_classes)

        # 分配样本到客户端
        for k in range(n_classes):
            idx_k = class_indices[k]
            proportions = class_proportions[k]

            # 计算每个客户端应获得的该类别样本数
            # 使用累积和来确定边界
            cumu_proportions = np.cumsum(proportions)
            cumu_proportions = np.round(cumu_proportions * len(idx_k)).astype(int)

            # 确保最后一个值等于该类别的样本总数
            cumu_proportions[-1] = len(idx_k)

            # 分配样本索引到客户端
            start_idx = 0
            for i in range(num_clients):
                end_idx = cumu_proportions[i]
                if end_idx > start_idx:  # 确保该客户端获得了样本
                    client_partitions[i].extend(idx_k[start_idx:end_idx].tolist())
                start_idx = end_idx

        # 打乱每个客户端内的索引
        for partition in client_partitions:
            np.random.shuffle(partition)

        # 验证：确保没有索引丢失或重复
        all_indices = []
        for partition in client_partitions:
            all_indices.extend(partition)

        assert len(all_indices) == n_samples, "样本总数不匹配"
        assert len(set(all_indices)) == n_samples, "存在重复索引"

        # 确保每个客户端至少有一个样本
        for i, partition in enumerate(client_partitions):
            if len(partition) == 0:
                # 从样本最多的客户端借用一些样本
                donor_idx = np.argmax([len(p) for p in client_partitions])
                n_donate = max(1, len(client_partitions[donor_idx]) // 20)  # 捐赠5%的样本

                donated_samples = client_partitions[donor_idx][:n_donate]
                client_partitions[i] = donated_samples
                client_partitions[donor_idx] = client_partitions[donor_idx][n_donate:]

                print(f"客户端 {i} 没有样本，从客户端 {donor_idx} 借用了 {n_donate} 个样本")

        return client_partitions

    @staticmethod
    def visualize_label_distribution(client_labels, num_classes, output_dir, title="客户端数据分布"):
        """
        可视化客户端标签分布

        Args:
            client_labels: 每个客户端的标签列表
            num_classes: 类别数量
            output_dir: 输出目录
            title: 图表标题
        """
        num_clients = len(client_labels)

        # 创建标签分布矩阵 [num_clients, num_classes]
        distribution = np.zeros((num_clients, num_classes))

        for i, labels in enumerate(client_labels):
            # 计算每个类别的样本数
            if len(labels) > 0:  # 确保客户端有标签
                for c in range(num_classes):
                    count = torch.sum(labels == c).item()
                    distribution[i, c] = count / len(labels)  # 计算比例

        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(distribution, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.xlabel("类别")
        plt.ylabel("客户端")
        plt.title(title)

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "client_label_distribution.png"), dpi=300)
        plt.close()

        # 计算每个类别在客户端间的分布偏差 (类别不平衡指标)
        class_std = np.std(distribution, axis=0)

        plt.figure(figsize=(10, 6))
        plt.bar(range(num_classes), class_std)
        plt.xlabel("类别")
        plt.ylabel("标准差")
        plt.title("各类别在客户端间的分布偏差")
        plt.xticks(range(num_classes))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_distribution_bias.png"), dpi=300)
        plt.close()

        # 计算每个客户端的类别分布偏差 (客户端不平衡指标)
        client_deviation = np.zeros(num_clients)
        uniform = np.ones(num_classes) / num_classes  # 均匀分布

        for i in range(num_clients):
            # 计算与均匀分布的JS散度
            m = 0.5 * (distribution[i] + uniform)
            client_deviation[i] = 0.5 * (entropy(distribution[i], m) + entropy(uniform, m))

        plt.figure(figsize=(10, 6))
        plt.bar(range(num_clients), client_deviation)
        plt.xlabel("客户端")
        plt.ylabel("与均匀分布的JS散度")
        plt.title("各客户端的分布偏差")
        plt.xticks(range(num_clients))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "client_distribution_bias.png"), dpi=300)
        plt.close()

        return distribution
    @staticmethod
    def load_and_split_data(config: FLConfig) -> Tuple:
        """加载并分割数据集，支持IID和non-IID分布

        Args:
            config: 联邦学习配置

        Returns:
            用户训练数据、验证数据和测试数据
        """
        import random
        dataset = config.dataset
        nusers = config.num_clients
        data_dir = config.data_dir
        is_iid = config.is_iid  # 是否使用IID数据分布
        dirichlet_alpha = config.dirichlet_alpha  # Dirichlet分布参数，控制non-IID程度

        # 设置随机种子以确保可重复性
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        print(f"加载 {dataset} 数据集")

        # 根据数据集类型选择不同的转换和加载方法
        if dataset == 'mnist':
            # 训练集转换 - 可以添加轻微的数据增强
            transform_train = transforms.Compose([
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            # 测试/验证集转换 - 只需要标准化
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform_train)
            test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform_test)

            # MNIST: 训练集60000样本，测试集10000样本
            val_len = 5000
            te_len = 5000
            num_classes = 10

        elif dataset == 'fashion-mnist':
            # 训练集转换 - 添加数据增强
            transform_train = transforms.Compose([
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

            # 测试/验证集转换
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

            train_dataset = FashionMNIST(root=data_dir, train=True, download=True, transform=transform_train)
            test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=transform_test)

            # Fashion-MNIST: 训练集60000样本，测试集10000样本
            val_len = 5000
            te_len = 5000
            num_classes = 10

        elif dataset == 'cifar10':
            # 训练集转换 - 添加CIFAR10典型的数据增强
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            # 测试/验证集转换
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
            test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

            # CIFAR10: 训练集50000样本，测试集10000样本
            val_len = 5000
            te_len = 5000
            num_classes = 10
        else:
            raise ValueError(f"不支持的数据集: {dataset}")

        # 获取原始训练和测试数据（无转换）
        # 注意：这里我们加载不应用转换的数据集，以避免混合不同的预处理
        if dataset == 'mnist':
            raw_train_dataset = MNIST(root=data_dir, train=True, download=True, transform=None)
            raw_test_dataset = MNIST(root=data_dir, train=False, download=True, transform=None)
        elif dataset == 'fashion-mnist':
            raw_train_dataset = FashionMNIST(root=data_dir, train=True, download=True, transform=None)
            raw_test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=None)
        elif dataset == 'cifar10':
            raw_train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=None)
            raw_test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=None)

        print(f"验证集大小: {val_len}")
        print(f"测试集大小: {te_len}")

        # 获取原始数据和标签
        if dataset == 'mnist' or dataset == 'fashion-mnist':
            # 检查数据是否已经是张量(PyTorch 1.x+)
            if isinstance(raw_train_dataset.data, torch.Tensor):
                train_images = raw_train_dataset.data.float() / 255.0  # 归一化到[0,1]
                test_images = raw_test_dataset.data.float() / 255.0

                # 如果需要，添加通道维度(MNIST是灰度图像)
                if train_images.dim() == 3:  # [样本数, 高度, 宽度]
                    train_images = train_images.unsqueeze(1)  # [样本数, 1, 高度, 宽度]
                    test_images = test_images.unsqueeze(1)
            else:
                # 处理旧版PyTorch或不同格式
                train_images = torch.stack([transforms.ToTensor()(img) for img in raw_train_dataset.data])
                test_images = torch.stack([transforms.ToTensor()(img) for img in raw_test_dataset.data])

            if hasattr(raw_train_dataset, 'targets'):
                train_labels = torch.tensor(raw_train_dataset.targets)
                test_labels = torch.tensor(raw_test_dataset.targets)
            else:
                train_labels = torch.tensor(raw_train_dataset.train_labels)
                test_labels = torch.tensor(raw_test_dataset.test_labels)

        elif dataset == 'cifar10':
            # CIFAR10数据为numpy数组，需要转换为张量
            train_images = torch.from_numpy(raw_train_dataset.data).permute(0, 3, 1, 2).float() / 255.0
            test_images = torch.from_numpy(raw_test_dataset.data).permute(0, 3, 1, 2).float() / 255.0

            train_labels = torch.tensor(raw_train_dataset.targets)
            test_labels = torch.tensor(raw_test_dataset.targets)

        # 合并所有数据
        all_images = torch.cat([train_images, test_images])
        all_labels = torch.cat([train_labels, test_labels])

        # 检查各个类别的样本数量
        label_counts = {}
        for label in range(num_classes):
            label_counts[label] = (all_labels == label).sum().item()

        print("全部数据中各类别样本数:")
        for label, count in label_counts.items():
            print(f"  类别 {label}: {count}样本")

        # 分割训练、验证和测试集
        # 先随机打乱索引
        indices = torch.randperm(len(all_labels))

        # 分配样本到验证集和测试集，其余用于训练
        val_indices = indices[:val_len]
        test_indices = indices[val_len:val_len + te_len]
        train_indices = indices[val_len + te_len:]

        # 提取数据
        train_data = all_images[train_indices]
        train_labels = all_labels[train_indices]

        val_data = all_images[val_indices]
        val_labels = all_labels[val_indices]

        test_data = all_images[test_indices]
        test_labels = all_labels[test_indices]

        print(f"训练集大小: {len(train_indices)}")
        print(f"验证集大小: {len(val_indices)}")
        print(f"测试集大小: {len(test_indices)}")

        # 数据分区
        if is_iid:
            print("使用IID数据分布")
            # IID分区: 随机均匀分配
            indices = torch.randperm(len(train_data))

            # 计算每个客户端的样本数
            samples_per_client = len(indices) // nusers

            # 分配样本给每个客户端
            user_train_data = []
            user_train_labels = []

            for i in range(nusers):
                start_idx = i * samples_per_client
                end_idx = min((i + 1) * samples_per_client, len(indices))

                client_indices = indices[start_idx:end_idx]
                user_data = train_data[client_indices]
                user_labels = train_labels[client_indices]

                user_train_data.append(user_data)
                user_train_labels.append(user_labels)

                print(f"客户端 {i}: 分配了 {len(user_data)} 个样本")
        else:
            print(f"使用non-IID数据分布 (Dirichlet alpha={dirichlet_alpha})")

            # 使用专用函数创建non-IID分区
            client_partitions = DataManager.create_non_iid_partition(
                train_labels, nusers, alpha=dirichlet_alpha, seed=config.seed
            )

            # 分配样本给每个客户端
            user_train_data = []
            user_train_labels = []

            for i, partition in enumerate(client_partitions):
                user_data = train_data[partition]
                user_labels = train_labels[partition]

                user_train_data.append(user_data)
                user_train_labels.append(user_labels)

                # 打印每个客户端的样本分布
                classes, counts = torch.unique(user_labels, return_counts=True)
                print(f"客户端 {i}: 总样本数 = {len(user_labels)}")
                for c, count in zip(classes.tolist(), counts.tolist()):
                    print(f"  类别 {c}: {count} 样本 ({count / len(user_labels) * 100:.1f}%)")

        # 应用标准化处理
        if dataset == 'mnist':
            # MNIST标准化
            mean, std = 0.1307, 0.3081
            for i in range(len(user_train_data)):
                user_train_data[i] = (user_train_data[i] - mean) / std
            val_data = (val_data - mean) / std
            test_data = (test_data - mean) / std
        elif dataset == 'fashion-mnist':
            # Fashion-MNIST标准化
            mean, std = 0.2860, 0.3530
            for i in range(len(user_train_data)):
                user_train_data[i] = (user_train_data[i] - mean) / std
            val_data = (val_data - mean) / std
            test_data = (test_data - mean) / std
        elif dataset == 'cifar10':
            # CIFAR10标准化
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            for i in range(len(user_train_data)):
                user_train_data[i] = (user_train_data[i] - mean) / std
            val_data = (val_data - mean) / std
            test_data = (test_data - mean) / std

        # 检查验证集和测试集的类别分布
        val_label_counts = {}
        test_label_counts = {}

        for label in range(num_classes):
            val_label_counts[label] = (val_labels == label).sum().item()
            test_label_counts[label] = (test_labels == label).sum().item()

        print("\n验证集类别分布:")
        for label, count in val_label_counts.items():
            print(f"  类别 {label}: {count}样本 ({count / len(val_labels) * 100:.1f}%)")

        print("\n测试集类别分布:")
        for label, count in test_label_counts.items():
            print(f"  类别 {label}: {count}样本 ({count / len(test_labels) * 100:.1f}%)")

        # 确保数据类型正确
        for i in range(len(user_train_data)):
            user_train_data[i] = user_train_data[i].float()
            user_train_labels[i] = user_train_labels[i].long()

        val_data = val_data.float()
        val_labels = val_labels.long()

        test_data = test_data.float()
        test_labels = test_labels.long()

        # 输出每个集合的形状和类型
        print(f"\n用户训练数据: {len(user_train_data)} 个客户端")
        print(f"用户0数据形状: {user_train_data[0].shape}, 类型: {user_train_data[0].dtype}")
        print(f"用户0标签形状: {user_train_labels[0].shape}, 类型: {user_train_labels[0].dtype}")

        print(f"验证数据形状: {val_data.shape}, 类型: {val_data.dtype}")
        print(f"验证标签形状: {val_labels.shape}, 类型: {val_labels.dtype}")

        print(f"测试数据形状: {test_data.shape}, 类型: {test_data.dtype}")
        print(f"测试标签形状: {test_labels.shape}, 类型: {test_labels.dtype}")

        return user_train_data, user_train_labels, val_data, val_labels, test_data, test_labels
# =========================================================
# 评估与测试
# =========================================================

class Evaluator:
    """模型评估类"""

    @staticmethod
    def test(data_tensor: torch.Tensor, label_tensor: torch.Tensor,
             model: nn.Module, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        """评估模型性能

        Args:
            data_tensor: 测试数据张量
            label_tensor: 测试标签张量
            model: 模型
            criterion: 损失函数
            device: 计算设备

        Returns:
            (loss, accuracy): 测试损失和准确率
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            # 分批处理大型测试集
            batch_size = 500
            num_batches = (len(data_tensor) + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data_tensor))

                data = data_tensor[start_idx:end_idx].to(device)
                labels = label_tensor[start_idx:end_idx].to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * (end_idx - start_idx)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    @staticmethod
    def plot_training_curves(history: Dict, output_dir: str, at_type: str, n_attackers: int) -> None:
        """绘制训练曲线

        Args:
            history: 训练历史记录
            output_dir: 输出目录
            at_type: 攻击类型
            n_attackers: 攻击者数量
        """
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history['epoch'], history['train_loss'], label='训练损失')
        plt.plot(history['epoch'], history['val_loss'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(history['epoch'], history['val_acc'], label='验证准确率')
        plt.plot(history['epoch'], history['test_acc'], label='测试准确率')
        plt.axhline(y=history['best_val_acc'], color='g', linestyle='--',
                    label=f'最佳验证准确率: {history["best_val_acc"]:.2f}%')
        plt.axhline(y=history['best_test_acc'], color='r', linestyle='--',
                    label=f'对应测试准确率: {history["best_test_acc"]:.2f}%')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.title('验证和测试准确率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 学习率曲线
        plt.subplot(2, 2, 3)
        plt.plot(history['epoch'], history['lr'])
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 防御性能曲线
        if history['defense_stats']:
            plt.subplot(2, 2, 4)
            epochs = [stat['epoch'] for stat in history['defense_stats']]
            detection_rates = [stat.get('detection_rate', 0) for stat in history['defense_stats']]
            false_positive_rates = [stat.get('false_positive_rate', 0) for stat in history['defense_stats']]
            precision = [stat.get('precision', 0) for stat in history['defense_stats']]
            recall = [stat.get('recall', 0) for stat in history['defense_stats']]

            plt.plot(epochs, detection_rates, label='检测率')
            plt.plot(epochs, false_positive_rates, label='误报率')
            plt.plot(epochs, precision, label='精确度')
            plt.plot(epochs, recall, label='召回率')
            plt.xlabel('轮次')
            plt.ylabel('比率')
            plt.title('防御性能指标')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_{at_type}_{n_attackers}.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_experiments_comparison(results: Dict, output_dir: str) -> None:
        """绘制不同实验结果的对比图表

        Args:
            results: 实验结果字典
            output_dir: 输出目录
        """
        # 过滤掉失败的实验
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            print("没有有效的实验结果可供比较")
            return

        # 提取结果数据
        exp_names = list(valid_results.keys())

        # 按攻击类型分组
        attack_types = set(v['config']['type'] for v in valid_results.values())

        # 1. 准确率对比
        plt.figure(figsize=(15, 10))

        # 按攻击类型分组绘制条形图
        width = 0.35
        x = np.arange(len(exp_names))

        val_accs = [valid_results[name]['best_val_acc'] for name in exp_names]
        test_accs = [valid_results[name]['best_test_acc'] for name in exp_names]

        plt.bar(x - width / 2, val_accs, width, label='验证准确率')
        plt.bar(x + width / 2, test_accs, width, label='测试准确率')

        plt.xlabel('实验配置')
        plt.ylabel('准确率 (%)')
        plt.title('不同攻击配置下的模型性能对比')
        plt.xticks(x, exp_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 防御性能对比 (如果有)
        defense_results = {k: v for k, v in valid_results.items() if v.get('defense_summary')}

        if defense_results:
            plt.figure(figsize=(15, 12))

            # 准备数据
            exp_names = list(defense_results.keys())
            x = np.arange(len(exp_names))
            metrics = ['avg_detection_rate', 'avg_false_positive_rate', 'avg_precision',
                       'avg_recall', 'avg_f1_score', 'avg_accuracy']
            metric_labels = ['检测率', '误报率', '精确度', '召回率', 'F1分数', '准确率']

            # 绘制每个指标的条形图
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                plt.subplot(2, 3, i + 1)
                values = [defense_results[name]['defense_summary'][metric] for name in exp_names]

                # 根据指标类型设置颜色
                if metric == 'avg_false_positive_rate':
                    colors = ['red' if v > 0.1 else 'orange' if v > 0.05 else 'green' for v in values]
                else:
                    colors = ['red' if v < 0.5 else 'orange' if v < 0.8 else 'green' for v in values]

                plt.bar(x, values, color=colors)
                plt.xlabel('实验配置')
                plt.ylabel(label)
                plt.title(f'平均{label}')
                plt.xticks(x, exp_names, rotation=45, ha='right')

                # 在柱状图上显示具体数值
                for i, v in enumerate(values):
                    plt.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontsize=8)

                plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'defense_performance_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 攻击者数量与防御性能的关系
            attack_numbers = {}

            for name, result in defense_results.items():
                at_type = result['config']['type']
                n_attackers = result['config']['n_attackers']

                if at_type not in attack_numbers:
                    attack_numbers[at_type] = {
                        'n_attackers': [],
                        'detection_rate': [],
                        'false_positive_rate': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'accuracy': [],
                        'test_acc': []
                    }

                attack_numbers[at_type]['n_attackers'].append(n_attackers)
                attack_numbers[at_type]['detection_rate'].append(result['defense_summary']['avg_detection_rate'])
                attack_numbers[at_type]['false_positive_rate'].append(
                    result['defense_summary']['avg_false_positive_rate'])
                attack_numbers[at_type]['precision'].append(result['defense_summary']['avg_precision'])
                attack_numbers[at_type]['recall'].append(result['defense_summary']['avg_recall'])
                attack_numbers[at_type]['f1_score'].append(result['defense_summary']['avg_f1_score'])
                attack_numbers[at_type]['accuracy'].append(result['defense_summary']['avg_accuracy'])
                attack_numbers[at_type]['test_acc'].append(result['best_test_acc'])

            # 绘制每种攻击类型的曲线
            plt.figure(figsize=(20, 15))
            metrics = ['detection_rate', 'false_positive_rate', 'precision', 'recall',
                       'f1_score', 'accuracy', 'test_acc']
            metric_labels = ['检测率', '误报率', '精确度', '召回率', 'F1分数', '防御准确率', '测试准确率']

            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                plt.subplot(3, 3, i + 1)

                for at_type, data in attack_numbers.items():
                    # 排序，确保按攻击者数量递增
                    sorted_indices = np.argsort(data['n_attackers'])
                    x_values = [data['n_attackers'][i] for i in sorted_indices]
                    y_values = [data[metric][i] for i in sorted_indices]

                    if metric == 'test_acc':
                        # 测试准确率需要除以100转换为比例
                        y_values = [y / 100 for y in y_values]

                    plt.plot(x_values, y_values, marker='o', label=at_type)

                plt.xlabel('攻击者数量')
                plt.ylabel(label)
                plt.title(f'攻击者数量与{label}的关系')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'attacker_number_impact.png'), dpi=300, bbox_inches='tight')
            plt.close()


# =========================================================
# 联邦学习主训练框架
# =========================================================

class FederatedLearning:
    """联邦学习主类"""

    def __init__(self, config: FLConfig):
        """初始化联邦学习系统

        Args:
            config: 联邦学习配置
        """
        self.config = config
        self.device = config.device
        self.logger = LoggerFactory.setup_logger('federated_learning',
                                                 os.path.join(config.output_dir, 'logs'))

        # 记录配置信息
        self.logger.info(f"初始化联邦学习系统 - 配置: {vars(config)}")
        self.logger.info(f"使用设备: {self.device}")
        # 初始化轮次计时统计
        self.round_overhead = {}  # 或使用 defaultdict: from collections import defaultdict; self.round_overhead = defaultdict(dict)

    def _fedavg_aggregate(self, global_model: nn.Module, client_models: List[nn.Module],
                          weights: List[float]) -> None:
        """
        使用联邦平均算法聚合客户端模型参数到全局模型

        Args:
            global_model: 全局模型，将被更新
            client_models: 客户端模型列表
            weights: 每个客户端的权重列表
        """
        if not client_models:
            self.logger.warning("没有客户端模型可供聚合")
            return

        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            self.logger.warning("客户端权重总和为零，使用均等权重")
            weights = [1.0 / len(client_models)] * len(client_models)
        else:
            weights = [w / total_weight for w in weights]

        self.logger.info(f"聚合 {len(client_models)} 个客户端模型，权重范围: {min(weights):.4f}-{max(weights):.4f}")

        # 为全局模型的每个参数创建零张量
        global_state_dict = global_model.state_dict()

        # 创建一个备份，以防聚合过程出现问题
        backup_state_dict = {k: v.clone() for k, v in global_state_dict.items()}

        try:
            # 创建一个新的状态字典，用于存储聚合后的参数
            aggregated_state_dict = {}

            # 对每个参数进行处理
            for key in global_state_dict.keys():
                # 批量归一化层的计数器需要特殊处理
                if 'num_batches_tracked' in key:
                    # 对于BatchNorm的计数器，我们取客户端的最大值
                    max_value = 0
                    for client_model in client_models:
                        client_value = client_model.state_dict()[key].item()
                        max_value = max(max_value, client_value)
                    # 确保值是整型
                    aggregated_state_dict[key] = torch.tensor(max_value, dtype=torch.int64,
                                                              device=global_state_dict[key].device)
                else:
                    # 对于其他参数，使用常规的加权平均
                    param_shape = global_state_dict[key].shape
                    param_dtype = global_state_dict[key].dtype
                    param_device = global_state_dict[key].device

                    # 初始化为零
                    aggregated_state_dict[key] = torch.zeros(param_shape, dtype=param_dtype, device=param_device)

                    # 累加每个客户端的加权参数
                    for client_idx, client_model in enumerate(client_models):
                        client_state_dict = client_model.state_dict()

                        # 检查客户端参数是否包含无效值
                        if torch.isnan(client_state_dict[key]).any() or torch.isinf(client_state_dict[key]).any():
                            self.logger.warning(f"客户端 {client_idx} 的参数 {key} 包含无效值，跳过该参数")
                            continue

                        # 添加加权参数，确保类型一致
                        weighted_param = client_state_dict[key] * weights[client_idx]
                        if weighted_param.dtype != param_dtype:
                            weighted_param = weighted_param.to(dtype=param_dtype)

                        aggregated_state_dict[key] += weighted_param

            # 将聚合后的参数加载到全局模型
            global_model.load_state_dict(aggregated_state_dict)

            # 验证聚合后的参数是否合法
            for key, param in global_model.state_dict().items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    self.logger.error(f"聚合后的参数 {key} 包含无效值，恢复备份")
                    global_model.load_state_dict(backup_state_dict)
                    return

            self.logger.info("联邦平均聚合成功完成")

            # 打印一些聚合后参数的统计信息
            for key, param in list(global_model.state_dict().items())[:3]:  # 只打印前几个参数的统计信息
                if param.numel() > 0:  # 确保参数不是空的
                    self.logger.info(
                        f"参数 {key}: 均值={param.float().mean().item():.6f}, 标准差={param.float().std().item():.6f}, " +
                        f"最小值={param.float().min().item():.6f}, 最大值={param.float().max().item():.6f}")

        except Exception as e:
            self.logger.error(f"聚合过程出错: {str(e)}")
            self.logger.info("恢复备份的全局模型参数")
            global_model.load_state_dict(backup_state_dict)
    def train_with_defense(self, user_tr_data_tensors: List[torch.Tensor],
                           user_tr_label_tensors: List[torch.Tensor],
                           val_data_tensor: torch.Tensor, val_label_tensor: torch.Tensor,
                           te_data_tensor: torch.Tensor, te_label_tensor: torch.Tensor,
                           model_fn: Callable) -> Tuple[nn.Module, Dict]:
        """使用小波防御机制进行联邦学习训练

        Args:
            user_tr_data_tensors: 客户端训练数据列表
            user_tr_label_tensors: 客户端训练标签列表
            val_data_tensor: 验证数据张量
            val_label_tensor: 验证标签张量
            te_data_tensor: 测试数据张量
            te_label_tensor: 测试标签张量
            model_fn: 模型构建函数

        Returns:
            (model, history): 训练后的模型和训练历史
        """

        # 定义辅助函数，打印模型参数统计信息

        def print_last_layer_params(model, stage, epoch_num):
            """
            打印模型最后一层的所有参数

            Args:
                model: 要检查的模型
                stage: 阶段标识（"开始"或"结束"）
                epoch_num: 当前轮次
            """
            # 尝试识别最后一层
            last_layer_params = {}

            # 检查常见的最后一层名称
            for name, param in model.named_parameters():
                if 'fc.' in name or 'classifier.' in name or 'output_layer.' in name or 'linear.' in name:
                    last_layer_params[name] = param

            # 如果没有找到明确的最后一层，尝试识别参数名排序中的最后几个
            if not last_layer_params:
                param_names = [name for name, _ in model.named_parameters()]
                if param_names:
                    # 假设最后几个参数属于最后一层
                    last_few = param_names[-2:]  # 通常权重和偏置是一对
                    for name, param in model.named_parameters():
                        if name in last_few:
                            last_layer_params[name] = param

            # 打印识别到的最后一层参数
            if last_layer_params:
                self.logger.info(f"轮次 {epoch_num} {stage}时最后一层参数:")
                for name, param in last_layer_params.items():
                    # 如果参数太大，只打印统计信息和前几个值
                    if param.numel() > 20:
                        stats = {
                            'shape': list(param.shape),
                            'mean': param.data.mean().item(),
                            'std': param.data.std().item(),
                            'min': param.data.min().item(),
                            'max': param.data.max().item(),
                            'first_10_values': param.data.flatten()[:10].cpu().numpy().tolist()
                        }
                        self.logger.info(f"  {name}: {stats}")
                    else:
                        # 如果参数较小，则打印全部值
                        self.logger.info(f"  {name}: {param.data.cpu().numpy().tolist()}")
            else:
                self.logger.warning(f"轮次 {epoch_num} {stage}时无法识别最后一层参数")


        # 设置随机种子
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        # 创建输出目录
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        # 提取配置参数
        n_attackers = self.config.num_attackers
        at_type = self.config.attack_type
        dev_type = self.config.deviation_type
        nepochs = self.config.num_epochs
        batch_size = self.config.batch_size
        fed_lr = self.config.learning_rate
        num_local_epochs = self.config.num_local_epochs
        device = self.device

        # 检查输入数据
        nusers = len(user_tr_data_tensors)
        self.logger.info(f"客户端总数: {nusers}")
        self.logger.info(f"每个客户端训练数据大小: {user_tr_data_tensors[0].size()}")

        if n_attackers > nusers:
            self.logger.warning(f"攻击者数量 ({n_attackers}) 超过了客户端总数 ({nusers})，设置为 {nusers - 1}")
            n_attackers = nusers - 1

        # 计算每个客户端的训练批次数
        user_tr_len = user_tr_data_tensors[0].size(0)
        nbatches = (user_tr_len + batch_size - 1) // batch_size
        self.logger.info(f"每个客户端训练样本数: {user_tr_len}, 批次数: {nbatches}")
        self.logger.info(f"客户端本地训练轮数: {num_local_epochs}")

        # 初始化模型和优化器
        fed_model, optimizer_fed = model_fn(self.config)
        fed_model.to(device)

        # 初始训练前打印模型参数
        #print_model_params_stats(fed_model, "初始全局")

        scheduler = StepLRScheduler(optimizer_fed, init_lr=fed_lr)
        criterion = nn.CrossEntropyLoss()

        # 初始化小波防御器
        input_shape = tuple(user_tr_data_tensors[0][0].size())
        self.logger.info(f"输入形状: {input_shape}")
        input_channels = input_shape[0]  # 获取通道数

        # 为不同数据集选择合适的小波防御器
        if self.config.dataset == 'fashion-mnist':
            self.wavelet_defender = self._create_fashion_mnist_defender(fed_model, nusers, input_shape)
        else:
            self.wavelet_defender = WaveletDefense(fed_model, nusers, input_shape=input_shape, device=device)

        # 初始化防御评估器
        evaluator = DefenseEvaluator(n_attackers, nusers)

        # 训练历史
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'best_val_acc': 0,
            'best_test_acc': 0,
            'defense_stats': [],
            'lr': [],
            'attack_debug': []
        }

        # 主训练循环
        epoch_num = 0
        best_val_acc = 0
        best_test_acc = 0
        epochs_without_improvement = 0
        max_epochs_without_improvement = 50  # 早停参数

        data_weights = [len(data) for data in user_tr_data_tensors]
        self.logger.info(f"客户端数据量分布: {data_weights}")

        start_time = time.time()

        def save_model_snapshot(model, filename):
            """保存模型参数快照"""
            try:
                torch.save(model.state_dict(), filename)
                self.logger.info(f"已保存模型快照到 {filename}")
            except Exception as e:
                self.logger.error(f"保存模型快照失败: {str(e)}")

        # 打印模型参数统计信息
        def print_model_params_stats(model, stage):
            """打印模型参数统计信息"""
            total_params = 0
            param_norms = []
            for name, param in model.named_parameters():
                total_params += param.numel()
                norm = torch.norm(param.data).item()
                param_norms.append(norm)

            avg_norm = np.mean(param_norms) if param_norms else 0
            max_norm = np.max(param_norms) if param_norms else 0
            min_norm = np.min(param_norms) if param_norms else 0

            self.logger.info(
                f"{stage} 模型参数统计 - 总参数数量: {total_params}, 平均范数: {avg_norm:.6f}, 最大范数: {max_norm:.6f}, 最小范数: {min_norm:.6f}")




        # 打印模型的初始状态
        print_model_params_stats(fed_model, "初始全局")
        # 保存初始模型快照
        save_model_snapshot(fed_model, os.path.join(self.config.output_dir, "model_initial.pth"))

        # 保留最佳模型的副本
        best_model_state = None

        while epoch_num < nepochs:
            epoch_start_time = time.time()
            self.logger.info(f"===== 联邦轮次 {epoch_num} =====")
            round_start_time = time.time()

            # 记录当前轮次全局模型参数，用于调试
            print_model_params_stats(fed_model, f"轮次 {epoch_num} 开始时全局")
            print_last_layer_params(fed_model, "开始", epoch_num)

            # 保存当前轮次的模型
            current_model_path = os.path.join(self.config.output_dir, f"model_epoch_{epoch_num}.pth")
            save_model_snapshot(fed_model, current_model_path)

            # 为每个客户端创建本地模型副本并训练
            all_client_models = []
            client_weights = []
            train_loss = 0.0
            client_metrics = []

            self.logger.info("开始客户端本地训练...")

            # 为每个客户端创建本地模型
            for i in range(nusers):
                # 创建客户端本地模型 - 从全局模型复制参数

                client_model, client_optimizer = model_fn(self.config)
                client_model.to(device)
                client_model.load_state_dict(fed_model.state_dict())  # 复制全局模型参数

                # 优化器已经通过model_fn创建，但我们可以调整学习率
                for param_group in client_optimizer.param_groups:
                    param_group['lr'] = fed_lr

                # 标记是否为恶意客户端
                is_malicious = i < n_attackers
                client_type = "恶意" if is_malicious else "良性"

                self.logger.info(f"训练客户端 {i} ({client_type})...")

                # 训练客户端模型
                if not is_malicious:  # 只训练良性客户端
                    client_loss, client_epoch_metrics, model_state_dict = self._train_client_model(
                        client_model, criterion, client_optimizer,
                        user_tr_data_tensors[i], user_tr_label_tensors[i],
                        nbatches, batch_size, user_tr_len, num_local_epochs, device,i
                    )

                    train_loss += client_loss
                    client_metrics.append({
                        'client_id': i,
                        'is_malicious': is_malicious,
                        'average_loss': client_loss,
                        'epoch_metrics': client_epoch_metrics
                    })

                    if client_epoch_metrics:
                        final_metrics = client_epoch_metrics[-1]
                        self.logger.info(
                            f"客户端 {i} 训练完成，平均损失: {client_loss:.4f}, 最终准确率: {final_metrics['accuracy']:.2f}%")
                    else:
                        self.logger.info(f"客户端 {i} 训练完成，平均损失: {client_loss:.4f}")


                # 保存客户端模型和权重
                all_client_models.append(client_model)
                client_weights.append(len(user_tr_data_tensors[i]))

            # 计算良性客户端的平均损失
            benign_count = nusers - n_attackers
            train_loss = train_loss / benign_count if benign_count > 0 else 0
            self.logger.info(f"所有良性客户端的平均训练损失: {train_loss:.4f}")

            # 生成恶意客户端的更新
            if n_attackers > 0:
                self.logger.info(f"生成 {at_type} 恶意更新...")

                # 收集所有良性客户端模型的参数
                benign_models_params = []
                for i in range(n_attackers, nusers):
                    client_model = all_client_models[i]
                    params = []
                    for param in client_model.parameters():
                        params.append(param.data.view(-1))
                    client_params = torch.cat(params)
                    benign_models_params.append(client_params)

                # 计算良性客户端的平均参数向量
                if len(benign_models_params) > 0:
                    benign_updates = torch.stack(benign_models_params)

                    # 提取当前全局模型参数
                    global_params = []
                    for param in fed_model.parameters():
                        global_params.append(param.data.view(-1))
                    global_params = torch.cat(global_params)

                    attack_context = AttackContext(
                        defender=self.wavelet_defender if at_type == "adaptive" else None,
                        epoch_num=epoch_num,
                        debug_store=history['attack_debug']
                    )

                    if at_type == "adaptive":
                        try:
                            benign_probed = self.wavelet_defender.generate_probing_parameters(benign_updates)
                            attack_context.benign_feature_prototype = self.wavelet_defender.estimate_benign_feature_prototype(
                                benign_updates,
                                benign_probed
                            )
                            self.logger.info(
                                f"已估计自适应攻击良性特征原型，维度: {len(attack_context.benign_feature_prototype)}"
                            )
                        except Exception as e:
                            self.logger.warning(f"估计良性特征原型失败，将退化为纯投毒目标: {str(e)}")

                    attack_fn = AttackFactory.create_attack(at_type, self.config, attack_context)

                    # 对每个恶意客户端生成特定的恶意更新
                    for i in range(n_attackers):
                        try:
                            # 生成恶意更新 (这里传入良性客户端的更新和全局模型的当前参数)
                            mal_update = attack_fn(benign_updates, global_params, n_attackers)

                            # 将恶意更新应用到客户端模型
                            mal_client_model = all_client_models[i]

                            # 恶意更新应用到客户端模型参数
                            idx = 0
                            for param in mal_client_model.parameters():
                                param_size = param.numel()
                                param.data = mal_update[idx:idx + param_size].reshape(param.shape).to(device)
                                idx += param_size

                            self.logger.info(f"为恶意客户端 {i} 生成了 {at_type} 攻击更新")

                        except Exception as e:
                            self.logger.error(f"恶意更新生成失败: {str(e)}")

            # 使用小波防御机制检测恶意客户端
            self.logger.info("应用小波防御检测恶意客户端...")

            # 准备客户端模型参数用于检测
            all_client_params = []
            for client_model in all_client_models:
                params = []
                for param in client_model.parameters():
                    params.append(param.data.view(-1))
                client_params = torch.cat(params)
                all_client_params.append(client_params)

            all_client_params_tensor = torch.stack(all_client_params)

            # 生成诱导参数
            probed_updates = self.wavelet_defender.generate_probing_parameters(all_client_params_tensor)
            self.logger.info(f"生成诱导参数完成，形状: {probed_updates.shape}")

            # 检测恶意客户端
            try:
                true_malicious_indices = list(range(n_attackers))
                malicious_indices = self.wavelet_defender.detect_malicious_clients(
                    all_client_params_tensor, probed_updates, epoch_num, true_malicious_indices
                )
                self.logger.info(f"检测到的恶意客户端索引: {malicious_indices}")

                # 选择被参与聚合的客户端索引
                selected_indices = [i for i in range(len(all_client_models)) if i not in malicious_indices]
                self.logger.info(f"选中的客户端索引: {selected_indices}")

                # 评估防御效果
                defense_stats = evaluator.evaluate_round(selected_indices, malicious_indices, epoch_num)
                evaluator.print_round_stats(defense_stats)
                history['defense_stats'].append(defense_stats)

            except Exception as e:
                self.logger.error(f"防御过程发生错误: {str(e)}")
                self.logger.info("回退到选择所有良性客户端...")
                # 如果小波防御出错，选择所有良性客户端
                malicious_indices = list(range(n_attackers))
                selected_indices = list(range(n_attackers, nusers))

            # 记录防御失败情况
                defense_stats = {
                    'epoch': epoch_num,
                    'error': str(e),
                    'detection_rate': 1.0 if n_attackers > 0 else 0,
                    'false_positive_rate': 0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1_score': 1.0,
                    'accuracy': 1.0
                }
                history['defense_stats'].append(defense_stats)

            self.logger.info("聚合选中客户端的模型参数更新全局模型...")

            # 仅使用选定的客户端进行聚合
            if len(selected_indices) > 0:
                # 直接使用FedAvg聚合客户端模型参数
                self._fedavg_aggregate(fed_model, [all_client_models[i] for i in selected_indices],
                                       [client_weights[i] for i in selected_indices])

                self.logger.info(f"已聚合 {len(selected_indices)} 个客户端的模型参数")

                # 保存聚合后的全局模型
                aggregated_model_path = os.path.join(self.config.output_dir, f"model_aggregated_epoch_{epoch_num}.pth")
                save_model_snapshot(fed_model, aggregated_model_path)
            else:
                self.logger.warning("没有选中任何客户端进行聚合！保持全局模型不变")

            # 打印聚合后的全局模型参数统计
            print_model_params_stats(fed_model, f"轮次 {epoch_num} 聚合后全局")
            print_last_layer_params(fed_model, "结束", epoch_num)

            # 更新学习率
            lr_changed = scheduler.step(epoch_num)
            if lr_changed:
                self.logger.info(f"学习率更新为: {scheduler.get_last_lr()[0]}")

            # 记录当前学习率
            history['lr'].append(scheduler.get_last_lr()[0])

            fed_model.eval()

            # 对验证集进行评估
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batch_size = 64  # 使用适当的批次大小

            with torch.no_grad():
                # 分批处理验证数据
                num_val_batches = (len(val_data_tensor) + val_batch_size - 1) // val_batch_size

                for i in range(num_val_batches):
                    start_idx = i * val_batch_size
                    end_idx = min((i + 1) * val_batch_size, len(val_data_tensor))

                    val_inputs = val_data_tensor[start_idx:end_idx].to(device)
                    val_targets = val_label_tensor[start_idx:end_idx].to(device)

                    val_outputs = fed_model(val_inputs)
                    batch_loss = criterion(val_outputs, val_targets)

                    val_loss += batch_loss.item() * (end_idx - start_idx)
                    _, predicted = torch.max(val_outputs, 1)
                    val_total += val_targets.size(0)
                    val_correct += (predicted == val_targets).sum().item()

            # 计算验证集上的平均损失和准确率
            val_loss = val_loss / val_total if val_total > 0 else float('inf')
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

            # 对测试集进行评估
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            test_batch_size = 128  # 使用适当的批次大小

            with torch.no_grad():
                # 分批处理测试数据
                num_test_batches = (len(te_data_tensor) + test_batch_size - 1) // test_batch_size

                for i in range(num_test_batches):
                    start_idx = i * test_batch_size
                    end_idx = min((i + 1) * test_batch_size, len(te_data_tensor))

                    test_inputs = te_data_tensor[start_idx:end_idx].to(device)
                    test_targets = te_label_tensor[start_idx:end_idx].to(device)

                    test_outputs = fed_model(test_inputs)
                    batch_loss = criterion(test_outputs, test_targets)

                    test_loss += batch_loss.item() * (end_idx - start_idx)
                    _, predicted = torch.max(test_outputs, 1)
                    test_total += test_targets.size(0)
                    test_correct += (predicted == test_targets).sum().item()

            # 计算测试集上的平均损失和准确率
            test_loss = test_loss / test_total if test_total > 0 else float('inf')
            test_acc = 100.0 * test_correct / test_total if test_total > 0 else 0.0

            # 记录指标
            self.logger.info(f"验证集: 损失 = {val_loss:.4f}, 准确率 = {val_acc:.2f}%")
            self.logger.info(f"测试集: 损失 = {test_loss:.4f}, 准确率 = {test_acc:.2f}%")

            # 添加诊断信息
            self.logger.info("全局模型评估诊断:")
            # 随机抽样5个样本输出预测值与真实值比较
            with torch.no_grad():
                random_indices = torch.randperm(len(val_data_tensor))[:5]
                sample_inputs = val_data_tensor[random_indices].to(device)
                sample_targets = val_label_tensor[random_indices].to(device)

                sample_outputs = fed_model(sample_inputs)
                _, sample_preds = torch.max(sample_outputs, 1)

                for i in range(len(random_indices)):
                    self.logger.info(f"样本 {i}: 预测 = {sample_preds[i].item()}, 真实 = {sample_targets[i].item()}, " +
                                     f"{'正确' if sample_preds[i] == sample_targets[i] else '错误'}")
            # 更新历史记录
            history['epoch'].append(epoch_num)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # 检查是否是最佳模型
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_test_acc = test_acc
                history['best_val_acc'] = best_val_acc
                history['best_test_acc'] = best_test_acc

                # 保存最佳模型状态
                best_model_state = {k: v.clone() for k, v in fed_model.state_dict().items()}

                # 保存最佳模型
                best_model_path = os.path.join(self.config.output_dir, f'best_model_{at_type}_{n_attackers}.pth')
                torch.save(fed_model.state_dict(), best_model_path)
                self.logger.info(f"保存新的最佳模型，验证准确率: {best_val_acc:.2f}%, 测试准确率: {best_test_acc:.2f}%")

                # 重置早停计数器
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                self.logger.info(f"模型性能未提升，已经{epochs_without_improvement}轮未改善")

            # 早停检查
            if epochs_without_improvement >= max_epochs_without_improvement:
                self.logger.info(f"早停：已连续{epochs_without_improvement}轮没有改善，停止训练")
                break

            # 计算轮次耗时
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"轮次{epoch_num}完成，耗时: {epoch_time:.2f}秒")

            # 定期清理GPU内存
            if torch.cuda.is_available() and epoch_num % 10 == 0:
                torch.cuda.empty_cache()
                self.logger.info("已清理GPU缓存")

            round_compute_time = time.time() - round_start_time
            self.round_overhead[f"epoch_{epoch_num}"] = {
                "total_compute_time": round_compute_time,
                "num_clients": nusers,
                "num_attackers": n_attackers,
                "defense_enabled": self.config.defense_enabled,
                "validation_accuracy": val_acc,
                "test_accuracy": test_acc
            }
            self.logger.info(f"轮次 {epoch_num} 完成，耗时: {round_compute_time:.2f} 秒")

            epoch_num += 1

        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        self.logger.info(f"最佳验证准确率: {best_val_acc:.2f}%, 对应测试准确率: {best_test_acc:.2f}%")

        # 如果保存了最佳模型，恢复它
        if best_model_state is not None:
            fed_model.load_state_dict(best_model_state)
            self.logger.info("已恢复最佳模型状态")

        # 绘制训练曲线
        Evaluator.plot_training_curves(history, self.config.output_dir, at_type, n_attackers)

        # 绘制防御性能
        evaluator.plot_performance(
            os.path.join(self.config.output_dir, f'defense_performance_{at_type}_{n_attackers}.png'))

        # 保存防御统计数据
        evaluator.save_stats(os.path.join(self.config.output_dir, f'defense_stats_{at_type}_{n_attackers}.csv'))

        # 保存小波防御器的统计信息
        self.wavelet_defender.save_statistics(
            os.path.join(self.config.output_dir, f'wavelet_stats_{at_type}_{n_attackers}.json'))

        # 保存历史记录
        history_path = os.path.join(self.config.output_dir, f'history_{at_type}_{n_attackers}.json')
        with open(history_path, 'w') as f:
            json.dump(FederatedLearning._convert_to_serializable(history), f, indent=2)

        self.logger.info(f"历史记录已保存到 {history_path}")

        # 打印小波防御器的性能总结
        self.wavelet_defender.print_performance_summary()

        # 打印防御评估器的总结
        defense_summary = evaluator.get_summary()
        if defense_summary:
            self.logger.info("\n===== 防御性能总结 =====")
            for key, value in defense_summary.items():
                if 'rate' in key or 'precision' in key or 'recall' in key or 'score' in key or 'accuracy' in key:
                    self.logger.info(f"{key}: {value:.2%}")
                else:
                    self.logger.info(f"{key}: {value:.2f}")

        self.wavelet_defender.save_overhead_statistics(
            os.path.join(self.config.output_dir, f'client_overhead_{at_type}_{n_attackers}.json'))

        # 保存轮次开销统计信息
        round_overhead_path = os.path.join(self.config.output_dir, f'round_overhead_{at_type}_{n_attackers}.json')
        self.save_round_overhead_statistics(round_overhead_path)

        return fed_model, history


    def _train_client_model(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer,
                            client_data: torch.Tensor, client_labels: torch.Tensor,
                            nbatches: int, batch_size: int, data_len: int,
                            num_local_epochs: int, device: torch.device,client_id: int) -> Tuple[float, List[Dict]]:
        """训练单个客户端模型并返回训练指标

        Args:
            model: 客户端本地模型
            criterion: 损失函数
            optimizer: 优化器
            client_data: 客户端数据
            client_labels: 客户端标签
            nbatches: 批次数
            batch_size: 批次大小
            data_len: 数据长度
            num_local_epochs: 本地训练轮数
            device: 计算设备

        Returns:
            (client_loss, epoch_metrics): 客户端损失和每轮指标
        """
        client_loss = 0.0
        num_batches_processed = 0

        client_start_time = time.time()
        # 用于存储每轮的指标
        epoch_metrics = []

        # 确保数据长度是正确的
        actual_data_len = len(client_data)
        if actual_data_len != data_len:
            self.logger.warning(f"客户端数据长度不匹配: 期望 {data_len}, 实际 {actual_data_len}. 使用实际长度.")
            data_len = actual_data_len

        # 安全检查：确保有足够的数据进行训练
        if data_len == 0:
            self.logger.warning("客户端没有数据，跳过训练")
            return 0.0, [], {}

        # 调整批量大小以适应小数据集
        actual_batch_size = min(batch_size, data_len)
        # 重新计算批次数
        actual_nbatches = (data_len + actual_batch_size - 1) // actual_batch_size

        # 对每个客户端进行本地训练
        for local_epoch in range(num_local_epochs):
            # 每轮的指标统计
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_batches = 0

            # 随机打乱客户端本地数据
            indices = torch.randperm(data_len)

            # 处理所有批次
            for batch_idx in range(actual_nbatches):
                start_idx = batch_idx * actual_batch_size
                end_idx = min((batch_idx + 1) * actual_batch_size, data_len)

                # 如果批次为空或只有一个样本（对于某些模型可能不够），跳过
                if start_idx >= end_idx or end_idx - start_idx < 2:
                    continue

                try:
                    batch_indices = indices[start_idx:end_idx]
                    inputs = client_data[batch_indices].to(device)
                    targets = client_labels[batch_indices].to(device)

                    # 前向传播
                    model.train()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # 计算准确度
                    _, predicted = torch.max(outputs.data, 1)
                    batch_correct = (predicted == targets).sum().item()
                    batch_total = targets.size(0)

                    # 累计统计
                    epoch_loss += loss.item()
                    epoch_correct += batch_correct
                    epoch_total += batch_total
                    epoch_batches += 1

                    client_loss += loss.item()
                    num_batches_processed += 1

                    # 检查损失是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except Exception as e:
                    self.logger.error(f"训练批次时出错: {str(e)}")
                    self.logger.error(f"批次信息: start_idx={start_idx}, end_idx={end_idx}, data_len={data_len}")
                    continue

            # 计算当前轮的平均损失和准确度
            if epoch_batches > 0:
                epoch_avg_loss = epoch_loss / epoch_batches
                epoch_accuracy = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0

                # 存储当前轮的指标
                epoch_metrics.append({
                    'epoch': local_epoch,
                    'loss': epoch_avg_loss,
                    'accuracy': epoch_accuracy,
                    'correct': epoch_correct,
                    'total': epoch_total
                })
                self.logger.info(
                    f"  - 本地轮次 {local_epoch}/{num_local_epochs - 1}: 损失 = {epoch_avg_loss:.4f}, 准确率 = {epoch_accuracy:.2f}%")

        # 计算客户端平均损失
        if num_batches_processed > 0:
            client_loss /= num_batches_processed

        client_compute_time = time.time() - client_start_time

        # 计算模型参数大小（KB）
        model_size_bytes = 0
        for param in model.parameters():
            model_size_bytes += param.numel() * param.element_size()
        model_size_kb = model_size_bytes / 1024  # 转换为KB

        # 记录客户端开销
        if hasattr(self, 'wavelet_defender'):
            self.wavelet_defender.record_client_overhead(client_id, client_compute_time, model_size_kb)

        return client_loss, epoch_metrics, model.state_dict()


    def _create_fashion_mnist_defender(self, model: nn.Module, nusers: int,
                                       input_shape: Tuple, epsilon: float = 0.01) -> WaveletDefense:
        """为Fashion-MNIST创建定制的小波防御器

        Args:
            model: 全局模型
            nusers: 客户端数量
            input_shape: 输入形状
            epsilon: 扰动大小

        Returns:
            定制的小波防御器
        """

        # 继承WaveletDefense类并重写部分方法
        class CustomWaveletDefender(WaveletDefense):
            def generate_probing_parameters(self, client_updates):
                """针对Fashion-MNIST优化的诱导参数生成"""
                start_time = time.time()
                probing_params = []

                # 保存当前模型状态
                was_training = self.model.training
                self.model.eval()  # 切换到评估模式

                for update in client_updates:
                    try:
                        # 创建批次大小为2的假输入数据
                        fake_input = torch.randn(2, *self.input_shape).to(self.device)
                        fake_target = torch.randint(0, 10, (2,)).to(self.device)

                        # 计算FGSM扰动
                        self.model.zero_grad()
                        with torch.set_grad_enabled(True):
                            output = self.model(fake_input)
                            loss = F.cross_entropy(output, fake_target)
                            loss.backward()

                        # 收集梯度
                        perturbation = []
                        for param in self.model.parameters():
                            if param.grad is not None:
                                pert = self.epsilon * torch.sign(param.grad.data)
                                perturbation.append(pert.view(-1))
                            else:
                                perturbation.append(torch.zeros(param.numel(), device=self.device))

                        # 连接梯度
                        perturbation = torch.cat(perturbation)

                        # 调整大小匹配
                        if perturbation.shape[0] > update.shape[0]:
                            perturbation = perturbation[:update.shape[0]]
                        elif perturbation.shape[0] < update.shape[0]:
                            padding = torch.zeros(update.shape[0] - perturbation.shape[0], device=self.device)
                            perturbation = torch.cat([perturbation, padding])

                        # 添加扰动
                        probed_update = update + perturbation.to(update.device)
                        probing_params.append(probed_update)

                    except Exception as e:
                        print(f"生成诱导参数时出错: {str(e)}")
                        random_noise = torch.randn_like(update) * self.epsilon
                        probed_update = update + random_noise
                        probing_params.append(probed_update)

                # 恢复模型状态
                if was_training:
                    self.model.train()

                self.execution_time['generate_probing'] = time.time() - start_time
                return torch.stack(probing_params)

        return CustomWaveletDefender(model, nusers, input_shape=input_shape, device=self.device)

    @staticmethod
    def _convert_to_serializable(obj):
        """将对象转换为可序列化的形式

        Args:
            obj: 要转换的对象

        Returns:
            可序列化的对象
        """
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: FederatedLearning._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FederatedLearning._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def run_attack_experiments(self, user_tr_data_tensors: List[torch.Tensor],
                               user_tr_label_tensors: List[torch.Tensor],
                               val_data_tensor: torch.Tensor, val_label_tensor: torch.Tensor,
                               te_data_tensor: torch.Tensor, te_label_tensor: torch.Tensor,
                               model_fn: Callable, attack_configs: List[Dict]) -> Dict:
        """
        运行一系列攻击和防御实验

        Args:
            user_tr_data_tensors: 客户端训练数据列表
            user_tr_label_tensors: 客户端训练标签列表
            val_data_tensor: 验证数据张量
            val_label_tensor: 验证标签张量
            te_data_tensor: 测试数据张量
            te_label_tensor: 测试标签张量
            model_fn: 模型构建函数
            attack_configs: 攻击配置列表，每个配置是一个字典，包含 'type', 'n_attackers', 'dev_type'

        Returns:
            results: 所有实验的结果字典
        """
        # 创建主输出目录
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        # 设置主日志
        main_logger = LoggerFactory.setup_logger('experiment_manager',
                                                 os.path.join(self.config.output_dir, 'main_log'))
        main_logger.info(f"开始运行攻击实验，共{len(attack_configs)}个配置")

        results = {}

        for i, config in enumerate(attack_configs):
            at_type = config['type']
            n_attackers = config['n_attackers']
            dev_type = config.get('dev_type', 'unit_vec')

            exp_name = f"{at_type}_{n_attackers}"
            if at_type in ['agr', 'min-max', 'min-sum']:
                exp_name += f"_{dev_type}"

            main_logger.info(f"========== 实验 {i + 1}/{len(attack_configs)}: {exp_name} ==========")

            # 为每个实验创建子目录
            exp_dir = os.path.join(self.config.output_dir, exp_name)
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            # 创建实验专用配置
            exp_config = FLConfig(
                dataset=self.config.dataset,
                data_dir=self.config.data_dir,
                num_clients=self.config.num_clients,
                num_attackers=n_attackers,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                num_epochs=self.config.num_epochs,
                num_local_epochs=self.config.num_local_epochs,
                attack_type=at_type,
                deviation_type=dev_type,
                adaptive_attack_weight=self.config.adaptive_attack_weight,
                adaptive_scale_candidates=self.config.adaptive_scale_candidates,
                attack_debug_enabled=self.config.attack_debug_enabled,
                defense_enabled=self.config.defense_enabled,
                seed=self.config.seed + i,  # 不同实验使用不同的种子
                output_dir=exp_dir,
                use_gpu=self.config.use_gpu,
                z_values=self.config.z_values
            )

            # 运行实验
            try:
                # 创建实验专用联邦学习器
                fl_runner = FederatedLearning(exp_config)

                # 执行训练
                model, history = fl_runner.train_with_defense(
                    user_tr_data_tensors=user_tr_data_tensors,
                    user_tr_label_tensors=user_tr_label_tensors,
                    val_data_tensor=val_data_tensor,
                    val_label_tensor=val_label_tensor,
                    te_data_tensor=te_data_tensor,
                    te_label_tensor=te_label_tensor,
                    model_fn=model_fn
                )

                data_distribution = [len(data) for data in user_tr_data_tensors]
                results[exp_name] = {
                    'config': config,
                    'best_val_acc': history['best_val_acc'],
                    'best_test_acc': history['best_test_acc'],
                    'final_epoch': history['epoch'][-1],
                    'defense_summary': None,
                    'data_distribution': data_distribution  # 添加数据分布信息
                }

                # 如果有防御评估器的结果，添加到结果中
                if history['defense_stats']:
                    defense_summary = {
                        'avg_detection_rate': np.mean(
                            [stat.get('detection_rate', 0) for stat in history['defense_stats']]),
                        'avg_false_positive_rate': np.mean(
                            [stat.get('false_positive_rate', 0) for stat in history['defense_stats']]),
                        'avg_precision': np.mean([stat.get('precision', 0) for stat in history['defense_stats']]),
                        'avg_recall': np.mean([stat.get('recall', 0) for stat in history['defense_stats']]),
                        'avg_f1_score': np.mean([stat.get('f1_score', 0) for stat in history['defense_stats']]),
                        'avg_accuracy': np.mean([stat.get('accuracy', 0) for stat in history['defense_stats']])
                    }
                    results[exp_name]['defense_summary'] = defense_summary

                main_logger.info(f"实验 {exp_name} 完成")
                main_logger.info(
                    f"最佳验证准确率: {history['best_val_acc']:.2f}%, 对应测试准确率: {history['best_test_acc']:.2f}%")

            except Exception as e:
                main_logger.error(f"实验 {exp_name} 失败: {str(e)}")
                results[exp_name] = {
                    'config': config,
                    'error': str(e)
                }

        # 保存所有实验的总结
        summary_path = os.path.join(self.config.output_dir, 'experiments_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(self._convert_to_serializable(results), f, indent=2)

        main_logger.info(f"所有实验完成，结果保存在 {summary_path}")

        # 生成实验对比图表
        Evaluator.plot_experiments_comparison(results, self.config.output_dir)

        return results

    def save_round_overhead_statistics(self, filename='round_overhead_stats.json'):
        """
        保存轮次计时统计信息到文件

        Args:
            filename: 输出文件路径
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.round_overhead, f, indent=2)
            self.logger.info(f"轮次开销统计信息已保存到 {filename}")
        except Exception as e:
            self.logger.error(f"保存轮次开销统计信息失败: {str(e)}")
# =========================================================
# 主程序入口
# =========================================================

def main():
    """主程序入口"""
    # 配置联邦学习参数
    config = FLConfig(
        dataset='cifar10',
        data_dir='./data',
        num_clients=20,
        num_attackers=10,
        batch_size=64,
        learning_rate=0.02,
        momentum=0.9,
        weight_decay=5e-4,
        num_epochs=5,
        num_local_epochs=3,
        attack_type='lie',
        deviation_type='unit_vec',
        defense_enabled=True,
        seed=42,
        output_dir='results_cifar10',
        use_gpu=True,
        is_iid=True,  # 使用non-IID分布
        dirichlet_alpha=0.01  # 设置non-IID程度，可以调整
    )

    # 初始化联邦学习系统
    fl_system = FederatedLearning(config)

    user_tr_data_tensors, user_tr_label_tensors, val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor = DataManager.load_and_split_data(
        config)

    # 可视化客户端数据分布
    num_classes = 10  # CIFAR-10有10个类别
    DataManager.visualize_label_distribution(
        user_tr_label_tensors,
        num_classes,
        config.output_dir,
        f"客户端数据分布 (Dirichlet alpha={config.dirichlet_alpha})"
    )



    # 定义模型构建函数
    model_builder = lambda config: ModelFactory.create_model(config.dataset, config)

    # 运行单次实验
    # model, history = fl_system.train_with_defense(
    #     user_tr_data_tensors, user_tr_label_tensors,
    #     val_data_tensor, val_label_tensor,
    #     te_data_tensor, te_label_tensor,
    #     model_builder
    # )

    # 定义多个攻击配置并运行实验
    attack_configs = [
        {'type': 'fang', 'n_attackers': 1},
        # {'type': 'fang', 'n_attackers': 12},
        # {'type': 'fang', 'n_attackers': 14},
        # {'type': 'fang', 'n_attackers': 16},
        # {'type': 'fang', 'n_attackers': 18},
    ]

    results = fl_system.run_attack_experiments(
        user_tr_data_tensors, user_tr_label_tensors,
        val_data_tensor, val_label_tensor,
        te_data_tensor, te_label_tensor,
        model_builder, attack_configs
    )

    print("所有实验完成!")


if __name__ == "__main__":
    main()
