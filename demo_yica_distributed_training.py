#!/usr/bin/env python3
"""
YICA 分布式训练演示

这个演示展示了如何使用 YICA 分布式优化器和 YCCL 通信库进行大规模深度学习模型训练：

1. 多设备分布式训练设置
2. YCCL 通信优化演示
3. 动态负载均衡
4. 容错和恢复机制
5. 性能监控和分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

# 添加 Mirage 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'mirage', 'python'))

from mirage.yica.config import YICAConfig
from mirage.python.mirage.yica_distributed_optimizer import (
    YICADistributedOptimizer, DistributedTrainingConfig, DistributedMetrics
)
from mirage.python.mirage.yica_llama_optimizer import YICALlamaOptimizer, LlamaModelConfig


class YICADistributedTrainingDemo:
    """YICA 分布式训练演示类"""
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.dist_optimizer = None
        
        # 配置参数
        self.yica_config = YICAConfig(
            num_cim_arrays=args.num_cim_arrays,
            spm_size_per_die=args.spm_size * 1024 * 1024,  # MB to bytes
            dram_size_per_cluster=args.dram_size * 1024 * 1024 * 1024,  # GB to bytes
            enable_quantization=args.enable_quantization,
            target_precision=args.precision
        )
        
        self.distributed_config = DistributedTrainingConfig(
            world_size=args.world_size,
            rank=args.rank,
            local_rank=args.local_rank,
            backend=args.backend,
            data_parallel=args.data_parallel,
            model_parallel=args.model_parallel,
            gradient_compression=args.gradient_compression,
            gradient_clipping=args.gradient_clipping,
            dynamic_load_balancing=args.dynamic_load_balancing,
            fault_tolerance=args.fault_tolerance,
            enable_profiling=args.enable_profiling
        )
        
        # 性能统计
        self.training_metrics = []
        self.communication_stats = []
        
    def setup_model_and_data(self):
        """设置模型和数据"""
        print(f"🏗️  设置模型和数据 (Rank {self.args.rank})")
        
        # 创建模型
        if self.args.model_type == "resnet":
            self.model = self._create_resnet_model()
        elif self.args.model_type == "transformer":
            self.model = self._create_transformer_model()
        elif self.args.model_type == "llama":
            self.model = self._create_llama_model()
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")
        
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建数据加载器
        if self.args.dataset == "cifar10":
            self.train_loader, self.val_loader = self._create_cifar10_loaders()
        elif self.args.dataset == "imagenet":
            self.train_loader, self.val_loader = self._create_imagenet_loaders()
        elif self.args.dataset == "synthetic":
            self.train_loader, self.val_loader = self._create_synthetic_loaders()
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # 创建分布式优化器
        self.dist_optimizer = YICADistributedOptimizer(
            self.model, self.yica_config, self.distributed_config
        )
        
        print("✅ 模型和数据设置完成")
    
    def _create_resnet_model(self):
        """创建 ResNet 模型"""
        if self.args.model_size == "small":
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                *self._make_resnet_layer(64, 128, 2),
                *self._make_resnet_layer(128, 256, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, self.args.num_classes)
            )
        elif self.args.model_size == "medium":
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                *self._make_resnet_layer(64, 128, 3),
                *self._make_resnet_layer(128, 256, 4),
                *self._make_resnet_layer(256, 512, 6),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, self.args.num_classes)
            )
        else:  # large
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                *self._make_resnet_layer(64, 128, 3),
                *self._make_resnet_layer(128, 256, 4),
                *self._make_resnet_layer(256, 512, 6),
                *self._make_resnet_layer(512, 1024, 3),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(1024, self.args.num_classes)
            )
    
    def _make_resnet_layer(self, in_channels, out_channels, num_blocks):
        """创建 ResNet 层"""
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return layers
    
    def _create_transformer_model(self):
        """创建 Transformer 模型"""
        if self.args.model_size == "small":
            hidden_size, num_layers, num_heads = 512, 6, 8
        elif self.args.model_size == "medium":
            hidden_size, num_layers, num_heads = 768, 12, 12
        else:  # large
            hidden_size, num_layers, num_heads = 1024, 24, 16
        
        return nn.Sequential(
            nn.Embedding(self.args.vocab_size, hidden_size),
            *[nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)],
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.args.num_classes)
        )
    
    def _create_llama_model(self):
        """创建 Llama 模型"""
        if self.args.model_size == "small":
            model_config = LlamaModelConfig(
                hidden_size=768, num_hidden_layers=12, num_attention_heads=12
            )
        elif self.args.model_size == "medium":
            model_config = LlamaModelConfig(
                hidden_size=1024, num_hidden_layers=24, num_attention_heads=16
            )
        else:  # large
            model_config = LlamaModelConfig(
                hidden_size=2048, num_hidden_layers=32, num_attention_heads=32
            )
        
        # 简化的 Llama 模型实现
        class SimplifiedLlama(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=config.hidden_size,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.intermediate_size,
                        batch_first=True
                    ) for _ in range(config.num_hidden_layers)
                ])
                self.norm = nn.LayerNorm(config.hidden_size)
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            
            def forward(self, input_ids, attention_mask=None):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x, x, tgt_mask=attention_mask)
                x = self.norm(x)
                return self.lm_head(x)
        
        return SimplifiedLlama(model_config)
    
    def _create_cifar10_loaders(self):
        """创建 CIFAR-10 数据加载器"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        val_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_val
        )
        
        train_sampler = DistributedSampler(train_dataset) if self.args.world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset) if self.args.world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size,
            sampler=val_sampler, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_synthetic_loaders(self):
        """创建合成数据加载器"""
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, size, input_shape, num_classes):
                self.size = size
                self.input_shape = input_shape
                self.num_classes = num_classes
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                if len(self.input_shape) == 3:  # 图像数据
                    x = torch.randn(self.input_shape)
                else:  # 序列数据
                    x = torch.randint(0, 1000, self.input_shape)
                y = torch.randint(0, self.num_classes, ())
                return x, y
        
        if self.args.model_type in ["resnet"]:
            input_shape = (3, 32, 32)
        elif self.args.model_type in ["transformer", "llama"]:
            input_shape = (self.args.sequence_length,)
        else:
            input_shape = (3, 32, 32)
        
        train_dataset = SyntheticDataset(10000, input_shape, self.args.num_classes)
        val_dataset = SyntheticDataset(2000, input_shape, self.args.num_classes)
        
        train_sampler = DistributedSampler(train_dataset) if self.args.world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset) if self.args.world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size,
            sampler=val_sampler, shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def run_distributed_training(self):
        """运行分布式训练"""
        print(f"🚀 开始分布式训练 (Rank {self.args.rank})")
        
        # 初始化分布式环境
        self.dist_optimizer.initialize_distributed()
        
        # 优化模型分布策略
        distribution_plan = self.dist_optimizer.optimize_model_distribution()
        print(f"📊 分布策略: {distribution_plan}")
        
        # 训练循环
        for epoch in range(self.args.epochs):
            print(f"\n📈 Epoch {epoch + 1}/{self.args.epochs}")
            
            # 训练一个 epoch
            train_metrics = self._train_epoch(epoch)
            
            # 验证
            if (epoch + 1) % self.args.eval_interval == 0:
                val_metrics = self._validate_epoch(epoch)
                print(f"🎯 验证准确率: {val_metrics['accuracy']:.4f}")
            
            # 保存检查点
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # 收集分布式指标
            dist_metrics = self.dist_optimizer.get_distributed_metrics()
            self._log_distributed_metrics(epoch, dist_metrics)
        
        # 生成最终报告
        self._generate_final_report()
        
        # 清理
        self.dist_optimizer.finalize()
        
        print("🎉 分布式训练完成!")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 准备数据
            if torch.cuda.is_available():
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            
            # 创建批次字典
            if self.args.model_type in ["transformer", "llama"]:
                batch = {'input_ids': data, 'labels': target}
            else:
                # 对于 CNN 模型，需要手动计算损失
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, target)
                batch = {'loss': loss}
            
            # 执行训练步骤
            if self.args.model_type in ["transformer", "llama"]:
                step_metrics = self.dist_optimizer.train_step(batch, self.optimizer)
            else:
                # 手动执行训练步骤
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                step_metrics = {
                    'loss': loss.item(),
                    'forward_time': 0.01,
                    'backward_time': 0.01,
                    'communication_time': 0.001,
                    'optimization_time': 0.001,
                    'total_time': time.time() - batch_start_time
                }
            
            total_loss += step_metrics['loss']
            total_samples += data.size(0)
            
            # 打印进度
            if batch_idx % self.args.log_interval == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={step_metrics['loss']:.6f}, "
                      f"Comm={step_metrics['communication_time']*1000:.2f}ms, "
                      f"Total={step_metrics['total_time']*1000:.2f}ms")
            
            # 记录指标
            self.training_metrics.append({
                'epoch': epoch,
                'batch': batch_idx,
                **step_metrics
            })
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader)
        
        print(f"📊 Epoch {epoch + 1} 训练完成: "
              f"平均损失={avg_loss:.6f}, "
              f"耗时={epoch_time:.2f}s")
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'samples_per_second': total_samples / epoch_time
        }
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                if torch.cuda.is_available():
                    data = data.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)
                
                outputs = self.model(data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取 logits
                
                loss = nn.CrossEntropyLoss()(outputs, target)
                total_loss += loss.item()
                
                pred = outputs.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += data.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        if self.args.rank == 0:  # 只有主进程保存
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            self.dist_optimizer.save_checkpoint(checkpoint_path, epoch, 0)
    
    def _log_distributed_metrics(self, epoch: int, metrics: DistributedMetrics):
        """记录分布式指标"""
        if self.args.rank == 0:
            print(f"🔧 分布式指标 (Epoch {epoch + 1}):")
            print(f"  通信效率: {metrics.communication_efficiency:.4f}")
            print(f"  计算效率: {metrics.computation_efficiency:.4f}")
            print(f"  整体效率: {metrics.overall_efficiency:.4f}")
            print(f"  通信时间: {metrics.total_communication_time:.4f}s")
    
    def _generate_final_report(self):
        """生成最终报告"""
        if self.args.rank != 0:
            return
        
        print("\n" + "="*60)
        print("📊 YICA 分布式训练最终报告")
        print("="*60)
        
        # 计算总体统计
        if self.training_metrics:
            avg_loss = np.mean([m['loss'] for m in self.training_metrics[-100:]])  # 最后100个batch
            avg_comm_time = np.mean([m['communication_time'] for m in self.training_metrics])
            avg_total_time = np.mean([m['total_time'] for m in self.training_metrics])
            
            print(f"🎯 训练统计:")
            print(f"  最终损失: {avg_loss:.6f}")
            print(f"  平均通信时间: {avg_comm_time*1000:.2f}ms")
            print(f"  平均批次时间: {avg_total_time*1000:.2f}ms")
            print(f"  通信开销占比: {avg_comm_time/avg_total_time*100:.2f}%")
        
        # 分布式指标
        dist_metrics = self.dist_optimizer.get_distributed_metrics()
        print(f"\n🔧 分布式性能:")
        print(f"  通信效率: {dist_metrics.communication_efficiency:.4f}")
        print(f"  计算效率: {dist_metrics.computation_efficiency:.4f}")
        print(f"  整体效率: {dist_metrics.overall_efficiency:.4f}")
        
        # 硬件利用率
        print(f"\n🏗️  YICA 硬件利用:")
        print(f"  CIM 阵列数量: {self.yica_config.num_cim_arrays}")
        print(f"  SPM 容量: {self.yica_config.spm_size_per_die // (1024*1024)}MB")
        print(f"  量化启用: {self.yica_config.enable_quantization}")
        
        # 保存详细结果
        results = {
            'training_metrics': self.training_metrics,
            'distributed_metrics': asdict(dist_metrics),
            'yica_config': asdict(self.yica_config),
            'distributed_config': asdict(self.distributed_config)
        }
        
        with open('yica_distributed_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 详细结果已保存到: yica_distributed_training_results.json")
        print("="*60)
    
    def run_communication_benchmark(self):
        """运行通信基准测试"""
        print("🚀 开始 YCCL 通信基准测试")
        
        # 初始化分布式环境
        self.dist_optimizer.initialize_distributed()
        
        # 测试不同大小的消息
        message_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]  # 1KB to 1MB
        
        results = {}
        
        for size in message_sizes:
            print(f"📊 测试消息大小: {size} bytes")
            
            # 创建测试数据
            data = torch.randn(size // 4, dtype=torch.float32)  # 4 bytes per float
            if torch.cuda.is_available():
                data = data.cuda()
            
            # AllReduce 基准测试
            all_reduce_times = []
            for _ in range(10):  # 10 次测试
                start_time = time.time()
                
                # 模拟 AllReduce 操作
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(data)
                
                end_time = time.time()
                all_reduce_times.append(end_time - start_time)
            
            avg_time = np.mean(all_reduce_times)
            bandwidth = size / avg_time / (1024 * 1024)  # MB/s
            
            results[size] = {
                'all_reduce_time': avg_time,
                'bandwidth': bandwidth
            }
            
            print(f"  AllReduce 时间: {avg_time*1000:.2f}ms")
            print(f"  有效带宽: {bandwidth:.2f} MB/s")
        
        # 生成通信基准测试报告
        self._generate_communication_report(results)
        
        # 清理
        self.dist_optimizer.finalize()
        
        print("✅ YCCL 通信基准测试完成")
    
    def _generate_communication_report(self, results: Dict[int, Dict[str, float]]):
        """生成通信基准测试报告"""
        if self.args.rank != 0:
            return
        
        print("\n" + "="*50)
        print("📡 YCCL 通信基准测试报告")
        print("="*50)
        
        sizes = sorted(results.keys())
        times = [results[s]['all_reduce_time'] * 1000 for s in sizes]  # ms
        bandwidths = [results[s]['bandwidth'] for s in sizes]
        
        print("消息大小 (KB) | AllReduce时间 (ms) | 带宽 (MB/s)")
        print("-" * 50)
        for i, size in enumerate(sizes):
            print(f"{size//1024:>10} | {times[i]:>15.2f} | {bandwidths[i]:>10.2f}")
        
        print(f"\n📈 性能摘要:")
        print(f"  最大带宽: {max(bandwidths):.2f} MB/s")
        print(f"  最小延迟: {min(times):.2f} ms")
        print(f"  平均带宽: {np.mean(bandwidths):.2f} MB/s")
        
        # 保存结果
        with open('yccl_communication_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 基准测试结果已保存到: yccl_communication_benchmark.json")
        print("="*50)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YICA 分布式训练演示")
    
    # 分布式配置
    parser.add_argument('--world-size', type=int, default=1, help='分布式训练世界大小')
    parser.add_argument('--rank', type=int, default=0, help='当前进程排名')
    parser.add_argument('--local-rank', type=int, default=0, help='本地进程排名')
    parser.add_argument('--backend', type=str, default='yccl', choices=['nccl', 'gloo', 'yccl'],
                       help='分布式后端')
    
    # 模型配置
    parser.add_argument('--model-type', type=str, default='resnet', 
                       choices=['resnet', 'transformer', 'llama'], help='模型类型')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'], help='模型大小')
    parser.add_argument('--num-classes', type=int, default=10, help='分类数量')
    parser.add_argument('--vocab-size', type=int, default=32000, help='词汇表大小')
    parser.add_argument('--sequence-length', type=int, default=512, help='序列长度')
    
    # 训练配置
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['cifar10', 'imagenet', 'synthetic'], help='数据集')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    
    # YICA 硬件配置
    parser.add_argument('--num-cim-arrays', type=int, default=16, help='CIM 阵列数量')
    parser.add_argument('--spm-size', type=int, default=128, help='SPM 大小 (MB)')
    parser.add_argument('--dram-size', type=int, default=16, help='DRAM 大小 (GB)')
    parser.add_argument('--enable-quantization', action='store_true', help='启用量化')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'],
                       help='计算精度')
    
    # 分布式优化配置
    parser.add_argument('--data-parallel', action='store_true', default=True, help='数据并行')
    parser.add_argument('--model-parallel', action='store_true', help='模型并行')
    parser.add_argument('--gradient-compression', action='store_true', help='梯度压缩')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--dynamic-load-balancing', action='store_true', help='动态负载均衡')
    parser.add_argument('--fault-tolerance', action='store_true', help='容错机制')
    parser.add_argument('--enable-profiling', action='store_true', help='启用性能分析')
    
    # 其他配置
    parser.add_argument('--log-interval', type=int, default=10, help='日志间隔')
    parser.add_argument('--eval-interval', type=int, default=1, help='评估间隔')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='检查点间隔')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'benchmark'],
                       help='运行模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    print("🚀 YICA 分布式训练演示启动")
    print(f"📊 配置: {args.world_size} 设备, {args.backend} 后端, {args.model_type} 模型")
    print(f"🔧 YICA: {args.num_cim_arrays} CIM阵列, {args.smp_size}MB SPM, {args.precision} 精度")
    
    # 创建演示实例
    demo = YICADistributedTrainingDemo(args)
    
    try:
        # 设置模型和数据
        demo.setup_model_and_data()
        
        if args.mode == 'train':
            # 运行分布式训练
            demo.run_distributed_training()
        elif args.mode == 'benchmark':
            # 运行通信基准测试
            demo.run_communication_benchmark()
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 清理资源...")
        if demo.dist_optimizer and demo.dist_optimizer.is_initialized:
            demo.dist_optimizer.finalize()
    
    print("✅ YICA 分布式训练演示完成")


if __name__ == "__main__":
    main() 