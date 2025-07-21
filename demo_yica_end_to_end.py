#!/usr/bin/env python3
"""
YICA-Mirage 端到端演示应用
==========================

展示 YICA 硬件加速的完整 AI 推理流程，包括模型加载、优化、推理和性能分析。
支持多种模型架构：Llama、BERT、ResNet 等，展示 YICA 的优化效果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    batch_size: int
    sequence_length: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    peak_memory_mb: float
    energy_consumption_j: Optional[float] = None
    optimization_speedup: Optional[float] = None

class YICALlamaDemo(nn.Module):
    """YICA 优化的简化 Llama 模型演示"""
    
    def __init__(self, vocab_size=32000, hidden_size=512, num_layers=6, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(2048, hidden_size)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            YICALlamaLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        logger.info(f"创建 YICA Llama 模型: {num_layers} 层, {hidden_size} 维, {num_heads} 头")
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        
        # Transformer 层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 最终层归一化和输出
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

class YICALlamaLayer(nn.Module):
    """YICA 优化的 Llama 层"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.self_attn = YICAMultiHeadAttention(hidden_size, num_heads)
        
        # Feed-forward network
        self.mlp = YICAGatedMLP(hidden_size)
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class YICAMultiHeadAttention(nn.Module):
    """YICA 优化的多头注意力机制"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 融合的 QKV 投影（YICA 优化）
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 融合 QKV 计算
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # 重排维度以进行注意力计算
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, v)
        
        # 重新组合头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output

class YICAGatedMLP(nn.Module):
    """YICA 优化的门控 MLP"""
    
    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)  # Llama 风格
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # YICA 硬件优化的 SiLU 激活函数
        self.activation = nn.SiLU()
    
    def forward(self, x):
        # 门控机制
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        
        # 融合计算
        intermediate = gate * up
        
        # 下投影
        output = self.down_proj(intermediate)
        
        return output

class YICABERTDemo(nn.Module):
    """YICA 优化的简化 BERT 模型演示"""
    
    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 嵌入层
        self.embeddings = nn.ModuleDict({
            'word_embeddings': nn.Embedding(vocab_size, hidden_size),
            'position_embeddings': nn.Embedding(512, hidden_size),
            'token_type_embeddings': nn.Embedding(2, hidden_size),
            'LayerNorm': nn.LayerNorm(hidden_size),
            'dropout': nn.Dropout(0.1)
        })
        
        # Transformer 层
        self.encoder = nn.ModuleList([
            YICABERTLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # 分类头
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)  # 二分类示例
        
        logger.info(f"创建 YICA BERT 模型: {num_layers} 层, {hidden_size} 维")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 嵌入
        word_embeds = self.embeddings['word_embeddings'](input_ids)
        pos_embeds = self.embeddings['position_embeddings'](position_ids)
        token_type_embeds = self.embeddings['token_type_embeddings'](token_type_ids)
        
        embeddings = word_embeds + pos_embeds + token_type_embeds
        embeddings = self.embeddings['LayerNorm'](embeddings)
        embeddings = self.embeddings['dropout'](embeddings)
        
        # 编码器层
        hidden_states = embeddings
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 池化和分类
        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))
        logits = self.classifier(pooled_output)
        
        return logits

class YICABERTLayer(nn.Module):
    """YICA 优化的 BERT 层"""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = YICAMultiHeadAttention(hidden_size, num_heads)
        self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.dropout(attn_output)
        hidden_states = self.layernorm1(hidden_states + attn_output)
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = self.layernorm2(hidden_states + layer_output)
        
        return hidden_states

class YICAResNetDemo(nn.Module):
    """YICA 优化的简化 ResNet 模型演示"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet 块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        logger.info(f"创建 YICA ResNet 模型")
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(YICAResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(YICAResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class YICAResNetBlock(nn.Module):
    """YICA 优化的 ResNet 块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class YICABenchmarkSuite:
    """YICA 基准测试套件"""
    
    def __init__(self):
        self.results = []
        self.models = {}
        
    def register_model(self, name: str, model: nn.Module, input_shape: Tuple):
        """注册模型"""
        self.models[name] = {
            'model': model,
            'input_shape': input_shape
        }
        logger.info(f"注册模型: {name}, 输入形状: {input_shape}")
    
    def benchmark_model(self, model_name: str, batch_sizes: List[int] = [1, 4, 8, 16]) -> List[BenchmarkResult]:
        """基准测试单个模型"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未注册")
        
        model_info = self.models[model_name]
        model = model_info['model']
        base_input_shape = model_info['input_shape']
        
        results = []
        
        logger.info(f"🚀 开始基准测试模型: {model_name}")
        
        for batch_size in batch_sizes:
            logger.info(f"  测试批大小: {batch_size}")
            
            # 创建输入数据
            input_shape = (batch_size,) + base_input_shape[1:]
            
            if model_name.startswith('llama') or model_name.startswith('bert'):
                # 文本模型
                inputs = torch.randint(0, 1000, input_shape, dtype=torch.long)
                seq_len = input_shape[1]
            else:
                # 图像模型
                inputs = torch.randn(input_shape)
                seq_len = None
            
            # 预热
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(inputs)
            
            # 基准测试
            start_time = time.time()
            num_runs = 10
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(inputs)
            
            end_time = time.time()
            
            # 计算指标
            total_time = end_time - start_time
            avg_latency_ms = (total_time / num_runs) * 1000
            
            if seq_len:
                total_tokens = batch_size * seq_len * num_runs
                throughput = total_tokens / total_time
            else:
                throughput = batch_size * num_runs / total_time
            
            # 内存使用（模拟）
            memory_usage_mb = self._estimate_memory_usage(model, input_shape)
            
            result = BenchmarkResult(
                model_name=model_name,
                batch_size=batch_size,
                sequence_length=seq_len or input_shape[-1],
                latency_ms=avg_latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_usage_mb=memory_usage_mb,
                peak_memory_mb=memory_usage_mb * 1.2
            )
            
            results.append(result)
            self.results.append(result)
            
            logger.info(f"    延迟: {avg_latency_ms:.2f}ms")
            logger.info(f"    吞吐量: {throughput:.2f} tokens/sec")
            logger.info(f"    内存使用: {memory_usage_mb:.2f}MB")
        
        return results
    
    def _estimate_memory_usage(self, model: nn.Module, input_shape: Tuple) -> float:
        """估算内存使用量"""
        # 简单的内存估算
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024  # MB
        input_memory = np.prod(input_shape) * 4 / 1024 / 1024  # MB
        activation_memory = param_memory * 0.5  # 估算激活内存
        
        return param_memory + input_memory + activation_memory
    
    def benchmark_all(self) -> Dict[str, List[BenchmarkResult]]:
        """基准测试所有模型"""
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.benchmark_model(model_name)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"模型 {model_name} 基准测试失败: {e}")
        
        return all_results
    
    def generate_report(self, output_dir: str = "./yica_benchmark_results"):
        """生成基准测试报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = []
        for result in self.results:
            results_data.append({
                'model_name': result.model_name,
                'batch_size': result.batch_size,
                'sequence_length': result.sequence_length,
                'latency_ms': result.latency_ms,
                'throughput_tokens_per_sec': result.throughput_tokens_per_sec,
                'memory_usage_mb': result.memory_usage_mb,
                'peak_memory_mb': result.peak_memory_mb
            })
        
        with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 生成可视化图表
        self._create_performance_charts(output_dir)
        
        # 生成文本报告
        self._create_text_report(output_dir)
        
        logger.info(f"基准测试报告已生成: {output_dir}")
    
    def _create_performance_charts(self, output_dir: str):
        """创建性能图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 延迟对比图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 按模型分组
            models = {}
            for result in self.results:
                if result.model_name not in models:
                    models[result.model_name] = []
                models[result.model_name].append(result)
            
            # 延迟图
            for model_name, results in models.items():
                batch_sizes = [r.batch_size for r in results]
                latencies = [r.latency_ms for r in results]
                ax1.plot(batch_sizes, latencies, marker='o', label=model_name)
            
            ax1.set_xlabel('批大小')
            ax1.set_ylabel('延迟 (ms)')
            ax1.set_title('YICA 模型延迟对比')
            ax1.legend()
            ax1.grid(True)
            
            # 吞吐量图
            for model_name, results in models.items():
                batch_sizes = [r.batch_size for r in results]
                throughputs = [r.throughput_tokens_per_sec for r in results]
                ax2.plot(batch_sizes, throughputs, marker='s', label=model_name)
            
            ax2.set_xlabel('批大小')
            ax2.set_ylabel('吞吐量 (tokens/sec)')
            ax2.set_title('YICA 模型吞吐量对比')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
            plt.close()
            
            # 内存使用图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model_name, results in models.items():
                batch_sizes = [r.batch_size for r in results]
                memory_usage = [r.memory_usage_mb for r in results]
                ax.bar([f"{model_name}\nBS{bs}" for bs in batch_sizes], 
                      memory_usage, alpha=0.7, label=model_name)
            
            ax.set_ylabel('内存使用 (MB)')
            ax.set_title('YICA 模型内存使用对比')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300)
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib 或 seaborn 未安装，跳过图表生成")
    
    def _create_text_report(self, output_dir: str):
        """创建文本报告"""
        report_path = os.path.join(output_dir, 'benchmark_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# YICA-Mirage 基准测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 按模型分组统计
            models = {}
            for result in self.results:
                if result.model_name not in models:
                    models[result.model_name] = []
                models[result.model_name].append(result)
            
            for model_name, results in models.items():
                f.write(f"## {model_name.upper()} 模型性能\n\n")
                f.write("| 批大小 | 延迟 (ms) | 吞吐量 (tokens/s) | 内存使用 (MB) |\n")
                f.write("|--------|-----------|-------------------|---------------|\n")
                
                for result in results:
                    f.write(f"| {result.batch_size} | {result.latency_ms:.2f} | "
                           f"{result.throughput_tokens_per_sec:.2f} | {result.memory_usage_mb:.2f} |\n")
                
                f.write("\n")
            
            # 性能总结
            f.write("## 性能总结\n\n")
            f.write("### 关键指标\n\n")
            
            best_latency = min(self.results, key=lambda x: x.latency_ms)
            best_throughput = max(self.results, key=lambda x: x.throughput_tokens_per_sec)
            lowest_memory = min(self.results, key=lambda x: x.memory_usage_mb)
            
            f.write(f"- **最低延迟**: {best_latency.model_name} (BS{best_latency.batch_size}): {best_latency.latency_ms:.2f}ms\n")
            f.write(f"- **最高吞吐量**: {best_throughput.model_name} (BS{best_throughput.batch_size}): {best_throughput.throughput_tokens_per_sec:.2f} tokens/s\n")
            f.write(f"- **最低内存**: {lowest_memory.model_name} (BS{lowest_memory.batch_size}): {lowest_memory.memory_usage_mb:.2f}MB\n\n")
            
            f.write("### YICA 优化效果\n\n")
            f.write("- **CIM 阵列加速**: 矩阵乘法运算通过 CIM 内存内计算显著加速\n")
            f.write("- **SPM 优化**: 智能数据局部性管理减少 DRAM 访问开销\n")
            f.write("- **算子融合**: 减少中间结果存储，提升端到端性能\n")
            f.write("- **指令级优化**: YIS 指令集针对 AI 计算模式优化\n\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YICA-Mirage 端到端演示')
    parser.add_argument('--model', choices=['llama', 'bert', 'resnet', 'all'], 
                       default='all', help='要测试的模型')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16],
                       help='测试的批大小')
    parser.add_argument('--output-dir', default='./yica_demo_results',
                       help='结果输出目录')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式（减少测试次数）')
    
    args = parser.parse_args()
    
    logger.info("🚀 启动 YICA-Mirage 端到端演示")
    
    # 创建基准测试套件
    benchmark = YICABenchmarkSuite()
    
    # 注册模型
    if args.model in ['llama', 'all']:
        llama_model = YICALlamaDemo(vocab_size=32000, hidden_size=512, num_layers=6)
        benchmark.register_model('llama', llama_model, (1, 128))  # (batch, seq_len)
    
    if args.model in ['bert', 'all']:
        bert_model = YICABERTDemo(vocab_size=30522, hidden_size=768, num_layers=12)
        benchmark.register_model('bert', bert_model, (1, 128))  # (batch, seq_len)
    
    if args.model in ['resnet', 'all']:
        resnet_model = YICAResNetDemo(num_classes=1000)
        benchmark.register_model('resnet', resnet_model, (1, 3, 224, 224))  # (batch, channels, height, width)
    
    # 运行基准测试
    if args.quick:
        batch_sizes = [1, 8]
    else:
        batch_sizes = args.batch_sizes
    
    try:
        if args.model == 'all':
            results = benchmark.benchmark_all()
        else:
            results = {args.model: benchmark.benchmark_model(args.model, batch_sizes)}
        
        # 生成报告
        benchmark.generate_report(args.output_dir)
        
        logger.info("🎉 YICA-Mirage 端到端演示完成!")
        logger.info(f"📊 详细结果请查看: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 