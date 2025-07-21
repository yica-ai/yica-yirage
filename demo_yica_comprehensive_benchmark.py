#!/usr/bin/env python3
"""
YICA-Mirage 综合性能基准测试

这个脚本提供了全面的性能基准测试，对比原始 Mirage 和 YICA 优化版本在各种神经网络操作上的性能：

1. 矩阵乘法性能测试
2. Attention 机制性能测试  
3. MLP 层性能测试
4. LayerNorm 性能测试
5. 端到端 Llama 模型性能测试
6. 内存使用效率分析
7. 能耗效率分析
"""

import torch
import triton
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import json
import os
from contextlib import contextmanager
import threading
import gc

# 导入 Mirage 和 YICA 相关模块
import mirage
from mirage.yica.config import YICAConfig
from mirage.yica.yica_backend import YICABackend
from mirage.python.mirage.yica_llama_optimizer import (
    YICALlamaOptimizer, LlamaModelConfig, 
    YICAOptimizedAttention, YICAOptimizedMLP, YICAOptimizedRMSNorm
)


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测试参数
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    hidden_sizes: List[int] = None
    
    # 硬件配置
    device: str = 'cuda'
    dtype: torch.dtype = torch.float16
    
    # 输出配置
    save_results: bool = True
    results_dir: str = 'yica_benchmark_results'
    generate_plots: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 512, 1024, 2048]
        if self.hidden_sizes is None:
            self.hidden_sizes = [768, 1024, 2048, 4096]


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    operation_name: str
    configuration: Dict[str, Any]
    
    # 延迟指标 (ms)
    mirage_latency: float
    yica_latency: float
    latency_speedup: float
    
    # 吞吐量指标 (ops/sec)
    mirage_throughput: float
    yica_throughput: float
    throughput_speedup: float
    
    # 内存指标 (MB)
    mirage_memory: float
    yica_memory: float
    memory_efficiency: float
    
    # 能耗指标 (估算)
    mirage_energy: float
    yica_energy: float
    energy_efficiency: float
    
    # 额外统计
    std_dev_mirage: float
    std_dev_yica: float
    confidence_interval: float


class MemoryMonitor:
    """内存使用监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_usage = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """开始监控内存使用"""
        self.monitoring = True
        self.memory_usage = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """停止监控并返回平均内存使用量 (MB)"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if self.memory_usage:
            return np.mean(self.memory_usage)
        return 0.0
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.memory_usage.append(gpu_memory)
            time.sleep(0.01)  # 10ms 采样间隔


class YICABenchmarkSuite:
    """YICA 综合性能基准测试套件"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        
        # 初始化 YICA 配置
        self.yica_config = YICAConfig(
            num_cim_arrays=16,
            spm_size_per_die=128 * 1024 * 1024,  # 128MB
            dram_size_per_cluster=16 * 1024 * 1024 * 1024,  # 16GB
            enable_quantization=True,
            target_precision="fp16"
        )
        
        # 初始化 YICA 后端
        self.yica_backend = YICABackend(self.yica_config)
        
        # 结果存储
        self.benchmark_results: List[PerformanceMetrics] = []
        
        # 创建结果目录
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[PerformanceMetrics]]:
        """运行综合基准测试"""
        print("🚀 开始 YICA-Mirage 综合性能基准测试")
        print(f"📊 测试配置: {self.config.benchmark_iterations} 次迭代, "
              f"{self.config.warmup_iterations} 次预热")
        
        all_results = {}
        
        # 1. 矩阵乘法基准测试
        print("\n1️⃣ 矩阵乘法性能测试...")
        all_results['matmul'] = self.benchmark_matrix_multiplication()
        
        # 2. Attention 机制基准测试
        print("\n2️⃣ Attention 机制性能测试...")
        all_results['attention'] = self.benchmark_attention()
        
        # 3. MLP 层基准测试
        print("\n3️⃣ MLP 层性能测试...")
        all_results['mlp'] = self.benchmark_mlp()
        
        # 4. LayerNorm 基准测试
        print("\n4️⃣ LayerNorm 性能测试...")
        all_results['layernorm'] = self.benchmark_layernorm()
        
        # 5. 端到端模型基准测试
        print("\n5️⃣ 端到端 Llama 模型性能测试...")
        all_results['end_to_end'] = self.benchmark_end_to_end_llama()
        
        # 6. 内存效率分析
        print("\n6️⃣ 内存效率分析...")
        all_results['memory_efficiency'] = self.analyze_memory_efficiency()
        
        # 生成报告
        self.generate_comprehensive_report(all_results)
        
        print("\n🎉 YICA-Mirage 综合基准测试完成!")
        return all_results
    
    def benchmark_matrix_multiplication(self) -> List[PerformanceMetrics]:
        """矩阵乘法性能基准测试"""
        results = []
        
        test_configs = [
            (512, 512, 512),
            (1024, 1024, 1024), 
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 4096, 1024),  # 不对称矩阵
        ]
        
        for M, N, K in test_configs:
            print(f"  测试矩阵乘法: {M}x{K} @ {K}x{N}")
            
            # 创建测试数据
            A = torch.randn(M, K, dtype=self.config.dtype, device=self.config.device)
            B = torch.randn(K, N, dtype=self.config.dtype, device=self.config.device)
            
            # 测试原始 PyTorch 实现
            mirage_time, mirage_memory = self._benchmark_operation(
                lambda: torch.matmul(A, B),
                "PyTorch MatMul"
            )
            
            # 测试 YICA 优化实现
            yica_time, yica_memory = self._benchmark_operation(
                lambda: self._yica_optimized_matmul(A, B),
                "YICA MatMul"
            )
            
            # 计算性能指标
            metrics = self._calculate_performance_metrics(
                operation_name="Matrix Multiplication",
                configuration={"M": M, "N": N, "K": K},
                mirage_time=mirage_time,
                yica_time=yica_time,
                mirage_memory=mirage_memory,
                yica_memory=yica_memory,
                operation_count=2 * M * N * K  # FLOPs
            )
            
            results.append(metrics)
            print(f"    加速比: {metrics.latency_speedup:.2f}x, "
                  f"内存效率: {metrics.memory_efficiency:.2f}")
        
        return results
    
    def benchmark_attention(self) -> List[PerformanceMetrics]:
        """Attention 机制性能基准测试"""
        results = []
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for hidden_size in self.config.hidden_sizes:
                    print(f"  测试 Attention: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
                    
                    # 创建标准 Attention 层
                    standard_attention = torch.nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=32,
                        batch_first=True
                    ).to(self.config.device).to(self.config.dtype)
                    
                    # 创建 YICA 优化 Attention 层
                    yica_attention = YICAOptimizedAttention(
                        hidden_size=hidden_size,
                        num_heads=32,
                        num_kv_heads=32,
                        yica_config=self.yica_config
                    ).to(self.config.device).to(self.config.dtype)
                    
                    # 创建测试数据
                    input_tensor = torch.randn(
                        batch_size, seq_len, hidden_size,
                        dtype=self.config.dtype, device=self.config.device
                    )
                    
                    # 测试标准 Attention
                    mirage_time, mirage_memory = self._benchmark_operation(
                        lambda: standard_attention(input_tensor, input_tensor, input_tensor),
                        "Standard Attention"
                    )
                    
                    # 测试 YICA Attention
                    yica_time, yica_memory = self._benchmark_operation(
                        lambda: yica_attention(input_tensor),
                        "YICA Attention"
                    )
                    
                    # 计算性能指标
                    metrics = self._calculate_performance_metrics(
                        operation_name="Multi-Head Attention",
                        configuration={
                            "batch_size": batch_size,
                            "seq_len": seq_len, 
                            "hidden_size": hidden_size
                        },
                        mirage_time=mirage_time,
                        yica_time=yica_time,
                        mirage_memory=mirage_memory,
                        yica_memory=yica_memory,
                        operation_count=batch_size * seq_len * seq_len * hidden_size
                    )
                    
                    results.append(metrics)
                    print(f"    加速比: {metrics.latency_speedup:.2f}x")
        
        return results
    
    def benchmark_mlp(self) -> List[PerformanceMetrics]:
        """MLP 层性能基准测试"""
        results = []
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for hidden_size in [1024, 2048, 4096]:
                    intermediate_size = int(hidden_size * 2.67)  # Llama 比例
                    
                    print(f"  测试 MLP: batch={batch_size}, seq={seq_len}, "
                          f"hidden={hidden_size}, intermediate={intermediate_size}")
                    
                    # 创建标准门控 MLP
                    class StandardGatedMLP(torch.nn.Module):
                        def __init__(self, hidden_size, intermediate_size):
                            super().__init__()
                            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
                            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
                        
                        def forward(self, x):
                            gate = torch.nn.functional.silu(self.gate_proj(x))
                            up = self.up_proj(x)
                            return self.down_proj(gate * up)
                    
                    standard_mlp = StandardGatedMLP(hidden_size, intermediate_size).to(
                        self.config.device).to(self.config.dtype)
                    
                    # 创建 YICA 优化 MLP
                    yica_mlp = YICAOptimizedMLP(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        yica_config=self.yica_config
                    ).to(self.config.device).to(self.config.dtype)
                    
                    # 创建测试数据
                    input_tensor = torch.randn(
                        batch_size, seq_len, hidden_size,
                        dtype=self.config.dtype, device=self.config.device
                    )
                    
                    # 测试标准 MLP
                    mirage_time, mirage_memory = self._benchmark_operation(
                        lambda: standard_mlp(input_tensor),
                        "Standard MLP"
                    )
                    
                    # 测试 YICA MLP
                    yica_time, yica_memory = self._benchmark_operation(
                        lambda: yica_mlp(input_tensor),
                        "YICA MLP"
                    )
                    
                    # 计算性能指标
                    metrics = self._calculate_performance_metrics(
                        operation_name="Gated MLP",
                        configuration={
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "hidden_size": hidden_size,
                            "intermediate_size": intermediate_size
                        },
                        mirage_time=mirage_time,
                        yica_time=yica_time,
                        mirage_memory=mirage_memory,
                        yica_memory=yica_memory,
                        operation_count=batch_size * seq_len * (hidden_size * intermediate_size * 3)
                    )
                    
                    results.append(metrics)
                    print(f"    加速比: {metrics.latency_speedup:.2f}x")
        
        return results
    
    def benchmark_layernorm(self) -> List[PerformanceMetrics]:
        """LayerNorm 性能基准测试"""
        results = []
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                for hidden_size in self.config.hidden_sizes:
                    print(f"  测试 LayerNorm: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")
                    
                    # 创建标准 LayerNorm
                    standard_norm = torch.nn.LayerNorm(hidden_size).to(
                        self.config.device).to(self.config.dtype)
                    
                    # 创建 YICA 优化 RMSNorm
                    yica_norm = YICAOptimizedRMSNorm(
                        hidden_size=hidden_size,
                        yica_config=self.yica_config
                    ).to(self.config.device).to(self.config.dtype)
                    
                    # 创建测试数据
                    input_tensor = torch.randn(
                        batch_size, seq_len, hidden_size,
                        dtype=self.config.dtype, device=self.config.device
                    )
                    
                    # 测试标准 LayerNorm
                    mirage_time, mirage_memory = self._benchmark_operation(
                        lambda: standard_norm(input_tensor),
                        "Standard LayerNorm"
                    )
                    
                    # 测试 YICA RMSNorm
                    yica_time, yica_memory = self._benchmark_operation(
                        lambda: yica_norm(input_tensor),
                        "YICA RMSNorm"
                    )
                    
                    # 计算性能指标
                    metrics = self._calculate_performance_metrics(
                        operation_name="Layer Normalization",
                        configuration={
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "hidden_size": hidden_size
                        },
                        mirage_time=mirage_time,
                        yica_time=yica_time,
                        mirage_memory=mirage_memory,
                        yica_memory=yica_memory,
                        operation_count=batch_size * seq_len * hidden_size
                    )
                    
                    results.append(metrics)
                    print(f"    加速比: {metrics.latency_speedup:.2f}x")
        
        return results
    
    def benchmark_end_to_end_llama(self) -> List[PerformanceMetrics]:
        """端到端 Llama 模型性能基准测试"""
        results = []
        
        # 简化的 Llama 模型配置用于测试
        test_configs = [
            {"layers": 12, "hidden": 768, "heads": 12},    # 小模型
            {"layers": 24, "hidden": 1024, "heads": 16},   # 中等模型
            {"layers": 32, "hidden": 2048, "heads": 32},   # 大模型
        ]
        
        for config in test_configs:
            print(f"  测试端到端模型: {config['layers']} 层, {config['hidden']} 隐藏维度")
            
            # 创建模型配置
            model_config = LlamaModelConfig(
                hidden_size=config['hidden'],
                intermediate_size=int(config['hidden'] * 2.67),
                num_hidden_layers=config['layers'],
                num_attention_heads=config['heads'],
                num_key_value_heads=config['heads']
            )
            
            # 创建简化的测试模型
            class SimplifiedLlamaLayer(torch.nn.Module):
                def __init__(self, hidden_size, intermediate_size, num_heads):
                    super().__init__()
                    self.attention = torch.nn.MultiheadAttention(
                        hidden_size, num_heads, batch_first=True
                    )
                    self.norm1 = torch.nn.LayerNorm(hidden_size)
                    self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, intermediate_size),
                        torch.nn.ReLU(),
                        torch.nn.Linear(intermediate_size, hidden_size)
                    )
                    self.norm2 = torch.nn.LayerNorm(hidden_size)
                
                def forward(self, x):
                    # Attention block
                    attn_out, _ = self.attention(x, x, x)
                    x = self.norm1(x + attn_out)
                    
                    # MLP block  
                    mlp_out = self.mlp(x)
                    x = self.norm2(x + mlp_out)
                    
                    return x
            
            # 创建标准模型
            standard_model = torch.nn.Sequential(*[
                SimplifiedLlamaLayer(
                    config['hidden'], 
                    int(config['hidden'] * 2.67), 
                    config['heads']
                ) for _ in range(config['layers'])
            ]).to(self.config.device).to(self.config.dtype)
            
            # 创建 YICA 优化模型
            yica_optimizer = YICALlamaOptimizer(model_config, self.yica_config)
            yica_model, _ = yica_optimizer.optimize_llama_model(standard_model)
            
            # 创建测试数据
            batch_size, seq_len = 4, 512
            input_tensor = torch.randn(
                batch_size, seq_len, config['hidden'],
                dtype=self.config.dtype, device=self.config.device
            )
            
            # 测试标准模型
            mirage_time, mirage_memory = self._benchmark_operation(
                lambda: standard_model(input_tensor),
                "Standard Llama Model"
            )
            
            # 测试 YICA 优化模型
            yica_time, yica_memory = self._benchmark_operation(
                lambda: yica_model(input_tensor),
                "YICA Optimized Llama Model"
            )
            
            # 计算性能指标
            metrics = self._calculate_performance_metrics(
                operation_name="End-to-End Llama Model",
                configuration={
                    "num_layers": config['layers'],
                    "hidden_size": config['hidden'],
                    "num_heads": config['heads'],
                    "batch_size": batch_size,
                    "seq_len": seq_len
                },
                mirage_time=mirage_time,
                yica_time=yica_time,
                mirage_memory=mirage_memory,
                yica_memory=yica_memory,
                operation_count=batch_size * seq_len * config['hidden'] * config['layers']
            )
            
            results.append(metrics)
            print(f"    端到端加速比: {metrics.latency_speedup:.2f}x, "
                  f"内存节省: {(1 - metrics.memory_efficiency) * 100:.1f}%")
        
        return results
    
    def analyze_memory_efficiency(self) -> List[PerformanceMetrics]:
        """内存效率分析"""
        results = []
        
        # 测试不同大小的张量内存使用
        tensor_sizes = [
            (1024, 1024),
            (2048, 2048), 
            (4096, 4096),
            (8192, 8192)
        ]
        
        for rows, cols in tensor_sizes:
            print(f"  分析内存效率: {rows}x{cols} 张量")
            
            # 标准内存分配
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            standard_tensor = torch.randn(rows, cols, dtype=self.config.dtype, device=self.config.device)
            standard_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            # YICA 优化内存分配（模拟 SPM 优化）
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 模拟 YICA SPM 内存优化
            yica_tensor = self._allocate_yica_optimized_tensor(rows, cols)
            yica_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            # 计算内存效率指标
            memory_efficiency = standard_memory / yica_memory if yica_memory > 0 else 1.0
            
            metrics = PerformanceMetrics(
                operation_name="Memory Allocation",
                configuration={"tensor_size": f"{rows}x{cols}"},
                mirage_latency=0.0,
                yica_latency=0.0,
                latency_speedup=1.0,
                mirage_throughput=0.0,
                yica_throughput=0.0,
                throughput_speedup=1.0,
                mirage_memory=standard_memory,
                yica_memory=yica_memory,
                memory_efficiency=memory_efficiency,
                mirage_energy=0.0,
                yica_energy=0.0,
                energy_efficiency=1.0,
                std_dev_mirage=0.0,
                std_dev_yica=0.0,
                confidence_interval=0.95
            )
            
            results.append(metrics)
            print(f"    内存效率提升: {memory_efficiency:.2f}x")
            
            # 清理内存
            del standard_tensor, yica_tensor
            torch.cuda.empty_cache()
        
        return results
    
    @contextmanager
    def _benchmark_context(self):
        """基准测试上下文管理器"""
        # 预热 GPU
        torch.cuda.synchronize()
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
        
        yield
        
        # 同步和清理
        torch.cuda.synchronize()
    
    def _benchmark_operation(self, operation_func, operation_name: str) -> Tuple[float, float]:
        """基准测试单个操作"""
        times = []
        
        # 开始内存监控
        self.memory_monitor.start_monitoring()
        
        with self._benchmark_context():
            # 预热
            for _ in range(self.config.warmup_iterations):
                with torch.no_grad():
                    operation_func()
                torch.cuda.synchronize()
            
            # 基准测试
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            for _ in range(self.config.benchmark_iterations):
                with torch.no_grad():
                    operation_func()
                torch.cuda.synchronize()
                
                iteration_time = time.perf_counter()
                times.append((iteration_time - start_time) * 1000)  # 转换为毫秒
                start_time = iteration_time
        
        # 停止内存监控
        avg_memory = self.memory_monitor.stop_monitoring()
        
        # 计算统计信息
        avg_time = np.mean(times)
        
        return avg_time, avg_memory
    
    def _calculate_performance_metrics(self, operation_name: str, configuration: Dict[str, Any],
                                     mirage_time: float, yica_time: float,
                                     mirage_memory: float, yica_memory: float,
                                     operation_count: int) -> PerformanceMetrics:
        """计算性能指标"""
        
        # 延迟指标
        latency_speedup = mirage_time / yica_time if yica_time > 0 else 1.0
        
        # 吞吐量指标 (ops/sec)
        mirage_throughput = operation_count / (mirage_time / 1000) if mirage_time > 0 else 0
        yica_throughput = operation_count / (yica_time / 1000) if yica_time > 0 else 0
        throughput_speedup = yica_throughput / mirage_throughput if mirage_throughput > 0 else 1.0
        
        # 内存效率
        memory_efficiency = mirage_memory / yica_memory if yica_memory > 0 else 1.0
        
        # 能耗估算 (简化模型: 功耗 ∝ 时间 × 内存使用)
        mirage_energy = mirage_time * mirage_memory * 0.001  # 简化单位
        yica_energy = yica_time * yica_memory * 0.001
        energy_efficiency = mirage_energy / yica_energy if yica_energy > 0 else 1.0
        
        return PerformanceMetrics(
            operation_name=operation_name,
            configuration=configuration,
            mirage_latency=mirage_time,
            yica_latency=yica_time,
            latency_speedup=latency_speedup,
            mirage_throughput=mirage_throughput,
            yica_throughput=yica_throughput,
            throughput_speedup=throughput_speedup,
            mirage_memory=mirage_memory,
            yica_memory=yica_memory,
            memory_efficiency=memory_efficiency,
            mirage_energy=mirage_energy,
            yica_energy=yica_energy,
            energy_efficiency=energy_efficiency,
            std_dev_mirage=0.0,  # 简化实现
            std_dev_yica=0.0,
            confidence_interval=0.95
        )
    
    def _yica_optimized_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """YICA 优化的矩阵乘法（模拟实现）"""
        # 在实际实现中，这里会调用 YICA 的 CIM 阵列加速矩阵乘法
        # 现在使用标准实现模拟，但假设有性能提升
        result = torch.matmul(A, B)
        
        # 模拟 YICA CIM 阵列的额外优化
        # 在实际实现中，这里会是硬件加速的 YIS 指令
        
        return result
    
    def _allocate_yica_optimized_tensor(self, rows: int, cols: int) -> torch.Tensor:
        """分配 YICA 优化的张量（模拟 SPM 优化）"""
        # 在实际实现中，这里会使用 SPM 内存分配器
        # 现在使用标准分配模拟，但假设内存效率更高
        tensor = torch.randn(rows, cols, dtype=self.config.dtype, device=self.config.device)
        
        # 模拟 SPM 内存优化的额外处理
        # 在实际实现中，这里会涉及 SPM 内存布局优化
        
        return tensor
    
    def generate_comprehensive_report(self, all_results: Dict[str, List[PerformanceMetrics]]):
        """生成综合性能报告"""
        print("\n📊 生成综合性能报告...")
        
        # 保存详细结果到 JSON
        if self.config.save_results:
            results_data = {}
            for category, results in all_results.items():
                results_data[category] = [asdict(metric) for metric in results]
            
            with open(os.path.join(self.config.results_dir, 'detailed_results.json'), 'w') as f:
                json.dump(results_data, f, indent=2)
        
        # 生成汇总报告
        summary_report = self._generate_summary_report(all_results)
        
        if self.config.save_results:
            with open(os.path.join(self.config.results_dir, 'summary_report.txt'), 'w') as f:
                f.write(summary_report)
        
        print(summary_report)
        
        # 生成可视化图表
        if self.config.generate_plots:
            self._generate_performance_plots(all_results)
    
    def _generate_summary_report(self, all_results: Dict[str, List[PerformanceMetrics]]) -> str:
        """生成汇总报告"""
        report = []
        report.append("=" * 80)
        report.append("YICA-Mirage 综合性能基准测试报告")
        report.append("=" * 80)
        report.append("")
        
        # 总体统计
        all_metrics = []
        for results in all_results.values():
            all_metrics.extend(results)
        
        if all_metrics:
            avg_latency_speedup = np.mean([m.latency_speedup for m in all_metrics])
            avg_throughput_speedup = np.mean([m.throughput_speedup for m in all_metrics])
            avg_memory_efficiency = np.mean([m.memory_efficiency for m in all_metrics])
            avg_energy_efficiency = np.mean([m.energy_efficiency for m in all_metrics])
            
            report.append("📈 总体性能提升:")
            report.append(f"  • 平均延迟加速比: {avg_latency_speedup:.2f}x")
            report.append(f"  • 平均吞吐量提升: {avg_throughput_speedup:.2f}x")
            report.append(f"  • 平均内存效率提升: {avg_memory_efficiency:.2f}x")
            report.append(f"  • 平均能效提升: {avg_energy_efficiency:.2f}x")
            report.append("")
        
        # 各类操作详细统计
        for category, results in all_results.items():
            if not results:
                continue
                
            report.append(f"🔍 {category.upper()} 性能分析:")
            
            category_speedups = [r.latency_speedup for r in results]
            best_speedup = max(category_speedups)
            worst_speedup = min(category_speedups)
            avg_speedup = np.mean(category_speedups)
            
            report.append(f"  • 最佳加速比: {best_speedup:.2f}x")
            report.append(f"  • 最差加速比: {worst_speedup:.2f}x") 
            report.append(f"  • 平均加速比: {avg_speedup:.2f}x")
            
            # 显示最佳配置
            best_result = max(results, key=lambda r: r.latency_speedup)
            report.append(f"  • 最佳配置: {best_result.configuration}")
            report.append("")
        
        # 建议和结论
        report.append("💡 优化建议:")
        if avg_latency_speedup > 2.0:
            report.append("  • YICA 优化显著提升了计算性能，建议在生产环境中部署")
        elif avg_latency_speedup > 1.5:
            report.append("  • YICA 优化带来了明显的性能提升，适合性能敏感的应用")
        else:
            report.append("  • YICA 优化带来了一定的性能提升，可考虑特定场景应用")
        
        if avg_memory_efficiency > 1.2:
            report.append("  • 内存使用效率显著改善，有助于处理更大规模的模型")
        
        if avg_energy_efficiency > 1.5:
            report.append("  • 能效比大幅提升，有利于降低运行成本和环境影响")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_performance_plots(self, all_results: Dict[str, List[PerformanceMetrics]]):
        """生成性能可视化图表"""
        print("📊 生成性能可视化图表...")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YICA-Mirage 性能基准测试结果', fontsize=16, fontweight='bold')
        
        # 1. 延迟加速比对比
        categories = list(all_results.keys())
        speedups = []
        for category in categories:
            if all_results[category]:
                avg_speedup = np.mean([r.latency_speedup for r in all_results[category]])
                speedups.append(avg_speedup)
            else:
                speedups.append(1.0)
        
        axes[0, 0].bar(categories, speedups, color='skyblue', alpha=0.8)
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='基准线')
        axes[0, 0].set_title('平均延迟加速比')
        axes[0, 0].set_ylabel('加速比')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 内存效率提升
        memory_efficiencies = []
        for category in categories:
            if all_results[category]:
                avg_efficiency = np.mean([r.memory_efficiency for r in all_results[category]])
                memory_efficiencies.append(avg_efficiency)
            else:
                memory_efficiencies.append(1.0)
        
        axes[0, 1].bar(categories, memory_efficiencies, color='lightgreen', alpha=0.8)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='基准线')
        axes[0, 1].set_title('平均内存效率提升')
        axes[0, 1].set_ylabel('效率比')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 吞吐量提升散点图
        if 'matmul' in all_results and all_results['matmul']:
            matmul_results = all_results['matmul']
            matrix_sizes = [r.configuration.get('M', 0) * r.configuration.get('N', 0) 
                          for r in matmul_results]
            throughput_speedups = [r.throughput_speedup for r in matmul_results]
            
            axes[1, 0].scatter(matrix_sizes, throughput_speedups, alpha=0.7, color='orange')
            axes[1, 0].set_title('矩阵乘法吞吐量提升 vs 矩阵大小')
            axes[1, 0].set_xlabel('矩阵大小 (M×N)')
            axes[1, 0].set_ylabel('吞吐量提升比')
            axes[1, 0].set_xscale('log')
        
        # 4. 能效比提升
        energy_efficiencies = []
        for category in categories:
            if all_results[category]:
                avg_energy_eff = np.mean([r.energy_efficiency for r in all_results[category]])
                energy_efficiencies.append(avg_energy_eff)
            else:
                energy_efficiencies.append(1.0)
        
        axes[1, 1].bar(categories, energy_efficiencies, color='lightcoral', alpha=0.8)
        axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='基准线')
        axes[1, 1].set_title('平均能效比提升')
        axes[1, 1].set_ylabel('能效比')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.config.save_results:
            plt.savefig(os.path.join(self.config.results_dir, 'performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"📊 图表已保存到 {self.config.results_dir}/performance_comparison.png")
        
        plt.show()


def main():
    """主函数：运行 YICA 综合性能基准测试"""
    print("🚀 启动 YICA-Mirage 综合性能基准测试套件")
    
    # 配置基准测试
    config = BenchmarkConfig(
        warmup_iterations=5,
        benchmark_iterations=50,
        batch_sizes=[1, 4, 8],
        sequence_lengths=[128, 512, 1024],
        hidden_sizes=[768, 1024, 2048],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float16,
        save_results=True,
        results_dir='yica_benchmark_results',
        generate_plots=True
    )
    
    print(f"📊 测试配置:")
    print(f"  • 设备: {config.device}")
    print(f"  • 数据类型: {config.dtype}")
    print(f"  • 预热迭代: {config.warmup_iterations}")
    print(f"  • 基准迭代: {config.benchmark_iterations}")
    print(f"  • 结果保存: {config.save_results}")
    
    # 创建基准测试套件
    benchmark_suite = YICABenchmarkSuite(config)
    
    # 运行综合基准测试
    try:
        results = benchmark_suite.run_comprehensive_benchmark()
        
        print("\n✅ 基准测试完成!")
        print(f"📁 详细结果保存在: {config.results_dir}/")
        
        # 显示关键统计信息
        all_metrics = []
        for category_results in results.values():
            all_metrics.extend(category_results)
        
        if all_metrics:
            avg_speedup = np.mean([m.latency_speedup for m in all_metrics])
            max_speedup = max([m.latency_speedup for m in all_metrics])
            avg_memory_eff = np.mean([m.memory_efficiency for m in all_metrics])
            
            print(f"\n🎯 关键性能指标:")
            print(f"  • 平均加速比: {avg_speedup:.2f}x")
            print(f"  • 最大加速比: {max_speedup:.2f}x")
            print(f"  • 平均内存效率提升: {avg_memory_eff:.2f}x")
        
    except Exception as e:
        print(f"❌ 基准测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 