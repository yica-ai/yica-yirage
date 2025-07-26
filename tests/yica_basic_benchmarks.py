#!/usr/bin/env python3
"""
YICA-Yirage 基础基准测试

仿照现有基准测试项目，为 YICA 功能创建对应的基准测试。
包含：
- 矩阵运算基准测试
- 注意力机制基准测试  
- MLP 基准测试
- LoRA 基准测试
- 性能对比分析
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加 yirage 包路径
sys.path.insert(0, str(Path(__file__).parent.parent / "yirage" / "python"))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yirage
    YIRAGE_AVAILABLE = True
except ImportError:
    YIRAGE_AVAILABLE = False


class YICABenchmarkRunner:
    """YICA 基准测试运行器"""
    
    def __init__(self, warmup_iters: int = 16, profile_iters: int = 1000):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.results = {}
        
    def time_operation(self, operation_func, *args, **kwargs) -> float:
        """测量操作执行时间"""
        if not TORCH_AVAILABLE:
            # 使用 time.time() 进行基准测试
            for _ in range(self.warmup_iters):
                operation_func(*args, **kwargs)
            
            start_time = time.time()
            for _ in range(self.profile_iters):
                operation_func(*args, **kwargs)
            end_time = time.time()
            
            return (end_time - start_time) * 1000 / self.profile_iters  # ms
        else:
            # 使用 CUDA 事件进行更精确的测试
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
            # 预热
            for _ in range(self.warmup_iters):
                operation_func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                starter.record()
                for _ in range(self.profile_iters):
                    operation_func(*args, **kwargs)
                ender.record()
                
                torch.cuda.synchronize()
                return starter.elapsed_time(ender) / self.profile_iters  # ms
            else:
                start_time = time.time()
                for _ in range(self.profile_iters):
                    operation_func(*args, **kwargs)
                end_time = time.time()
                
                return (end_time - start_time) * 1000 / self.profile_iters  # ms

    def benchmark_gated_mlp(self, batch_size: int = 8, hidden_size: int = 4096) -> Dict:
        """基准测试门控 MLP"""
        print(f"🧠 基准测试门控 MLP (batch_size={batch_size}, hidden_size={hidden_size})")
        
        results = {
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "test_type": "gated_mlp"
        }
        
        if not TORCH_AVAILABLE:
            # NumPy 实现
            if NUMPY_AVAILABLE:
                def numpy_gated_mlp():
                    x = np.random.randn(batch_size, hidden_size).astype(np.float32)
                    w1 = np.random.randn(hidden_size, hidden_size).astype(np.float32)
                    w2 = np.random.randn(hidden_size, hidden_size).astype(np.float32)
                    
                    o1 = np.dot(x, w1)
                    o2 = np.dot(x, w2)
                    # 简化的 SiLU 激活
                    o1_activated = o1 / (1 + np.exp(-o1))
                    result = o1_activated * o2
                    return result
                
                numpy_time = self.time_operation(numpy_gated_mlp)
                results["numpy_time_ms"] = numpy_time
                results["numpy_throughput"] = 1000 / numpy_time
                
                print(f"  NumPy 实现: {numpy_time:.3f} ms")
            else:
                results["numpy_time_ms"] = "N/A - NumPy not available"
        else:
            # PyTorch 实现
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
            def torch_gated_mlp():
                x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
                w1 = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float16)
                w2 = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float16)
                
                o1 = torch.mm(x, w1)
                o2 = torch.mm(x, w2)
                # SiLU 激活 (兼容旧版 PyTorch)
                if hasattr(torch, 'silu'):
                    o1_activated = torch.silu(o1)
                else:
                    o1_activated = o1 * torch.sigmoid(o1)  # SiLU = x * sigmoid(x)
                result = o1_activated * o2
                return result
            
            torch_time = self.time_operation(torch_gated_mlp)
            results["torch_time_ms"] = torch_time
            results["torch_throughput"] = 1000 / torch_time
            results["device"] = device
            
            print(f"  PyTorch 实现 ({device}): {torch_time:.3f} ms")
            
            # 使用 nn.Linear 的更高级实现
            class GatedMLP(nn.Module):
                def __init__(self, hidden_size):
                    super().__init__()
                    self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                    self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                    
                def forward(self, x):
                    gate = self.gate_proj(x)
                    up = self.up_proj(x)
                    # SiLU 激活 (兼容旧版 PyTorch)
                    if hasattr(torch, 'silu'):
                        return torch.silu(gate) * up
                    else:
                        return (gate * torch.sigmoid(gate)) * up
            
            model = GatedMLP(hidden_size).to(device).half()
            x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
            
            def torch_nn_gated_mlp():
                return model(x)
            
            torch_nn_time = self.time_operation(torch_nn_gated_mlp)
            results["torch_nn_time_ms"] = torch_nn_time
            results["torch_nn_throughput"] = 1000 / torch_nn_time
            
            print(f"  PyTorch nn.Module 实现 ({device}): {torch_nn_time:.3f} ms")
        
        return results

    def benchmark_group_query_attention(self, batch_size: int = 2, seq_len: int = 256, 
                                       hidden_size: int = 64, kv_len: int = 4096) -> Dict:
        """基准测试组查询注意力"""
        print(f"🎯 基准测试组查询注意力 (batch_size={batch_size}, seq_len={seq_len})")
        
        results = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "kv_len": kv_len,
            "test_type": "group_query_attention"
        }
        
        if not TORCH_AVAILABLE:
            results["status"] = "SKIPPED - PyTorch required for attention"
            return results
            
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        def torch_group_query_attention():
            # 模拟组查询注意力
            Q = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
            K = torch.randn(batch_size, hidden_size, kv_len, device=device, dtype=torch.float16)
            V = torch.randn(batch_size, kv_len, hidden_size, device=device, dtype=torch.float16)
            
            # 注意力计算
            A = torch.matmul(Q, K)  # [batch, seq_len, kv_len]
            A_exp = torch.exp(A)
            A_sum = torch.sum(A_exp, dim=-1, keepdim=True)  # [batch, seq_len, 1]
            A_softmax = A_exp / A_sum
            O = torch.matmul(A_softmax, V)  # [batch, seq_len, hidden_size]
            
            return O
        
        attention_time = self.time_operation(torch_group_query_attention)
        results["torch_time_ms"] = attention_time
        results["torch_throughput"] = 1000 / attention_time
        results["device"] = device
        
        print(f"  PyTorch 实现 ({device}): {attention_time:.3f} ms")
        
        return results

    def benchmark_lora(self, input_size: int = 16, hidden_size: int = 256, 
                      output_size: int = 4096, rank: int = 16) -> Dict:
        """基准测试 LoRA（低秩适应）"""
        print(f"🔄 基准测试 LoRA (rank={rank}, sizes=[{input_size}, {hidden_size}, {output_size}])")
        
        results = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "rank": rank,
            "test_type": "lora"
        }
        
        if not TORCH_AVAILABLE:
            results["status"] = "SKIPPED - PyTorch required for LoRA"
            return results
            
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        def torch_lora():
            # 输入数据
            X = torch.randn(input_size, hidden_size, device=device, dtype=torch.float16)
            
            # 原始权重矩阵
            W = torch.randn(hidden_size, output_size, device=device, dtype=torch.float16)
            
            # LoRA 权重矩阵
            A = torch.randn(hidden_size, rank, device=device, dtype=torch.float16)
            B = torch.randn(rank, output_size, device=device, dtype=torch.float16)
            
            # LoRA 计算：X @ W + X @ A @ B
            base_output = torch.matmul(X, W)
            lora_delta = torch.matmul(torch.matmul(X, A), B)
            final_output = base_output + lora_delta
            
            return final_output
        
        lora_time = self.time_operation(torch_lora)
        results["torch_time_ms"] = lora_time
        results["torch_throughput"] = 1000 / lora_time
        results["device"] = device
        
        print(f"  PyTorch LoRA 实现 ({device}): {lora_time:.3f} ms")
        
        # 对比标准全连接
        def torch_standard():
            X = torch.randn(input_size, hidden_size, device=device, dtype=torch.float16)
            W = torch.randn(hidden_size, output_size, device=device, dtype=torch.float16)
            return torch.matmul(X, W)
        
        standard_time = self.time_operation(torch_standard)
        results["standard_time_ms"] = standard_time
        results["standard_throughput"] = 1000 / standard_time
        results["lora_overhead"] = (lora_time / standard_time - 1) * 100  # 百分比开销
        
        print(f"  标准全连接 ({device}): {standard_time:.3f} ms")
        print(f"  LoRA 开销: {results['lora_overhead']:.1f}%")
        
        return results

    def benchmark_matrix_operations(self, sizes: List[int] = None) -> Dict:
        """基准测试基础矩阵运算"""
        if sizes is None:
            sizes = [128, 256, 512, 1024]
            
        print("📊 基准测试矩阵运算")
        
        results = {
            "test_type": "matrix_operations",
            "sizes_tested": sizes,
            "results": {}
        }
        
        for size in sizes:
            print(f"  测试 {size}x{size} 矩阵")
            size_results = {}
            
            # NumPy 基准
            if NUMPY_AVAILABLE:
                def numpy_matmul():
                    a = np.random.randn(size, size).astype(np.float32)
                    b = np.random.randn(size, size).astype(np.float32)
                    return np.dot(a, b)
                
                numpy_time = self.time_operation(numpy_matmul)
                size_results["numpy_time_ms"] = numpy_time
                size_results["numpy_gflops"] = (2 * size**3) / (numpy_time * 1e6)  # GFLOPS
                
                print(f"    NumPy: {numpy_time:.3f} ms ({size_results['numpy_gflops']:.2f} GFLOPS)")
            
            # PyTorch 基准
            if TORCH_AVAILABLE:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                def torch_matmul():
                    a = torch.randn(size, size, device=device, dtype=torch.float32)
                    b = torch.randn(size, size, device=device, dtype=torch.float32)
                    return torch.mm(a, b)
                
                torch_time = self.time_operation(torch_matmul)
                size_results["torch_time_ms"] = torch_time
                size_results["torch_gflops"] = (2 * size**3) / (torch_time * 1e6)  # GFLOPS
                size_results["device"] = device
                
                print(f"    PyTorch ({device}): {torch_time:.3f} ms ({size_results['torch_gflops']:.2f} GFLOPS)")
            
            results["results"][f"{size}x{size}"] = size_results
        
        return results

    def benchmark_yica_api(self) -> Dict:
        """基准测试 YICA API 性能"""
        print("🔧 基准测试 YICA API")
        
        results = {
            "test_type": "yica_api",
            "yirage_available": YIRAGE_AVAILABLE
        }
        
        if not YIRAGE_AVAILABLE:
            results["status"] = "SKIPPED - YICA package not available"
            return results
        
        # 测试 API 调用性能
        def test_optimizer_creation():
            return yirage.create_yica_optimizer()
        
        def test_performance_monitor_creation():
            return yirage.create_performance_monitor()
        
        def test_version_info():
            return yirage.get_version_info()
        
        # 测试各种 API 调用的延迟
        optimizer_time = self.time_operation(test_optimizer_creation)
        monitor_time = self.time_operation(test_performance_monitor_creation)
        version_time = self.time_operation(test_version_info)
        
        results["optimizer_creation_ms"] = optimizer_time
        results["monitor_creation_ms"] = monitor_time
        results["version_info_ms"] = version_time
        
        print(f"  优化器创建: {optimizer_time:.3f} ms")
        print(f"  性能监控器创建: {monitor_time:.3f} ms")  
        print(f"  版本信息获取: {version_time:.3f} ms")
        
        return results

    def run_all_benchmarks(self) -> Dict:
        """运行所有基准测试"""
        print("🚀 开始运行 YICA 基准测试套件")
        print(f"配置: warmup={self.warmup_iters}, profile={self.profile_iters}")
        
        all_results = {
            "benchmark_config": {
                "warmup_iterations": self.warmup_iters,
                "profile_iterations": self.profile_iters,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "environment": {
                "numpy_available": NUMPY_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "yirage_available": YIRAGE_AVAILABLE,
                "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False
            },
            "results": {}
        }
        
        # 运行各项基准测试
        benchmarks = [
            ("matrix_operations", lambda: self.benchmark_matrix_operations()),
            ("gated_mlp", lambda: self.benchmark_gated_mlp()),
            ("group_query_attention", lambda: self.benchmark_group_query_attention()),
            ("lora", lambda: self.benchmark_lora()),
            ("yica_api", lambda: self.benchmark_yica_api()),
        ]
        
        for name, benchmark_func in benchmarks:
            try:
                print(f"\n--- {name.upper()} ---")
                result = benchmark_func()
                all_results["results"][name] = result
                print(f"✅ {name} 完成")
                
            except Exception as e:
                print(f"❌ {name} 失败: {str(e)}")
                all_results["results"][name] = {
                    "status": "FAILED",
                    "error": str(e)
                }
        
        return all_results
    
    def save_results(self, results: Dict, output_file: str = None):
        """保存基准测试结果"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"yica_benchmark_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YICA-Yirage 基础基准测试")
    parser.add_argument('--warmup', type=int, default=16, help='预热迭代次数')
    parser.add_argument('--profile', type=int, default=1000, help='性能测试迭代次数')
    parser.add_argument('--output', type=str, help='输出文件名')
    parser.add_argument('--test', type=str, choices=['all', 'matrix', 'gated_mlp', 'attention', 'lora', 'api'],
                       default='all', help='要运行的测试类型')
    
    args = parser.parse_args()
    
    runner = YICABenchmarkRunner(warmup_iters=args.warmup, profile_iters=args.profile)
    
    if args.test == 'all':
        results = runner.run_all_benchmarks()
    elif args.test == 'matrix':
        results = {"results": {"matrix_operations": runner.benchmark_matrix_operations()}}
    elif args.test == 'gated_mlp':
        results = {"results": {"gated_mlp": runner.benchmark_gated_mlp()}}
    elif args.test == 'attention':
        results = {"results": {"group_query_attention": runner.benchmark_group_query_attention()}}
    elif args.test == 'lora':
        results = {"results": {"lora": runner.benchmark_lora()}}
    elif args.test == 'api':
        results = {"results": {"yica_api": runner.benchmark_yica_api()}}
    
    # 保存结果
    runner.save_results(results, args.output)
    
    print("\n🏁 基准测试完成!")


if __name__ == "__main__":
    main() 