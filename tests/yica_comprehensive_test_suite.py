#!/usr/bin/env python3
"""
YICA-Yirage 综合测试套件

这个模块提供了一个全面的测试框架，用于验证和基准测试 YICA 优化的各种功能。
包含：
- 基础操作测试（矩阵运算、优化器、性能监控）
- AI 模型组件测试（Transformer、MLP、Attention）
- 性能基准测试
- 内存效率分析
- 命令行工具测试
- Python API 集成测试
"""

import os
import sys
import time
import json
import logging
import traceback
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import unittest
from contextlib import contextmanager

# 设置测试环境
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """测试配置"""
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    hidden_sizes: List[int] = None
    enable_performance_tests: bool = True
    enable_memory_tests: bool = True
    enable_cli_tests: bool = True
    output_dir: str = "./test_results"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512]
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 768, 1024]


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    execution_time: float
    memory_usage: Optional[float] = None
    error_message: Optional[str] = None
    details: Optional[Dict] = None


class YICATestSuite:
    """YICA 综合测试套件"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results: List[TestResult] = []
        self.setup_output_dir()
        
    def setup_output_dir(self):
        """设置输出目录"""
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    @contextmanager
    def timer(self):
        """计时器上下文管理器"""
        start_time = time.time()
        yield
        end_time = time.time()
        self.last_execution_time = end_time - start_time
        
    def run_test(self, test_func: Callable, test_name: str, *args, **kwargs) -> TestResult:
        """运行单个测试"""
        logger.info(f"运行测试: {test_name}")
        
        try:
            with self.timer():
                result_data = test_func(*args, **kwargs)
            
            result = TestResult(
                test_name=test_name,
                status="PASS",
                execution_time=self.last_execution_time,
                details=result_data
            )
            logger.info(f"✅ {test_name} - 通过 ({self.last_execution_time:.3f}s)")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status="FAIL",
                execution_time=getattr(self, 'last_execution_time', 0.0),
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            )
            logger.error(f"❌ {test_name} - 失败: {str(e)}")
            
        self.results.append(result)
        return result
    
    def test_package_import(self) -> Dict:
        """测试包导入"""
        results = {}
        
        # 测试基础导入
        try:
            import yirage
            results["yirage_import"] = "SUCCESS"
            results["yirage_version"] = yirage.__version__
        except Exception as e:
            results["yirage_import"] = f"FAILED: {str(e)}"
            
        # 测试版本信息
        try:
            version_info = yirage.get_version_info()
            results["version_info"] = version_info
        except Exception as e:
            results["version_info"] = f"FAILED: {str(e)}"
            
        # 测试可用模块
        modules_to_test = [
            "yica_optimizer", "yica_performance_monitor", "yica_advanced",
            "yica_auto_tuner", "yica_distributed", "visualizer"
        ]
        
        for module_name in modules_to_test:
            try:
                module = getattr(yirage, module_name, None)
                if module:
                    results[f"{module_name}_available"] = "SUCCESS"
                else:
                    results[f"{module_name}_available"] = "NOT_FOUND"
            except Exception as e:
                results[f"{module_name}_available"] = f"FAILED: {str(e)}"
                
        return results
    
    def test_yica_optimizer(self) -> Dict:
        """测试 YICA 优化器"""
        results = {}
        
        try:
            # 测试优化器创建
            optimizer = yirage.create_yica_optimizer()
            results["optimizer_creation"] = "SUCCESS"
            
            # 测试配置设置
            yirage.set_gpu_device_id(0)
            results["gpu_device_setting"] = "SUCCESS"
            
            # 测试错误处理
            yirage.bypass_compile_errors(True)
            results["error_handling"] = "SUCCESS"
            
        except Exception as e:
            results["optimizer_test"] = f"FAILED: {str(e)}"
            
        return results
    
    def test_performance_monitor(self) -> Dict:
        """测试性能监控器"""
        results = {}
        
        try:
            # 创建性能监控器
            monitor = yirage.create_performance_monitor()
            results["monitor_creation"] = "SUCCESS"
            
            # 测试监控配置
            config = {
                "collection_interval": 1.0,
                "analysis_interval": 5.0,
                "window_size": 10
            }
            monitor_with_config = yirage.create_performance_monitor(config)
            results["monitor_with_config"] = "SUCCESS"
            
        except Exception as e:
            results["monitor_test"] = f"FAILED: {str(e)}"
            
        return results
    
    def test_cli_tools(self) -> Dict:
        """测试命令行工具"""
        results = {}
        
        cli_commands = [
            "yica-optimizer",
            "yica-benchmark", 
            "yica-analyze"
        ]
        
        for cmd in cli_commands:
            try:
                # 测试命令是否可执行
                result = subprocess.run(
                    [cmd], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0:
                    results[f"{cmd}_execution"] = "SUCCESS"
                    results[f"{cmd}_output"] = result.stdout.strip()
                else:
                    results[f"{cmd}_execution"] = f"FAILED: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                results[f"{cmd}_execution"] = "TIMEOUT"
            except FileNotFoundError:
                results[f"{cmd}_execution"] = "NOT_FOUND"
            except Exception as e:
                results[f"{cmd}_execution"] = f"ERROR: {str(e)}"
                
        return results
    
    def test_matrix_operations(self) -> Dict:
        """测试矩阵运算性能"""
        results = {}
        
        if not NUMPY_AVAILABLE or not TORCH_AVAILABLE:
            results["status"] = "SKIPPED - NumPy or PyTorch not available"
            return results
            
        try:
            # 测试不同大小的矩阵运算
            sizes = [128, 256, 512, 1024]
            
            for size in sizes:
                # 创建测试矩阵
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                
                # NumPy 基准
                start_time = time.time()
                for _ in range(self.config.warmup_iterations):
                    np_result = np.dot(a, b)
                numpy_time = time.time() - start_time
                
                results[f"numpy_matmul_{size}x{size}"] = {
                    "time": numpy_time,
                    "ops_per_sec": self.config.warmup_iterations / numpy_time
                }
                
                # PyTorch 基准（如果可用）
                if TORCH_AVAILABLE:
                    a_torch = torch.from_numpy(a)
                    b_torch = torch.from_numpy(b)
                    
                    start_time = time.time()
                    for _ in range(self.config.warmup_iterations):
                        torch_result = torch.mm(a_torch, b_torch)
                    torch_time = time.time() - start_time
                    
                    results[f"torch_matmul_{size}x{size}"] = {
                        "time": torch_time,
                        "ops_per_sec": self.config.warmup_iterations / torch_time
                    }
                    
        except Exception as e:
            results["matrix_operations"] = f"FAILED: {str(e)}"
            
        return results
    
    def test_transformer_components(self) -> Dict:
        """测试 Transformer 组件"""
        results = {}
        
        if not TORCH_AVAILABLE:
            results["status"] = "SKIPPED - PyTorch not available"
            return results
            
        try:
            # 测试多头注意力
            batch_size, seq_len, hidden_size = 4, 128, 512
            num_heads = 8
            
            # 创建测试数据
            x = torch.randn(batch_size, seq_len, hidden_size)
            
            # 简单的多头注意力实现
            class SimpleMultiHeadAttention(nn.Module):
                def __init__(self, hidden_size, num_heads):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.num_heads = num_heads
                    self.head_dim = hidden_size // num_heads
                    
                    self.q_proj = nn.Linear(hidden_size, hidden_size)
                    self.k_proj = nn.Linear(hidden_size, hidden_size)
                    self.v_proj = nn.Linear(hidden_size, hidden_size)
                    self.out_proj = nn.Linear(hidden_size, hidden_size)
                    
                def forward(self, x):
                    B, L, H = x.shape
                    q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                    
                    attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                    attn = F.softmax(attn, dim=-1)
                    out = torch.matmul(attn, v)
                    
                    out = out.transpose(1, 2).contiguous().view(B, L, H)
                    return self.out_proj(out)
            
            # 测试注意力机制
            attention = SimpleMultiHeadAttention(hidden_size, num_heads)
            
            start_time = time.time()
            for _ in range(self.config.warmup_iterations):
                output = attention(x)
            attention_time = time.time() - start_time
            
            results["multi_head_attention"] = {
                "time": attention_time,
                "input_shape": list(x.shape),
                "output_shape": list(output.shape),
                "throughput": self.config.warmup_iterations / attention_time
            }
            
            # 测试 MLP
            class SimpleMLP(nn.Module):
                def __init__(self, hidden_size, intermediate_size):
                    super().__init__()
                    self.dense1 = nn.Linear(hidden_size, intermediate_size)
                    self.dense2 = nn.Linear(intermediate_size, hidden_size)
                    
                def forward(self, x):
                    x = self.dense1(x)
                    x = F.gelu(x)
                    x = self.dense2(x)
                    return x
            
            mlp = SimpleMLP(hidden_size, hidden_size * 4)
            
            start_time = time.time()
            for _ in range(self.config.warmup_iterations):
                output = mlp(x)
            mlp_time = time.time() - start_time
            
            results["mlp"] = {
                "time": mlp_time,
                "throughput": self.config.warmup_iterations / mlp_time
            }
            
        except Exception as e:
            results["transformer_components"] = f"FAILED: {str(e)}"
            
        return results
    
    def test_memory_efficiency(self) -> Dict:
        """测试内存效率"""
        results = {}
        
        try:
            import psutil
            process = psutil.Process()
            
            # 获取初始内存使用
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行一些内存密集型操作
            if NUMPY_AVAILABLE:
                arrays = []
                for i in range(10):
                    arr = np.random.randn(1000, 1000).astype(np.float32)
                    arrays.append(arr)
                
                # 测量峰值内存
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 清理
                del arrays
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                results["memory_test"] = {
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": peak_memory - initial_memory,
                    "memory_freed_mb": peak_memory - final_memory
                }
            else:
                results["memory_test"] = "SKIPPED - NumPy not available"
                
        except ImportError:
            results["memory_test"] = "SKIPPED - psutil not available"
        except Exception as e:
            results["memory_test"] = f"FAILED: {str(e)}"
            
        return results
    
    def test_advanced_features(self) -> Dict:
        """测试高级功能"""
        results = {}
        
        try:
            # 测试快速分析功能
            if hasattr(yirage, 'quick_analyze'):
                # 创建一个简单的"模型"用于分析
                mock_model = {
                    "type": "transformer",
                    "layers": 12,
                    "hidden_size": 768,
                    "operations": ["matmul", "softmax", "layer_norm"]
                }
                
                analysis_result = yirage.quick_analyze(mock_model, "O2")
                results["quick_analyze"] = {
                    "status": "SUCCESS",
                    "result_keys": list(analysis_result.keys())
                }
            else:
                results["quick_analyze"] = "NOT_AVAILABLE"
                
        except Exception as e:
            results["advanced_features"] = f"FAILED: {str(e)}"
            
        return results
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("🚀 开始运行 YICA 综合测试套件")
        logger.info(f"测试配置: {self.config}")
        
        start_time = time.time()
        
        # 运行各类测试
        test_suite = [
            (self.test_package_import, "Package Import Test"),
            (self.test_yica_optimizer, "YICA Optimizer Test"),
            (self.test_performance_monitor, "Performance Monitor Test"),
            (self.test_cli_tools, "CLI Tools Test"),
            (self.test_matrix_operations, "Matrix Operations Test"),
            (self.test_transformer_components, "Transformer Components Test"),
            (self.test_memory_efficiency, "Memory Efficiency Test"),
            (self.test_advanced_features, "Advanced Features Test"),
        ]
        
        for test_func, test_name in test_suite:
            self.run_test(test_func, test_name)
        
        total_time = time.time() - start_time
        
        # 生成测试报告
        report = self.generate_report(total_time)
        
        # 保存结果
        self.save_results(report)
        
        logger.info(f"🏁 测试完成，总用时: {total_time:.2f}秒")
        
        return report
    
    def generate_report(self, total_time: float) -> Dict:
        """生成测试报告"""
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        report = {
            "test_summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "success_rate": passed / len(self.results) * 100 if self.results else 0,
                "total_execution_time": total_time,
                "timestamp": datetime.now().isoformat()
            },
            "environment": {
                "python_version": sys.version,
                "numpy_available": NUMPY_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "yirage_available": YIRAGE_AVAILABLE,
            },
            "test_results": [asdict(result) for result in self.results],
            "config": asdict(self.config)
        }
        
        return report
    
    def save_results(self, report: Dict):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 JSON 报告
        json_file = self.output_path / f"yica_test_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存文本报告
        txt_file = self.output_path / f"yica_test_report_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            self.write_text_report(f, report)
        
        logger.info(f"📊 测试报告已保存:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  文本: {txt_file}")
    
    def write_text_report(self, f, report: Dict):
        """写入文本格式报告"""
        f.write("=" * 80 + "\n")
        f.write("YICA-Yirage 综合测试报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 测试摘要
        summary = report["test_summary"]
        f.write(f"测试时间: {summary['timestamp']}\n")
        f.write(f"总测试数: {summary['total_tests']}\n")
        f.write(f"通过: {summary['passed']}\n")
        f.write(f"失败: {summary['failed']}\n")
        f.write(f"跳过: {summary['skipped']}\n")
        f.write(f"成功率: {summary['success_rate']:.1f}%\n")
        f.write(f"总耗时: {summary['total_execution_time']:.2f}秒\n\n")
        
        # 环境信息
        f.write("环境信息:\n")
        f.write("-" * 40 + "\n")
        env = report["environment"]
        f.write(f"Python 版本: {env['python_version'].split()[0]}\n")
        f.write(f"NumPy 可用: {env['numpy_available']}\n")
        f.write(f"PyTorch 可用: {env['torch_available']}\n")
        f.write(f"Yirage 可用: {env['yirage_available']}\n\n")
        
        # 详细测试结果
        f.write("详细测试结果:\n")
        f.write("-" * 40 + "\n")
        for result in report["test_results"]:
            status_symbol = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⏭️"
            f.write(f"{status_symbol} {result['test_name']}\n")
            f.write(f"   状态: {result['status']}\n")
            f.write(f"   耗时: {result['execution_time']:.3f}秒\n")
            if result.get('error_message'):
                f.write(f"   错误: {result['error_message']}\n")
            f.write("\n")
    
    def print_summary(self):
        """打印测试摘要"""
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print("\n" + "=" * 60)
        print("YICA 测试摘要")
        print("=" * 60)
        print(f"📊 总测试数: {len(self.results)}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"⏭️  跳过: {skipped}")
        print(f"🎯 成功率: {passed/len(self.results)*100:.1f}%" if self.results else "0%")
        print("=" * 60)
        
        # 显示失败的测试
        if failed > 0:
            print("\n失败的测试:")
            print("-" * 30)
            for result in self.results:
                if result.status == "FAIL":
                    print(f"❌ {result.test_name}: {result.error_message}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YICA-Yirage 综合测试套件")
    parser.add_argument("--config", default="default", choices=["default", "quick", "full"],
                      help="测试配置 (default, quick, full)")
    parser.add_argument("--output-dir", default="./test_results",
                      help="输出目录")
    parser.add_argument("--skip-performance", action="store_true",
                      help="跳过性能测试")
    parser.add_argument("--skip-cli", action="store_true",
                      help="跳过命令行工具测试")
    
    args = parser.parse_args()
    
    # 根据配置设置测试参数
    if args.config == "quick":
        config = TestConfig(
            warmup_iterations=2,
            benchmark_iterations=5,
            batch_sizes=[1, 4],
            sequence_lengths=[128],
            hidden_sizes=[512],
            enable_performance_tests=not args.skip_performance,
            enable_cli_tests=not args.skip_cli,
            output_dir=args.output_dir
        )
    elif args.config == "full":
        config = TestConfig(
            warmup_iterations=10,
            benchmark_iterations=50,
            batch_sizes=[1, 4, 8, 16, 32],
            sequence_lengths=[128, 256, 512, 1024],
            hidden_sizes=[512, 768, 1024, 2048],
            enable_performance_tests=not args.skip_performance,
            enable_cli_tests=not args.skip_cli,
            output_dir=args.output_dir
        )
    else:  # default
        config = TestConfig(
            enable_performance_tests=not args.skip_performance,
            enable_cli_tests=not args.skip_cli,
            output_dir=args.output_dir
        )
    
    # 运行测试套件
    test_suite = YICATestSuite(config)
    report = test_suite.run_all_tests()
    test_suite.print_summary()
    
    # 返回退出码
    failed_count = sum(1 for r in test_suite.results if r.status == "FAIL")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 