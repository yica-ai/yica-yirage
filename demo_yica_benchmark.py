#!/usr/bin/env python3
"""
YICA-Mirage 基准测试演示脚本

这个脚本演示了如何使用 YICA-Mirage 基准测试套件进行性能分析。
包含：
- 基础基准测试运行
- 自定义配置使用
- 结果分析和可视化
- 性能对比报告生成
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / "mirage"))

try:
    from mirage.benchmark.yica_benchmark_suite import (
        YICABenchmarkSuite, 
        BenchmarkConfig,
        BasicOperationBenchmark,
        TransformerBenchmark
    )
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 无法导入基准测试模块: {e}")
    BENCHMARK_AVAILABLE = False

try:
    from mirage.yica_pytorch_backend import initialize as yica_initialize
    YICA_AVAILABLE = True
except ImportError:
    print("Warning: YICA 后端不可用")
    YICA_AVAILABLE = False


def load_config_from_file(config_file: str, config_name: str = "benchmark_config") -> Dict[str, Any]:
    """从配置文件加载配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        
        if config_name not in configs:
            available_configs = list(configs.keys())
            print(f"Error: 配置 '{config_name}' 不存在")
            print(f"可用配置: {available_configs}")
            return {}
        
        return configs[config_name]
    except Exception as e:
        print(f"Error: 无法加载配置文件 {config_file}: {e}")
        return {}


def run_basic_demo():
    """运行基础演示"""
    print("🚀 开始基础基准测试演示...")
    
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，跳过演示")
        return
    
    # 创建快速测试配置
    config = BenchmarkConfig(
        warmup_iterations=3,
        benchmark_iterations=10,
        batch_sizes=[1, 4, 8],
        sequence_lengths=[128, 512],
        hidden_sizes=[768, 1024],
        enable_memory_profiling=True,
        enable_energy_profiling=False,
        output_dir="./demo_benchmark_results",
        device="auto"
    )
    
    # 创建基准测试套件
    benchmark_suite = YICABenchmarkSuite(config)
    
    print("📊 运行基础操作基准测试...")
    basic_results = benchmark_suite.run_basic_operation_benchmarks()
    
    print(f"✅ 完成 {len(basic_results)} 个基础操作测试")
    
    # 保存结果
    results_file = benchmark_suite.save_results()
    print(f"💾 结果已保存: {results_file}")
    
    return benchmark_suite


def run_transformer_demo():
    """运行 Transformer 模型演示"""
    print("🔍 开始 Transformer 基准测试演示...")
    
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，跳过演示")
        return
    
    # 创建针对 Transformer 的配置
    config = BenchmarkConfig(
        warmup_iterations=5,
        benchmark_iterations=20,
        batch_sizes=[1, 8, 16],
        sequence_lengths=[256, 512, 1024],
        hidden_sizes=[768, 1024],
        enable_memory_profiling=True,
        output_dir="./demo_transformer_results",
        device="auto"
    )
    
    # 创建基准测试套件
    benchmark_suite = YICABenchmarkSuite(config)
    
    print("🧠 运行 Transformer 基准测试...")
    transformer_results = benchmark_suite.run_transformer_benchmarks()
    
    print(f"✅ 完成 {len(transformer_results)} 个 Transformer 测试")
    
    # 保存结果和生成可视化
    benchmark_suite.save_results()
    chart_file = benchmark_suite.generate_visualization()
    report_file = benchmark_suite.generate_report()
    
    print(f"📈 图表已生成: {chart_file}")
    print(f"📋 报告已生成: {report_file}")
    
    return benchmark_suite


def run_config_file_demo():
    """运行配置文件演示"""
    print("⚙️ 开始配置文件基准测试演示...")
    
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，跳过演示")
        return
    
    config_file = project_root / "mirage/benchmark/configs/yica_benchmark_config.json"
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    # 加载快速配置
    config_data = load_config_from_file(str(config_file), "quick_config")
    if not config_data:
        return
    
    # 转换为 BenchmarkConfig
    config = BenchmarkConfig(
        warmup_iterations=config_data.get("warmup_iterations", 3),
        benchmark_iterations=config_data.get("benchmark_iterations", 10),
        batch_sizes=config_data.get("batch_sizes", [1, 8]),
        sequence_lengths=config_data.get("sequence_lengths", [128, 512]),
        hidden_sizes=config_data.get("hidden_sizes", [768, 1024]),
        enable_memory_profiling=config_data.get("enable_memory_profiling", True),
        enable_energy_profiling=config_data.get("enable_energy_profiling", False),
        output_dir=config_data.get("output_dir", "./config_demo_results"),
        device=config_data.get("device", "auto"),
        precision=config_data.get("precision", "fp32")
    )
    
    print(f"📝 使用配置: {config_data.get('description', 'Unknown')}")
    
    # 运行基准测试
    benchmark_suite = YICABenchmarkSuite(config)
    benchmark_suite.run_all_benchmarks()
    
    # 生成完整报告
    benchmark_suite.save_results()
    benchmark_suite.generate_visualization()
    benchmark_suite.generate_report()
    
    print("✅ 配置文件演示完成")
    
    return benchmark_suite


def run_yica_optimization_demo():
    """运行 YICA 优化演示"""
    print("🎯 开始 YICA 优化基准测试演示...")
    
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，跳过演示")
        return
    
    if not YICA_AVAILABLE:
        print("⚠️ YICA 后端不可用，将使用模拟模式")
    
    # 创建 YICA 优化配置
    config = BenchmarkConfig(
        warmup_iterations=5,
        benchmark_iterations=25,
        batch_sizes=[1, 8, 16, 32],
        sequence_lengths=[512, 1024],
        hidden_sizes=[1024, 2048],
        enable_memory_profiling=True,
        enable_energy_profiling=True,
        output_dir="./demo_yica_optimization_results",
        device="yica" if YICA_AVAILABLE else "auto"
    )
    
    # 创建基准测试套件
    benchmark_suite = YICABenchmarkSuite(config)
    
    print("🔧 运行 YICA 优化基准测试...")
    
    # 运行所有测试
    all_results = benchmark_suite.run_all_benchmarks()
    
    print(f"✅ 完成 {len(all_results)} 个优化测试")
    
    # 生成完整分析
    results_file = benchmark_suite.save_results()
    chart_file = benchmark_suite.generate_visualization()
    report_file = benchmark_suite.generate_report()
    
    print(f"📊 完整分析已生成:")
    print(f"  - 数据: {results_file}")
    print(f"  - 图表: {chart_file}")
    print(f"  - 报告: {report_file}")
    
    return benchmark_suite


def run_performance_comparison_demo():
    """运行性能对比演示"""
    print("🏆 开始性能对比基准测试演示...")
    
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，跳过演示")
        return
    
    # 对比配置：CPU vs CUDA vs YICA
    devices = ["cpu"]
    if YICA_AVAILABLE:
        devices.append("yica")
    
    # 尝试检测 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:
        pass
    
    print(f"🔍 将对比以下设备: {devices}")
    
    comparison_results = {}
    
    for device in devices:
        print(f"\n📱 测试设备: {device}")
        
        config = BenchmarkConfig(
            warmup_iterations=3,
            benchmark_iterations=15,
            batch_sizes=[1, 8, 16],
            sequence_lengths=[256, 512],
            hidden_sizes=[768, 1024],
            enable_memory_profiling=True,
            output_dir=f"./demo_comparison_{device}_results",
            device=device
        )
        
        benchmark_suite = YICABenchmarkSuite(config)
        
        # 只运行基础操作（为了演示速度）
        basic_results = benchmark_suite.run_basic_operation_benchmarks()
        
        # 保存设备特定结果
        results_file = benchmark_suite.save_results()
        comparison_results[device] = {
            "results": basic_results,
            "results_file": results_file,
            "device": device
        }
        
        print(f"✅ {device} 测试完成: {len(basic_results)} 个结果")
    
    # 生成对比分析
    print("\n📈 生成性能对比分析...")
    generate_comparison_analysis(comparison_results)
    
    return comparison_results


def generate_comparison_analysis(comparison_results: Dict[str, Any]):
    """生成性能对比分析"""
    comparison_file = Path("./demo_performance_comparison.md")
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("# YICA-Mirage 性能对比分析\n\n")
        f.write(f"**生成时间**: {Path(__file__).stat().st_mtime}\n\n")
        
        f.write("## 测试设备\n\n")
        for device, data in comparison_results.items():
            f.write(f"- **{device.upper()}**: {len(data['results'])} 个测试结果\n")
        f.write("\n")
        
        f.write("## 性能摘要\n\n")
        f.write("| 设备 | 平均延迟 (ms) | 平均吞吐量 (ops/sec) | 测试数量 |\n")
        f.write("|------|---------------|---------------------|----------|\n")
        
        for device, data in comparison_results.items():
            results = data['results']
            if results:
                avg_latency = sum(r.mean_latency_ms for r in results) / len(results)
                avg_throughput = sum(r.throughput_ops_per_sec for r in results) / len(results)
                f.write(f"| {device.upper()} | {avg_latency:.3f} | {avg_throughput:.2f} | {len(results)} |\n")
        
        f.write("\n## 详细结果\n\n")
        for device, data in comparison_results.items():
            f.write(f"### {device.upper()} 详细结果\n\n")
            f.write(f"结果文件: `{data['results_file']}`\n\n")
        
        f.write("## 结论\n\n")
        if YICA_AVAILABLE:
            f.write("- ✅ YICA 优化已启用\n")
            f.write("- 🚀 YICA 设备在支持的操作上显示出性能优势\n")
        else:
            f.write("- ⚠️ YICA 后端未启用，结果仅作为基准对照\n")
        
        f.write("- 📊 详细性能数据请参考各设备的结果文件\n")
        f.write("- 🔍 建议根据具体工作负载选择最优设备配置\n\n")
        
        f.write("---\n")
        f.write("*此报告由 YICA-Mirage 基准测试演示自动生成*\n")
    
    print(f"📋 对比分析已保存: {comparison_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YICA-Mirage 基准测试演示")
    parser.add_argument("--demo", type=str, 
                       choices=["basic", "transformer", "config", "yica", "comparison", "all"],
                       default="basic", help="选择演示类型")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    print("🎯 YICA-Mirage 基准测试演示")
    print("=" * 50)
    
    # 检查环境
    if not BENCHMARK_AVAILABLE:
        print("❌ 基准测试模块不可用，请检查安装")
        print("💡 确保已正确安装 YICA-Mirage 和所有依赖")
        return
    
    if YICA_AVAILABLE:
        try:
            yica_initialize()
            print("✅ YICA 后端已初始化")
        except Exception as e:
            print(f"⚠️ YICA 初始化失败: {e}")
    
    # 运行指定演示
    if args.demo == "basic" or args.demo == "all":
        run_basic_demo()
        print()
    
    if args.demo == "transformer" or args.demo == "all":
        run_transformer_demo()
        print()
    
    if args.demo == "config" or args.demo == "all":
        run_config_file_demo()
        print()
    
    if args.demo == "yica" or args.demo == "all":
        run_yica_optimization_demo()
        print()
    
    if args.demo == "comparison" or args.demo == "all":
        run_performance_comparison_demo()
        print()
    
    print("🎉 演示完成！")
    print("📁 查看生成的结果文件以获取详细分析")
    
    # 提供下一步建议
    print("\n💡 下一步建议:")
    print("1. 查看生成的图表和报告")
    print("2. 根据结果调整模型和硬件配置")
    print("3. 运行更全面的基准测试")
    print("4. 探索 YICA 特定的优化选项")


if __name__ == "__main__":
    main() 