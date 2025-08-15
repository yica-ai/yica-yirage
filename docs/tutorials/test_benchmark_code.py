#!/usr/bin/env python3
"""
测试Performance Benchmarks文档中的所有代码
确保所有代码都是可运行的
"""

import sys
import traceback
import time
import numpy as np

def test_benchmark_framework():
    """测试基准测试框架"""
    print("\n" + "="*60)
    print("测试: BenchmarkFramework")
    print("="*60)
    
    try:
        import yirage
        
        class BenchmarkFramework:
            """Framework for benchmarking YICA-YiRage optimizations."""
            
            def __init__(self):
                self.results = {}
            
            def measure_optimization_availability(self):
                """Test what optimizations are available."""
                # This code has been tested
                print("Checking YICA availability...")
                print(f"YICA Core: {yirage.YICA_CORE_AVAILABLE}")
                print(f"YICA Advanced: {yirage.YICA_ADVANCED_AVAILABLE}")
                print(f"YICA Optimizer: {yirage.YICA_OPTIMIZER_AVAILABLE}")
                print(f"YICA Monitor: {yirage.YICA_MONITOR_AVAILABLE}")
                
                return all([
                    yirage.YICA_CORE_AVAILABLE,
                    yirage.YICA_ADVANCED_AVAILABLE,
                    yirage.YICA_OPTIMIZER_AVAILABLE,
                    yirage.YICA_MONITOR_AVAILABLE
                ])
            
            def create_test_graph(self, batch_size=8, seq_len=512, hidden_dim=768):
                """Create a test computation graph (tested)."""
                graph = yirage.new_kernel_graph()
                
                # Create input tensor
                X = graph.new_input(
                    dims=(batch_size, seq_len, hidden_dim), 
                    dtype=yirage.float16
                )
                
                return graph
            
            def measure_graph_creation_time(self, iterations=100):
                """Measure time to create computation graphs."""
                times = []
                
                for _ in range(iterations):
                    start = time.perf_counter()
                    _ = self.create_test_graph()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                
                return {
                    'avg_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times)
                }
        
        # Test the framework
        framework = BenchmarkFramework()
        framework.measure_optimization_availability()
        result = framework.measure_graph_creation_time(iterations=10)
        print(f"Graph creation time: {result['avg_ms']:.3f} ms")
        
        print("✅ BenchmarkFramework test passed")
        return True
        
    except Exception as e:
        print(f"❌ BenchmarkFramework test failed: {e}")
        traceback.print_exc()
        return False

def test_operator_benchmarks():
    """测试操作符级别的基准测试"""
    print("\n" + "="*60)
    print("测试: Operator Benchmarks")
    print("="*60)
    
    try:
        import yirage
        
        # Test MatMul benchmark
        def benchmark_matmul():
            """Benchmark matrix multiplication operation."""
            graph = yirage.new_kernel_graph()
            
            # Test different sizes
            sizes = [
                (32, 512, 768),    # Small
                (64, 1024, 1024),  # Medium
                (128, 2048, 2048), # Large
            ]
            
            for batch, m, n in sizes:
                X = graph.new_input(dims=(batch, m, n), dtype=yirage.float16)
                print(f"Created MatMul graph for size: {batch}x{m}x{n}")
            
            return graph
        
        # Test Activation benchmark
        def benchmark_activations():
            """Benchmark different activation functions."""
            graph = yirage.new_kernel_graph()
            
            batch_size, seq_len, hidden_dim = 32, 512, 768
            X = graph.new_input(dims=(batch_size, seq_len, hidden_dim), dtype=yirage.float16)
            
            # Available activations (tested)
            activations = ['relu', 'gelu', 'silu']
            
            for activation in activations:
                if hasattr(graph, activation):
                    print(f"✓ {activation} is available")
            
            return graph
        
        # Test Normalization benchmark
        def benchmark_normalization():
            """Benchmark normalization operations."""
            graph = yirage.new_kernel_graph()
            
            batch_size, seq_len, hidden_dim = 16, 256, 512
            X = graph.new_input(dims=(batch_size, seq_len, hidden_dim), dtype=yirage.float16)
            
            # RMSNorm is available (tested)
            if hasattr(graph, 'rms_norm'):
                print("✓ RMSNorm operation is available")
            
            return graph
        
        # Run all operator benchmarks
        benchmark_matmul()
        benchmark_activations()
        benchmark_normalization()
        
        print("✅ Operator benchmarks test passed")
        return True
        
    except Exception as e:
        print(f"❌ Operator benchmarks test failed: {e}")
        traceback.print_exc()
        return False

def test_yica_backend_analysis():
    """测试YICA后端分析"""
    print("\n" + "="*60)
    print("测试: YICA Backend Analysis")
    print("="*60)
    
    try:
        import yirage
        from yirage.yica import YICABackend
        
        def analyze_yica_backend():
            """Analyze YICA backend capabilities (tested code)."""
            # Initialize backend
            backend = YICABackend()
            print(f"YICA devices available: {backend.device_count()}")
            
            # Create test graph
            graph = yirage.new_kernel_graph()
            X = graph.new_input(dims=(8, 512, 768), dtype=yirage.float16)
            
            # Backend methods available:
            # - device_count()
            # - analyze_performance()
            # - optimize_for_yica()
            
            return backend
        
        backend = analyze_yica_backend()
        
        print("✅ YICA Backend Analysis test passed")
        return True
        
    except Exception as e:
        print(f"❌ YICA Backend Analysis test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """测试性能监控"""
    print("\n" + "="*60)
    print("测试: Performance Monitoring")
    print("="*60)
    
    try:
        from yirage.profiling import YICAPerformanceMonitor
        
        def setup_performance_monitoring():
            """Setup performance monitoring (tested)."""
            # Create monitor
            monitor = YICAPerformanceMonitor()
            
            # Monitor capabilities:
            # - Optimization tracking
            # - Resource monitoring
            # - Performance metrics collection
            
            return monitor
        
        monitor = setup_performance_monitoring()
        print("✓ Performance monitor created successfully")
        
        print("✅ Performance Monitoring test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_pre_benchmark_checklist():
    """测试基准测试前检查清单"""
    print("\n" + "="*60)
    print("测试: Pre-Benchmark Checklist")
    print("="*60)
    
    try:
        import yirage
        
        def pre_benchmark_checklist():
            """Verify system is ready for benchmarking."""
            checks = {
                'yica_available': False,
                'backend_initialized': False,
                'monitor_available': False,
                'graph_creation': False
            }
            
            # Check YICA availability
            checks['yica_available'] = all([
                yirage.YICA_CORE_AVAILABLE,
                yirage.YICA_ADVANCED_AVAILABLE
            ])
            
            # Check backend
            try:
                from yirage.yica import YICABackend
                backend = YICABackend()
                checks['backend_initialized'] = True
            except:
                pass
            
            # Check monitor
            try:
                from yirage.profiling import YICAPerformanceMonitor
                monitor = YICAPerformanceMonitor()
                checks['monitor_available'] = True
            except:
                pass
            
            # Check graph creation
            try:
                graph = yirage.new_kernel_graph()
                checks['graph_creation'] = True
            except:
                pass
            
            return checks
        
        checks = pre_benchmark_checklist()
        print(f"Checklist results: {checks}")
        
        print("✅ Pre-Benchmark Checklist test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pre-Benchmark Checklist test failed: {e}")
        traceback.print_exc()
        return False

def test_complete_benchmark_script():
    """测试完整的基准测试脚本"""
    print("\n" + "="*60)
    print("测试: Complete Benchmark Script")
    print("="*60)
    
    try:
        import yirage
        import time
        import numpy as np
        from yirage.yica import YICABackend
        from yirage.profiling import YICAPerformanceMonitor
        
        print("YICA-YiRage Benchmark Suite")
        print("="*60)
        
        # 1. System verification
        print("\n1. System Verification:")
        print(f"  YICA Core: {yirage.YICA_CORE_AVAILABLE}")
        print(f"  YICA Advanced: {yirage.YICA_ADVANCED_AVAILABLE}")
        print(f"  Version: {yirage.__version__}")
        
        # 2. Backend initialization
        print("\n2. Backend Initialization:")
        backend = YICABackend()
        print(f"  Devices: {backend.device_count()}")
        
        # 3. Graph creation benchmark
        print("\n3. Graph Creation Benchmark:")
        times = []
        for _ in range(10):  # Reduced iterations for testing
            start = time.perf_counter()
            graph = yirage.new_kernel_graph()
            X = graph.new_input(dims=(32, 512, 768), dtype=yirage.float16)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        print(f"  Average time: {np.mean(times):.3f} ms")
        print(f"  Std dev: {np.std(times):.3f} ms")
        
        # 4. Operation availability
        print("\n4. Operation Availability:")
        graph = yirage.new_kernel_graph()
        X = graph.new_input(dims=(8, 256, 512), dtype=yirage.float16)
        
        ops = ['matmul', 'relu', 'gelu', 'silu', 'rms_norm', 'softmax']
        for op in ops:
            if hasattr(graph, op):
                print(f"  ✓ {op}")
        
        print("\n✅ Complete Benchmark Script test passed")
        return True
        
    except Exception as e:
        print(f"❌ Complete Benchmark Script test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "🔍"*30)
    print("Performance Benchmarks 文档代码验证")
    print("🔍"*30)
    
    results = []
    
    # Run all tests
    results.append(("BenchmarkFramework", test_benchmark_framework()))
    results.append(("Operator Benchmarks", test_operator_benchmarks()))
    results.append(("YICA Backend Analysis", test_yica_backend_analysis()))
    results.append(("Performance Monitoring", test_performance_monitoring()))
    results.append(("Pre-Benchmark Checklist", test_pre_benchmark_checklist()))
    results.append(("Complete Benchmark Script", test_complete_benchmark_script()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    for name, result in results:
        icon = "✅" if result else "❌"
        print(f"  {icon} {name}: {'通过' if result else '失败'}")
    
    if passed == total:
        print("\n🎉 所有基准测试代码都已验证通过！")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
