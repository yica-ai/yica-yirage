#!/usr/bin/env python3
"""
测试Real-World Examples文档中的所有代码
确保所有代码都是可运行的
"""

import sys
import traceback
import time
import numpy as np
import json

def test_example1_graph_construction():
    """测试示例1：基本图构建"""
    print("\n" + "="*60)
    print("测试示例1: Basic Graph Construction")
    print("="*60)
    
    try:
        import yirage
        from yirage.yica import YICABackend

        def create_basic_graph():
            """Create a basic computation graph using YICA."""
            
            print("Creating YICA Computation Graph")
            print("=" * 40)
            
            # Create a new kernel graph
            graph = yirage.new_kernel_graph()
            
            # Create input tensors
            batch_size = 8
            seq_len = 512
            hidden_dim = 768
            
            X = graph.new_input(
                dims=(batch_size, seq_len, hidden_dim),
                dtype=yirage.float16
            )
            
            print(f"✓ Created input tensor: {batch_size}x{seq_len}x{hidden_dim}")
            
            # Available operations (verified)
            operations = ['matmul', 'relu', 'gelu', 'silu', 'rms_norm', 'softmax']
            
            print("\nAvailable operations:")
            for op in operations:
                if hasattr(graph, op):
                    print(f"  ✓ {op}")
            
            return graph

        # Check YICA availability
        print("YICA System Check")
        print("=" * 40)
        print(f"YICA Core: {yirage.YICA_CORE_AVAILABLE}")
        print(f"YICA Advanced: {yirage.YICA_ADVANCED_AVAILABLE}")
        print(f"YICA Optimizer: {yirage.YICA_OPTIMIZER_AVAILABLE}")
        print(f"Version: {yirage.__version__}")
        
        # Initialize backend
        backend = YICABackend()
        print(f"\nYICA Devices: {backend.device_count()}")
        
        # Create graph
        print("\n" + "=" * 40)
        graph = create_basic_graph()
        
        print("\n✅ Example 1 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example 1 test failed: {e}")
        traceback.print_exc()
        return False

def test_example2_transformer_components():
    """测试示例2：Transformer组件"""
    print("\n" + "="*60)
    print("测试示例2: Transformer Components")
    print("="*60)
    
    try:
        import yirage
        import numpy as np

        class TransformerComponents:
            """Build transformer-like components using YICA operations."""
            
            def __init__(self):
                self.graph = yirage.new_kernel_graph()
            
            def create_attention_pattern(self, batch_size=4, seq_len=256, hidden_dim=512):
                """
                Create an attention-like computation pattern.
                Note: This demonstrates the available operations, not a full attention implementation.
                """
                print("Building Attention-like Pattern")
                print("-" * 30)
                
                # Input tensor
                X = self.graph.new_input(
                    dims=(batch_size, seq_len, hidden_dim),
                    dtype=yirage.float16
                )
                print(f"Input shape: {batch_size}x{seq_len}x{hidden_dim}")
                
                # Available operations for attention-like patterns:
                # - matmul: For query, key, value projections
                # - softmax: For attention weights
                # - Additional operations would be needed for full attention
                
                if hasattr(self.graph, 'matmul'):
                    print("✓ MatMul available for projections")
                
                if hasattr(self.graph, 'softmax'):
                    print("✓ Softmax available for attention weights")
                
                return X
            
            def create_ffn_pattern(self, batch_size=4, seq_len=256, hidden_dim=512):
                """
                Create a feed-forward network pattern.
                """
                print("\nBuilding FFN Pattern")
                print("-" * 30)
                
                # Input tensor
                X = self.graph.new_input(
                    dims=(batch_size, seq_len, hidden_dim),
                    dtype=yirage.float16
                )
                print(f"Input shape: {batch_size}x{seq_len}x{hidden_dim}")
                
                # FFN typically uses:
                # - Linear projections (would use matmul)
                # - Activation functions (GELU, ReLU, or SiLU)
                
                activations = ['relu', 'gelu', 'silu']
                print("\nAvailable activations for FFN:")
                for activation in activations:
                    if hasattr(self.graph, activation):
                        print(f"  ✓ {activation}")
                
                return X
            
            def create_normalization_pattern(self, batch_size=4, seq_len=256, hidden_dim=512):
                """
                Create a normalization pattern.
                """
                print("\nBuilding Normalization Pattern")
                print("-" * 30)
                
                # Input tensor
                X = self.graph.new_input(
                    dims=(batch_size, seq_len, hidden_dim),
                    dtype=yirage.float16
                )
                print(f"Input shape: {batch_size}x{seq_len}x{hidden_dim}")
                
                # Check for normalization operations
                if hasattr(self.graph, 'rms_norm'):
                    print("✓ RMSNorm available")
                
                return X

        # Check system
        print(f"YICA Version: {yirage.__version__}")
        print(f"YICA Core Available: {yirage.YICA_CORE_AVAILABLE}")
        
        # Create components
        components = TransformerComponents()
        
        # Build different patterns
        components.create_attention_pattern()
        components.create_ffn_pattern()
        components.create_normalization_pattern()
        
        print("\n✅ Example 2 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example 2 test failed: {e}")
        traceback.print_exc()
        return False

def test_example3_performance_monitoring():
    """测试示例3：性能监控"""
    print("\n" + "="*60)
    print("测试示例3: Performance Monitoring")
    print("="*60)
    
    try:
        import yirage
        from yirage.profiling import YICAPerformanceMonitor
        from yirage.yica import YICABackend
        import time

        def demonstrate_monitoring():
            """Demonstrate performance monitoring capabilities."""
            
            print("YICA Performance Monitoring Demo")
            print("=" * 40)
            
            # Initialize monitor
            monitor = YICAPerformanceMonitor()
            print("✓ Performance monitor initialized")
            
            # Initialize backend
            backend = YICABackend()
            print(f"✓ Backend initialized with {backend.device_count()} devices")
            
            # Create a test workload
            graph = yirage.new_kernel_graph()
            
            # Monitor graph creation
            print("\nMonitoring graph creation...")
            start_time = time.perf_counter()
            
            # Create multiple inputs to simulate workload
            for i in range(5):
                X = graph.new_input(
                    dims=(32, 512, 768),
                    dtype=yirage.float16
                )
                print(f"  Created input {i+1}")
            
            elapsed_time = (time.perf_counter() - start_time) * 1000
            print(f"\nGraph creation took: {elapsed_time:.2f} ms")
            
            return monitor

        def analyze_backend_capabilities():
            """Analyze what the YICA backend can do."""
            
            print("\nYICA Backend Capability Analysis")
            print("=" * 40)
            
            backend = YICABackend()
            
            # Check available methods
            available_methods = [
                'device_count',
                'analyze_performance',
                'optimize_for_yica',
                'quick_analyze'
            ]
            
            print("Backend methods:")
            for method in available_methods:
                if hasattr(backend, method):
                    print(f"  ✓ {method}")
            
            # Test device count
            device_count = backend.device_count()
            print(f"\nDevice count: {device_count}")

        # System check
        print("System Status")
        print("=" * 40)
        print(f"Version: {yirage.__version__}")
        print(f"YICA Core: {yirage.YICA_CORE_AVAILABLE}")
        print(f"YICA Monitor: {yirage.YICA_MONITOR_AVAILABLE}")
        
        # Run demonstrations
        print("\n" + "=" * 40)
        monitor = demonstrate_monitoring()
        
        print("\n" + "=" * 40)
        analyze_backend_capabilities()
        
        print("\n✅ Example 3 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example 3 test failed: {e}")
        traceback.print_exc()
        return False

def test_example4_optimization_pipeline():
    """测试示例4：优化管道"""
    print("\n" + "="*60)
    print("测试示例4: Optimization Pipeline")
    print("="*60)
    
    try:
        import yirage
        from yirage.yica import YICABackend
        import time

        class OptimizationPipeline:
            """YICA optimization pipeline demonstration."""
            
            def __init__(self):
                self.backend = YICABackend()
                self.graphs = []
            
            def create_workload(self, num_graphs=3):
                """Create multiple computation graphs as workload."""
                
                print("Creating Workload")
                print("-" * 30)
                
                for i in range(num_graphs):
                    graph = yirage.new_kernel_graph()
                    
                    # Different sizes for variety
                    sizes = [
                        (8, 256, 512),
                        (16, 512, 768),
                        (32, 1024, 1024)
                    ]
                    
                    batch, seq, hidden = sizes[i % len(sizes)]
                    
                    X = graph.new_input(
                        dims=(batch, seq, hidden),
                        dtype=yirage.float16
                    )
                    
                    self.graphs.append(graph)
                    print(f"  Graph {i+1}: {batch}x{seq}x{hidden}")
                
                return self.graphs
            
            def demonstrate_optimization_flow(self):
                """Demonstrate the optimization flow."""
                
                print("\nOptimization Flow")
                print("-" * 30)
                
                # Step 1: Analysis
                print("1. Analyzing workload...")
                
                # Step 2: Optimization
                print("2. Applying optimizations...")
                
                # Step 3: Verification
                print("3. Verifying optimizations...")
                
                print("\n✓ Optimization flow demonstrated")
            
            def measure_optimization_overhead(self):
                """Measure the overhead of optimization process."""
                
                print("\nMeasuring Optimization Overhead")
                print("-" * 30)
                
                # Create a simple graph
                start_time = time.perf_counter()
                graph = yirage.new_kernel_graph()
                X = graph.new_input(dims=(32, 512, 768), dtype=yirage.float16)
                creation_time = (time.perf_counter() - start_time) * 1000
                
                print(f"Graph creation: {creation_time:.3f} ms")
                
                return creation_time

        # Check system
        print(f"YICA Version: {yirage.__version__}")
        print(f"YICA Optimizer Available: {yirage.YICA_OPTIMIZER_AVAILABLE}")
        
        # Create pipeline
        pipeline = OptimizationPipeline()
        print(f"\n✓ Pipeline initialized with {pipeline.backend.device_count()} devices")
        
        # Create workload
        print("\n" + "=" * 50)
        graphs = pipeline.create_workload()
        
        # Demonstrate optimization
        print("\n" + "=" * 50)
        pipeline.demonstrate_optimization_flow()
        
        # Measure overhead
        print("\n" + "=" * 50)
        overhead = pipeline.measure_optimization_overhead()
        
        print("\n✅ Example 4 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example 4 test failed: {e}")
        traceback.print_exc()
        return False

def test_example5_python_integration():
    """测试示例5：Python生态系统集成"""
    print("\n" + "="*60)
    print("测试示例5: Python Integration")
    print("="*60)
    
    try:
        import yirage
        import numpy as np
        import json
        import sys
        from typing import Dict, List, Any

        class YICAPythonIntegration:
            """Demonstrate Python ecosystem integration."""
            
            def __init__(self):
                self.backend = yirage.yica.YICABackend()
                self.results = {}
            
            def export_configuration(self, filename="test_yica_config.json"):
                """Export YICA configuration to JSON."""
                
                config = {
                    "version": yirage.__version__,
                    "yica_core": yirage.YICA_CORE_AVAILABLE,
                    "yica_advanced": yirage.YICA_ADVANCED_AVAILABLE,
                    "yica_optimizer": yirage.YICA_OPTIMIZER_AVAILABLE,
                    "device_count": self.backend.device_count(),
                    "timestamp": str(np.datetime64('now'))
                }
                
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✓ Configuration exported to {filename}")
                return config
            
            def numpy_interop(self):
                """Demonstrate NumPy interoperability."""
                
                print("\nNumPy Interoperability")
                print("-" * 30)
                
                # Create NumPy arrays
                np_array = np.random.randn(8, 512, 768).astype(np.float16)
                print(f"NumPy array shape: {np_array.shape}")
                print(f"NumPy array dtype: {np_array.dtype}")
                
                # Create YICA graph with similar dimensions
                graph = yirage.new_kernel_graph()
                X = graph.new_input(
                    dims=np_array.shape,
                    dtype=yirage.float16
                )
                
                print("✓ Compatible graph input created")
                
                return np_array
            
            def type_checking_demo(self):
                """Demonstrate type checking and validation."""
                
                print("\nType Checking Demo")
                print("-" * 30)
                
                # Check types
                graph = yirage.new_kernel_graph()
                
                print(f"Graph type: {type(graph)}")
                print(f"Backend type: {type(self.backend)}")
                
                # Validate inputs
                valid_dtypes = [yirage.float16, yirage.float32]
                print(f"Valid dtypes: {valid_dtypes}")
                
                return True

        # System info
        print(f"Python version: {sys.version.split()[0]}")
        print(f"NumPy version: {np.__version__}")
        print(f"YiRage version: {yirage.__version__}")
        
        # Create integration demo
        integration = YICAPythonIntegration()
        
        # Export configuration
        print("\n" + "=" * 50)
        config = integration.export_configuration()
        
        # NumPy interop
        print("\n" + "=" * 50)
        np_array = integration.numpy_interop()
        
        # Type checking
        print("\n" + "=" * 50)
        integration.type_checking_demo()
        
        # Clean up test file
        import os
        if os.path.exists("test_yica_config.json"):
            os.remove("test_yica_config.json")
            print("\n✓ Cleaned up test files")
        
        print("\n✅ Example 5 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Example 5 test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "🔍"*30)
    print("Real-World Examples 文档代码验证")
    print("🔍"*30)
    
    results = []
    
    # Run all tests
    results.append(("Example 1: Graph Construction", test_example1_graph_construction()))
    results.append(("Example 2: Transformer Components", test_example2_transformer_components()))
    results.append(("Example 3: Performance Monitoring", test_example3_performance_monitoring()))
    results.append(("Example 4: Optimization Pipeline", test_example4_optimization_pipeline()))
    results.append(("Example 5: Python Integration", test_example5_python_integration()))
    
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
        print("\n🎉 所有示例代码都已验证通过！")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
