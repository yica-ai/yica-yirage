#!/usr/bin/env python3
"""
YICA Backend 简单验证脚本
验证YICA backend集成的基本功能
"""

import os
import sys
import traceback

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yirage', 'python'))

def test_module_imports():
    """测试模块导入"""
    print("🧪 Testing module imports...")
    
    try:
        # 测试基本导入
        import torch
        print("✅ PyTorch import successful")
        
        import numpy as np
        print("✅ NumPy import successful")
        
    except ImportError as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        # 测试YICA backend模块导入
        from yirage.yica_backend_integration import (
            YICABackendIntegration, YICAKernelConfig, YICAMatMulKernel,
            YISInstructionType, YICAMemoryType, YICADataLayout
        )
        print("✅ YICA backend modules import successful")
        return True
        
    except ImportError as e:
        print(f"⚠️  YICA backend import failed: {e}")
        print("💡 This is expected if the full yirage environment is not set up")
        return False

def test_yica_backend_classes():
    """测试YICA backend类的基本功能"""
    print("\n🧪 Testing YICA backend classes...")
    
    try:
        from yirage.yica_backend_integration import (
            YICABackendIntegration, YICAKernelConfig, YICAMatMulKernel,
            YISInstructionType, YICAMemoryType, YICADataLayout, YICAComputeLevel
        )
        
        # 测试配置类
        config = YICAKernelConfig()
        assert config.grid_dim == (1, 1, 1)
        assert config.use_spm == True
        print("✅ YICAKernelConfig class working")
        
        # 测试枚举类
        assert YISInstructionType.YISMMA.value == "matrix_multiply"
        assert YICAMemoryType.SPM.value == "spm"
        assert YICADataLayout.TROWMAJOR.value == "tiled_row"
        print("✅ YICA enums working")
        
        # 测试MatMul kernel
        matmul_kernel = YICAMatMulKernel()
        assert matmul_kernel.operation_name == "yica_matmul"
        print("✅ YICAMatMulKernel class working")
        
        # 测试backend集成类
        backend = YICABackendIntegration()
        assert backend.device_properties.name == "YICA-G100"
        assert len(backend.kernel_registry.list_operations()) > 0
        print("✅ YICABackendIntegration class working")
        
        return True
        
    except Exception as e:
        print(f"❌ YICA backend class test failed: {e}")
        traceback.print_exc()
        return False

def test_yis_instruction_generation():
    """测试YIS指令生成"""
    print("\n🧪 Testing YIS instruction generation...")
    
    try:
        import torch
        from yirage.yica_backend_integration import YICAMatMulKernel
        
        # 创建测试矩阵
        A = torch.randn(64, 32, dtype=torch.float16)
        B = torch.randn(32, 128, dtype=torch.float16)
        
        # 创建MatMul kernel并生成指令
        matmul_kernel = YICAMatMulKernel()
        yis_instructions = matmul_kernel.generate_yis_instructions(A, B)
        
        # 验证指令
        assert isinstance(yis_instructions, list)
        assert len(yis_instructions) > 0
        
        # 检查指令内容
        instruction_text = "\n".join(yis_instructions)
        assert "yis.ecopy.g2spm" in instruction_text
        assert "yis.mma." in instruction_text
        assert "yis.sync.bar" in instruction_text
        
        print(f"✅ YIS instruction generation working - {len(yis_instructions)} instructions")
        print("📝 Sample YIS instructions:")
        for i, instr in enumerate(yis_instructions[:5]):
            print(f"   {i+1}: {instr}")
        if len(yis_instructions) > 5:
            print(f"   ... and {len(yis_instructions)-5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ YIS instruction generation test failed: {e}")
        traceback.print_exc()
        return False

def test_kernel_execution():
    """测试kernel执行"""
    print("\n🧪 Testing kernel execution...")
    
    try:
        import torch
        from yirage.yica_backend_integration import get_yica_backend, yica_matmul
        
        # 创建测试数据
        A = torch.randn(32, 64, dtype=torch.float16)
        B = torch.randn(64, 128, dtype=torch.float16)
        
        # 执行YICA矩阵乘法
        result = yica_matmul(A, B)
        
        # 验证结果
        expected = torch.matmul(A, B)
        assert result.shape == expected.shape
        
        # 验证数值正确性（允许小误差）
        max_diff = torch.max(torch.abs(result - expected)).item()
        assert max_diff < 1e-2, f"Accuracy error too large: {max_diff}"
        
        print(f"✅ Kernel execution working - max difference: {max_diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Kernel execution test failed: {e}")
        traceback.print_exc()
        return False

def test_superoptimize_integration():
    """测试superoptimize集成"""
    print("\n🧪 Testing superoptimize integration...")
    
    try:
        from yirage.kernel import Graph
        
        # 检查kernel.py中是否有YICA backend支持
        import inspect
        from yirage.kernel import Graph
        
        # 获取superoptimize方法的源码
        source = inspect.getsource(Graph.superoptimize)
        
        # 检查是否包含YICA backend代码
        assert 'backend == "yica"' in source
        assert 'yica_backend_integration' in source
        
        print("✅ Superoptimize YICA integration found in kernel.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Superoptimize integration test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_estimation():
    """测试性能估算"""
    print("\n🧪 Testing performance estimation...")
    
    try:
        import torch
        from yirage.yica_backend_integration import get_yica_backend
        
        backend = get_yica_backend()
        
        # 测试矩阵乘法性能估算
        A = torch.randn(128, 256, dtype=torch.float16)
        B = torch.randn(256, 512, dtype=torch.float16)
        
        matmul_kernel = backend.kernel_registry.get_kernel("matmul")
        perf_estimate = matmul_kernel.estimate_performance(A, B)
        
        # 验证性能估算结果
        assert "estimated_flops" in perf_estimate
        assert "estimated_latency_ms" in perf_estimate
        assert "spm_utilization" in perf_estimate
        assert "cim_efficiency" in perf_estimate
        
        assert perf_estimate["estimated_flops"] > 0
        assert perf_estimate["spm_utilization"] <= 1.0
        assert perf_estimate["cim_efficiency"] <= 1.0
        
        print("✅ Performance estimation working")
        print(f"   Estimated FLOPS: {perf_estimate['estimated_flops']:.0f}")
        print(f"   Estimated latency: {perf_estimate['estimated_latency_ms']:.3f} ms")
        print(f"   SPM utilization: {perf_estimate['spm_utilization']:.2f}")
        print(f"   CIM efficiency: {perf_estimate['cim_efficiency']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance estimation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("🚀 YICA Backend Simple Validation")
    print("="*60)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("Module Imports", test_module_imports),
        ("YICA Backend Classes", test_yica_backend_classes),
        ("YIS Instruction Generation", test_yis_instruction_generation),
        ("Kernel Execution", test_kernel_execution),
        ("Superoptimize Integration", test_superoptimize_integration),
        ("Performance Estimation", test_performance_estimation),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"❌ {test_name} encountered an error: {e}")
            test_results.append((test_name, "ERROR"))
    
    # 打印测试结果总结
    print("\n" + "="*60)
    print("📊 YICA Backend Validation Summary")
    print("="*60)
    
    passed = sum(1 for _, status in test_results if status == "PASSED")
    total = len(test_results)
    
    for test_name, status in test_results:
        status_icon = "✅" if status == "PASSED" else ("⚠️" if status == "FAILED" else "❌")
        print(f"   {status_icon} {test_name}: {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! YICA Backend integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the details above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 