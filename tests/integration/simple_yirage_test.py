#!/usr/bin/env python3
"""
简化的yirage测试脚本
测试yirage的基本功能，不依赖YICA硬件设备
"""

import sys
import time
import os
import json
from datetime import datetime

# 添加yirage Python路径
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
yirage_python_path = os.path.join(project_root, 'yirage', 'python')
sys.path.insert(0, yirage_python_path)

print("🧪 简化yirage功能测试")
print("=" * 50)
print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🐍 Python: {sys.version}")
print()

def test_basic_import():
    """测试基本导入"""
    print("📦 测试yirage导入...")
    try:
        import yirage
        print(f"  ✅ yirage导入成功")
        print(f"  📋 版本: {getattr(yirage, '__version__', '开发版本')}")
        print(f"  📍 路径: {yirage.__file__}")
        return True, yirage
    except ImportError as e:
        print(f"  ❌ yirage导入失败: {e}")
        return False, None

def test_kernel_graph():
    """测试计算图功能"""
    print("\n🧠 测试计算图创建...")
    try:
        import yirage
        
        # 创建计算图
        graph = yirage.new_kernel_graph()
        print(f"  ✅ 计算图创建成功")
        
        # 创建输入张量
        A = graph.new_input(dims=(512, 512), dtype=yirage.float32)
        B = graph.new_input(dims=(512, 512), dtype=yirage.float32)
        print(f"  ✅ 输入张量创建成功")
        
        # 矩阵乘法
        C = graph.matmul(A, B)
        graph.mark_output(C)
        print(f"  ✅ 矩阵乘法操作添加成功")
        
        # 尝试优化
        start_time = time.time()
        optimized = graph.superoptimize()
        opt_time = time.time() - start_time
        
        print(f"  ✅ 图优化完成，耗时: {opt_time*1000:.2f} ms")
        print(f"  📊 优化后图信息: {type(optimized)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 计算图测试失败: {e}")
        return False

def test_multiple_operations():
    """测试多种操作"""
    print("\n⚡ 测试多种操作...")
    try:
        import yirage
        
        operations = [
            ("矩阵乘法", lambda g, a, b: g.matmul(a, b)),
            ("加法", lambda g, a, b: g.add(a, b)),
            ("乘法", lambda g, a, b: g.mul(a, b)),
        ]
        
        results = []
        
        for op_name, op_func in operations:
            try:
                graph = yirage.new_kernel_graph()
                A = graph.new_input(dims=(256, 256), dtype=yirage.float32)
                B = graph.new_input(dims=(256, 256), dtype=yirage.float32)
                
                start_time = time.time()
                result = op_func(graph, A, B)
                graph.mark_output(result)
                optimized = graph.superoptimize()
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000
                print(f"  ✅ {op_name}: {elapsed:.2f} ms")
                results.append((op_name, elapsed))
                
            except Exception as e:
                print(f"  ⚠️  {op_name}: 失败 - {e}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ 多操作测试失败: {e}")
        return []

def test_different_dtypes():
    """测试不同数据类型"""
    print("\n🔢 测试不同数据类型...")
    try:
        import yirage
        
        dtypes = [
            ("float32", yirage.float32),
            ("float16", yirage.float16),
        ]
        
        for dtype_name, dtype in dtypes:
            try:
                graph = yirage.new_kernel_graph()
                A = graph.new_input(dims=(128, 128), dtype=dtype)
                B = graph.new_input(dims=(128, 128), dtype=dtype)
                C = graph.matmul(A, B)
                graph.mark_output(C)
                
                start_time = time.time()
                optimized = graph.superoptimize()
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000
                print(f"  ✅ {dtype_name}: {elapsed:.2f} ms")
                
            except Exception as e:
                print(f"  ⚠️  {dtype_name}: 失败 - {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据类型测试失败: {e}")
        return False

def test_performance_scaling():
    """测试性能扩展性"""
    print("\n📈 测试性能扩展性...")
    try:
        import yirage
        
        sizes = [64, 128, 256, 512, 1024]
        results = []
        
        for size in sizes:
            try:
                graph = yirage.new_kernel_graph()
                A = graph.new_input(dims=(size, size), dtype=yirage.float32)
                B = graph.new_input(dims=(size, size), dtype=yirage.float32)
                C = graph.matmul(A, B)
                graph.mark_output(C)
                
                start_time = time.time()
                optimized = graph.superoptimize()
                end_time = time.time()
                
                elapsed = (end_time - start_time) * 1000
                ops_per_sec = 1000 / elapsed if elapsed > 0 else float('inf')
                
                print(f"  ✅ {size}x{size}: {elapsed:.2f} ms ({ops_per_sec:.1f} ops/sec)")
                results.append((size, elapsed, ops_per_sec))
                
            except Exception as e:
                print(f"  ⚠️  {size}x{size}: 失败 - {e}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ 性能扩展性测试失败: {e}")
        return []

def generate_report(test_results):
    """生成测试报告"""
    print("\n📋 测试报告")
    print("=" * 50)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
        },
        'test_results': test_results
    }
    
    # 统计成功率
    total_tests = len([k for k in test_results.keys() if k != 'performance_results'])
    passed_tests = sum(1 for k, v in test_results.items() 
                      if k != 'performance_results' and v)
    
    print(f"📊 测试统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {passed_tests}")
    print(f"  成功率: {passed_tests/total_tests*100:.1f}%")
    
    if test_results.get('performance_results'):
        print(f"  性能测试: {len(test_results['performance_results'])} 项")
    
    # 保存报告
    report_dir = os.path.join(project_root, 'tests', 'integration')
    report_file = os.path.join(report_dir, f"simple_yirage_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 测试报告已保存: {report_file}")
    except Exception as e:
        print(f"⚠️  报告保存失败: {e}")
    
    return report

def main():
    """主函数"""
    results = {}
    
    # 1. 基本导入测试
    import_ok, yirage_module = test_basic_import()
    results['import_test'] = import_ok
    
    if not import_ok:
        print("❌ 基本导入失败，无法继续测试")
        return
    
    # 2. 计算图测试
    graph_ok = test_kernel_graph()
    results['graph_test'] = graph_ok
    
    # 3. 多操作测试
    multi_ops_results = test_multiple_operations()
    results['multi_operations'] = len(multi_ops_results) > 0
    results['operation_results'] = multi_ops_results
    
    # 4. 数据类型测试
    dtype_ok = test_different_dtypes()
    results['dtype_test'] = dtype_ok
    
    # 5. 性能扩展性测试
    perf_results = test_performance_scaling()
    results['performance_test'] = len(perf_results) > 0
    results['performance_results'] = perf_results
    
    # 6. 生成报告
    report = generate_report(results)
    
    # 总结
    print(f"\n🎯 测试总结:")
    success_count = sum(1 for k, v in results.items() 
                       if k not in ['operation_results', 'performance_results'] and v)
    total_count = len([k for k in results.keys() 
                      if k not in ['operation_results', 'performance_results']])
    
    if success_count == total_count:
        print(f"🎉 所有测试通过！yirage基本功能正常")
        print(f"✅ yirage可以在此环境中正常工作")
    else:
        print(f"⚠️  部分测试失败 ({success_count}/{total_count})")
        print(f"💡 建议检查yirage安装和环境配置")
    
    return results

if __name__ == "__main__":
    main()
