#!/usr/bin/env python3
"""
简化的YICA集成测试
避免复杂的模块导入，直接验证核心功能
"""

import subprocess
import sys
import time
import json
import os

def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n🔄 {description}")
    print(f"   命令: {cmd}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ 成功 (耗时: {execution_time:.2f}s)")
            return True, result.stdout, execution_time
        else:
            print(f"   ❌ 失败 (返回码: {result.returncode})")
            print(f"   错误输出: {result.stderr}")
            return False, result.stderr, execution_time
    except subprocess.TimeoutExpired:
        print(f"   ⏰ 超时 (>60s)")
        return False, "Timeout", 60.0
    except Exception as e:
        print(f"   💥 异常: {e}")
        return False, str(e), 0.0

def test_yica_standalone():
    """测试YICA独立演示程序"""
    print("🚀 测试YICA独立演示程序")
    print("=" * 50)
    
    success, output, exec_time = run_command("python demo_yica_standalone.py", "运行YICA演示")
    
    if success:
        print("   📊 输出分析:")
        lines = output.split('\n')
        
        # 提取关键信息
        performance_lines = [line for line in lines if '加速比:' in line or 'CIM利用率:' in line]
        for line in performance_lines[:5]:  # 只显示前5行
            print(f"     {line.strip()}")
        
        # 检查生成的文件
        expected_files = ['yica_matmul_kernel.py', 'yica_attention_kernel.py']
        generated_files = []
        for file in expected_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                generated_files.append(f"{file} ({size} bytes)")
                print(f"   📄 生成文件: {file} ({size} bytes)")
        
        print(f"   ✅ 演示成功完成，生成 {len(generated_files)} 个文件")
        return True, {
            'execution_time': exec_time,
            'generated_files': generated_files,
            'output_length': len(output)
        }
    else:
        print(f"   ❌ 演示失败")
        return False, {'error': output}

def test_file_structure():
    """测试YICA文件结构"""
    print("\n🗂️  测试YICA文件结构")
    print("=" * 50)
    
    expected_files = [
        'mirage/python/mirage/yica_optimizer.py',
        'mirage/include/mirage/triton_transpiler/runtime/yica_runtime.py',
        'mirage/cmake/yica.cmake',
        'mirage/include/mirage/yica/config.h',
        'mirage/include/mirage/yica/optimizer.h',
        'YICA-MIRAGE-INTEGRATION-PLAN.md',
        'demo_yica_optimization.py',
        'demo_yica_standalone.py'
    ]
    
    results = {
        'total_files': len(expected_files),
        'existing_files': 0,
        'missing_files': [],
        'file_sizes': {}
    }
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            results['existing_files'] += 1
            results['file_sizes'][file_path] = size
            print(f"   ✅ {file_path} ({size} bytes)")
        else:
            results['missing_files'].append(file_path)
            print(f"   ❌ {file_path} (缺失)")
    
    success_rate = results['existing_files'] / results['total_files']
    print(f"\n   📊 文件完整性: {results['existing_files']}/{results['total_files']} ({success_rate:.1%})")
    
    return success_rate >= 0.8, results

def test_cmake_integration():
    """测试CMake集成"""
    print("\n🔧 测试CMake集成")
    print("=" * 50)
    
    # 检查CMakeLists.txt是否包含YICA支持
    cmake_file = 'mirage/CMakeLists.txt'
    if os.path.exists(cmake_file):
        with open(cmake_file, 'r') as f:
            content = f.read()
            if 'yica.cmake' in content:
                print("   ✅ CMakeLists.txt包含YICA支持")
                yica_integration = True
            else:
                print("   ❌ CMakeLists.txt缺少YICA支持")
                yica_integration = False
    else:
        print("   ❌ CMakeLists.txt不存在")
        yica_integration = False
    
    # 检查yica.cmake文件
    yica_cmake = 'mirage/cmake/yica.cmake'
    if os.path.exists(yica_cmake):
        size = os.path.getsize(yica_cmake)
        print(f"   ✅ yica.cmake存在 ({size} bytes)")
        cmake_file_exists = True
    else:
        print("   ❌ yica.cmake不存在")
        cmake_file_exists = False
    
    overall_success = yica_integration and cmake_file_exists
    print(f"   📊 CMake集成状态: {'✅ 成功' if overall_success else '❌ 失败'}")
    
    return overall_success, {
        'yica_integration': yica_integration,
        'cmake_file_exists': cmake_file_exists
    }

def generate_test_report(results):
    """生成测试报告"""
    print("\n📋 YICA集成测试报告")
    print("=" * 60)
    
    # 计算总体成功率
    total_tests = len(results)
    successful_tests = sum(1 for result in results.values() if result['success'])
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"📊 总体测试结果: {successful_tests}/{total_tests} ({success_rate:.1%})")
    print()
    
    # 详细结果
    for test_name, result in results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(f"🔹 {test_name}: {status}")
        if 'details' in result:
            for key, value in result['details'].items():
                print(f"   - {key}: {value}")
        print()
    
    # 保存报告到JSON
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_success_rate': success_rate,
        'successful_tests': successful_tests,
        'total_tests': total_tests,
        'test_results': results
    }
    
    with open('yica_integration_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 详细报告已保存: yica_integration_report.json")
    
    # 总结建议
    print(f"\n🎯 测试总结:")
    if success_rate >= 0.8:
        print(f"   ✅ YICA集成基本成功 ({success_rate:.1%})")
        print(f"   🚀 可以进入下一阶段的开发和优化")
    elif success_rate >= 0.5:
        print(f"   ⚠️  YICA集成部分成功 ({success_rate:.1%})")
        print(f"   🔧 需要修复一些问题后再继续")
    else:
        print(f"   ❌ YICA集成失败 ({success_rate:.1%})")
        print(f"   🛠️  需要重新检查集成过程")
    
    return success_rate

def main():
    """主测试流程"""
    print("YICA-Mirage集成验证测试")
    print("目标: 验证YICA优化器的基本功能和集成状态")
    print("=" * 60)
    
    # 运行各项测试
    test_results = {}
    
    # 1. 测试文件结构
    success, details = test_file_structure()
    test_results['文件结构完整性'] = {
        'success': success,
        'details': details
    }
    
    # 2. 测试CMake集成
    success, details = test_cmake_integration()
    test_results['CMake集成'] = {
        'success': success,
        'details': details
    }
    
    # 3. 测试独立演示程序
    success, details = test_yica_standalone()
    test_results['YICA演示程序'] = {
        'success': success,
        'details': details
    }
    
    # 生成最终报告
    success_rate = generate_test_report(test_results)
    
    # 返回状态码
    return 0 if success_rate >= 0.7 else 1

if __name__ == "__main__":
    sys.exit(main()) 