#!/usr/bin/env python3
"""
完整环境测试脚本
测试所有YICA-Mirage集成所需的依赖项和组件
"""

import sys
import os
import subprocess
import importlib
from typing import Dict, List, Tuple, Any

def test_result(test_name: str, success: bool, details: str = "") -> Dict[str, Any]:
    """格式化测试结果"""
    status = "✅ 通过" if success else "❌ 失败"
    print(f"{status} {test_name}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"   {line}")
    return {
        'name': test_name,
        'success': success,
        'details': details
    }

def test_python_environment() -> List[Dict[str, Any]]:
    """测试Python环境"""
    print("\n🐍 Python环境测试")
    print("=" * 50)
    
    results = []
    
    # Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    details = f"版本: {python_version}\n路径: {sys.executable}"
    results.append(test_result("Python版本", True, details))
    
    return results

def test_core_dependencies() -> List[Dict[str, Any]]:
    """测试核心依赖项"""
    print("\n📦 核心依赖项测试")
    print("=" * 50)
    
    results = []
    dependencies = [
        ('numpy', '数值计算库'),
        ('z3', 'Z3定理证明器'),
        ('json', 'JSON处理'),
        ('typing', '类型注解'),
        ('subprocess', '进程管理'),
        ('ctypes', '动态库调用'),
        ('time', '时间处理'),
        ('os', '操作系统接口'),
        ('sys', '系统接口')
    ]
    
    for module_name, description in dependencies:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'N/A')
            if hasattr(module, 'get_version_string'):
                version = module.get_version_string()
            details = f"{description}\n版本: {version}"
            results.append(test_result(f"{module_name}模块", True, details))
        except ImportError as e:
            details = f"{description}\n错误: {e}"
            results.append(test_result(f"{module_name}模块", False, details))
    
    return results

def test_z3_functionality() -> List[Dict[str, Any]]:
    """专门测试Z3功能"""
    print("\n🔬 Z3功能测试")
    print("=" * 50)
    
    results = []
    
    try:
        import z3
        
        # 基本Z3测试
        x = z3.Int('x')
        solver = z3.Solver()
        solver.add(x > 0)
        solver.add(x < 10)
        check_result = solver.check()
        
        if check_result == z3.sat:
            model = solver.model()
            solution = model[x].as_long()
            details = f"Z3版本: {z3.get_version_string()}\n求解器测试: 成功\n示例解: x = {solution}"
            results.append(test_result("Z3求解器功能", True, details))
        else:
            details = f"Z3版本: {z3.get_version_string()}\n求解器测试: 失败 ({check_result})"
            results.append(test_result("Z3求解器功能", False, details))
            
    except Exception as e:
        details = f"Z3测试失败: {e}"
        results.append(test_result("Z3求解器功能", False, details))
    
    return results

def test_file_system() -> List[Dict[str, Any]]:
    """测试文件系统和目录结构"""
    print("\n📁 文件系统测试")
    print("=" * 50)
    
    results = []
    
    # 检查必要的目录
    required_dirs = [
        'mirage',
        'mirage/python',
        'mirage/python/mirage',
        'mirage/include',
        'mirage/cmake'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files_count = len(os.listdir(dir_path))
            details = f"路径: {os.path.abspath(dir_path)}\n文件数量: {files_count}"
            results.append(test_result(f"{dir_path}目录", True, details))
        else:
            details = f"路径: {os.path.abspath(dir_path) if os.path.exists(dir_path) else '不存在'}"
            results.append(test_result(f"{dir_path}目录", False, details))
    
    return results

def test_yica_components() -> List[Dict[str, Any]]:
    """测试YICA组件"""
    print("\n🔧 YICA组件测试")
    print("=" * 50)
    
    results = []
    
    # 检查YICA关键文件
    yica_files = [
        'mirage/python/mirage/yica_optimizer.py',
        'mirage/include/mirage/triton_transpiler/runtime/yica_runtime.py',
        'mirage/cmake/yica.cmake',
        'mirage/include/mirage/yica/config.h',
        'mirage/include/mirage/yica/optimizer.h',
        'YICA-MIRAGE-INTEGRATION-PLAN.md',
        'demo_yica_standalone.py'
    ]
    
    for file_path in yica_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            details = f"路径: {os.path.abspath(file_path)}\n文件大小: {size} bytes"
            results.append(test_result(f"{os.path.basename(file_path)}", True, details))
        else:
            details = f"路径: {os.path.abspath(file_path) if os.path.exists(file_path) else '文件不存在'}"
            results.append(test_result(f"{os.path.basename(file_path)}", False, details))
    
    return results

def test_yica_demo() -> List[Dict[str, Any]]:
    """测试YICA演示程序"""
    print("\n🚀 YICA演示程序测试")
    print("=" * 50)
    
    results = []
    
    try:
        # 运行YICA独立演示
        result = subprocess.run(
            [sys.executable, 'demo_yica_standalone.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output_lines = result.stdout.count('\n')
            performance_lines = [line for line in result.stdout.split('\n') if '加速比:' in line]
            speedups = []
            for line in performance_lines:
                if '加速比:' in line:
                    try:
                        speedup = float(line.split('加速比:')[1].split('x')[0].strip())
                        speedups.append(speedup)
                    except:
                        pass
            
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0
            details = f"执行成功\n输出行数: {output_lines}\n平均加速比: {avg_speedup:.1f}x\n检测到 {len(speedups)} 个性能测试"
            results.append(test_result("YICA演示程序执行", True, details))
            
            # 检查生成的文件
            generated_files = ['yica_matmul_kernel.py', 'yica_attention_kernel.py']
            for gen_file in generated_files:
                if os.path.exists(gen_file):
                    size = os.path.getsize(gen_file)
                    details = f"文件大小: {size} bytes"
                    results.append(test_result(f"生成文件 {gen_file}", True, details))
                else:
                    results.append(test_result(f"生成文件 {gen_file}", False, "文件未生成"))
                    
        else:
            details = f"执行失败\n返回码: {result.returncode}\n错误: {result.stderr[:200]}"
            results.append(test_result("YICA演示程序执行", False, details))
            
    except subprocess.TimeoutExpired:
        results.append(test_result("YICA演示程序执行", False, "执行超时 (>30s)"))
    except Exception as e:
        results.append(test_result("YICA演示程序执行", False, f"执行异常: {e}"))
    
    return results

def test_cmake_integration() -> List[Dict[str, Any]]:
    """测试CMake集成"""
    print("\n🔨 CMake集成测试")
    print("=" * 50)
    
    results = []
    
    # 检查CMake文件
    cmake_main = 'mirage/CMakeLists.txt'
    if os.path.exists(cmake_main):
        with open(cmake_main, 'r') as f:
            content = f.read()
            has_yica = 'yica.cmake' in content
            details = f"文件大小: {len(content)} bytes\nYICA集成: {'是' if has_yica else '否'}"
            results.append(test_result("主CMakeLists.txt", has_yica, details))
    else:
        results.append(test_result("主CMakeLists.txt", False, "文件不存在"))
    
    # 检查YICA CMake文件
    yica_cmake = 'mirage/cmake/yica.cmake'
    if os.path.exists(yica_cmake):
        size = os.path.getsize(yica_cmake)
        with open(yica_cmake, 'r') as f:
            content = f.read()
            has_enable_option = 'ENABLE_YICA' in content
            details = f"文件大小: {size} bytes\n配置选项: {'完整' if has_enable_option else '不完整'}"
            results.append(test_result("YICA CMake配置", has_enable_option, details))
    else:
        results.append(test_result("YICA CMake配置", False, "文件不存在"))
    
    return results

def test_build_tools() -> List[Dict[str, Any]]:
    """测试构建工具"""
    print("\n🛠️  构建工具测试")
    print("=" * 50)
    
    results = []
    
    # 测试cmake
    try:
        result = subprocess.run(['cmake', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            details = f"版本信息: {version_line}"
            results.append(test_result("CMake", True, details))
        else:
            results.append(test_result("CMake", False, "无法获取版本信息"))
    except FileNotFoundError:
        results.append(test_result("CMake", False, "未安装"))
    
    # 测试make
    try:
        result = subprocess.run(['make', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            details = f"版本信息: {version_line}"
            results.append(test_result("Make", True, details))
        else:
            results.append(test_result("Make", False, "无法获取版本信息"))
    except FileNotFoundError:
        results.append(test_result("Make", False, "未安装"))
    
    return results

def generate_environment_report(all_results: List[List[Dict[str, Any]]]) -> None:
    """生成环境测试报告"""
    print("\n📋 环境测试报告")
    print("=" * 60)
    
    # 统计总体结果
    total_tests = sum(len(results) for results in all_results)
    successful_tests = sum(
        sum(1 for result in results if result['success']) 
        for results in all_results
    )
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"📊 总体测试结果: {successful_tests}/{total_tests} ({success_rate:.1%})")
    print()
    
    # 按类别统计
    categories = [
        "Python环境", "核心依赖项", "Z3功能", "文件系统", 
        "YICA组件", "YICA演示程序", "CMake集成", "构建工具"
    ]
    
    for i, (category, results) in enumerate(zip(categories, all_results)):
        if results:
            category_success = sum(1 for result in results if result['success'])
            category_total = len(results)
            category_rate = category_success / category_total
            status = "✅" if category_rate >= 0.8 else "⚠️" if category_rate >= 0.5 else "❌"
            print(f"{status} {category}: {category_success}/{category_total} ({category_rate:.1%})")
    
    print()
    
    # 环境评估
    if success_rate >= 0.9:
        print("🎉 环境状态: 优秀 - 所有组件基本就绪")
        print("✅ 可以正常进行YICA-Mirage开发工作")
    elif success_rate >= 0.8:
        print("✅ 环境状态: 良好 - 主要组件工作正常")
        print("⚠️ 部分次要问题需要关注")
    elif success_rate >= 0.6:
        print("⚠️ 环境状态: 可用 - 基本功能可用")
        print("🔧 建议修复一些问题以获得更好体验")
    else:
        print("❌ 环境状态: 需要修复 - 存在严重问题")
        print("🛠️ 需要解决关键问题才能正常使用")
    
    # 保存详细报告
    report = {
        'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
        'success_rate': success_rate,
        'successful_tests': successful_tests,
        'total_tests': total_tests,
        'categories': {}
    }
    
    for category, results in zip(categories, all_results):
        report['categories'][category] = results
    
    import json
    with open('environment_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细报告已保存: environment_test_report.json")

def main():
    """主测试函数"""
    print("🧪 YICA-Mirage完整环境测试")
    print("目标: 验证所有依赖项和组件的完整性")
    print("=" * 60)
    
    # 运行所有测试
    all_results = [
        test_python_environment(),
        test_core_dependencies(),
        test_z3_functionality(),
        test_file_system(),
        test_yica_components(),
        test_yica_demo(),
        test_cmake_integration(),
        test_build_tools()
    ]
    
    # 生成报告
    generate_environment_report(all_results)
    
    # 计算总体成功率
    total_tests = sum(len(results) for results in all_results)
    successful_tests = sum(
        sum(1 for result in results if result['success']) 
        for results in all_results
    )
    success_rate = successful_tests / total_tests if total_tests > 0 else 0
    
    print(f"\n🎯 环境测试完成")
    print(f"总体成功率: {success_rate:.1%}")
    
    return 0 if success_rate >= 0.8 else 1

if __name__ == "__main__":
    sys.exit(main()) 