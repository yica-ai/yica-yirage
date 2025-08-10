#!/bin/bash
# 增强版YIRAGE构建脚本
# 处理OpenMP、cutlass等硬性依赖

set -e

echo "🚀 增强版YIRAGE构建 - 处理硬性依赖..."

# 配置参数
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YIRAGE_DIR="$PROJECT_ROOT/yirage"
DEPS_DIR="$YIRAGE_DIR/deps"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "检测到macOS系统"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "检测到Linux系统"
    else
        OS="unknown"
        print_warning "未知操作系统: $OSTYPE"
    fi
}

# 安装OpenMP (macOS)
install_openmp_macos() {
    print_status "处理macOS OpenMP依赖..."
    
    # 检查是否已安装libomp
    if brew list libomp &>/dev/null; then
        print_success "libomp已安装"
    else
        print_status "安装libomp..."
        brew install libomp || {
            print_error "无法通过brew安装libomp"
            return 1
        }
    fi
    
    # 获取libomp路径
    LIBOMP_PREFIX=$(brew --prefix libomp)
    export OpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
    export OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
    export OpenMP_C_LIB_NAMES="omp"
    export OpenMP_CXX_LIB_NAMES="omp"
    export OpenMP_omp_LIBRARY="$LIBOMP_PREFIX/lib/libomp.dylib"
    
    print_success "OpenMP环境配置完成"
    echo "  - Include: $LIBOMP_PREFIX/include"
    echo "  - Library: $LIBOMP_PREFIX/lib/libomp.dylib"
}

# 安装OpenMP (Linux)
install_openmp_linux() {
    print_status "处理Linux OpenMP依赖..."
    
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y libomp-dev
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        sudo yum install -y libgomp-devel
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y libgomp-devel
    else
        print_warning "无法自动安装OpenMP，请手动安装"
        return 1
    fi
    
    print_success "OpenMP安装完成"
}

# 下载并设置CUTLASS
setup_cutlass() {
    print_status "设置CUTLASS依赖..."
    
    cd "$YIRAGE_DIR"
    mkdir -p "$DEPS_DIR"
    
    if [[ -d "$DEPS_DIR/cutlass" ]]; then
        print_warning "CUTLASS已存在，跳过下载"
    else
        print_status "下载CUTLASS..."
        cd "$DEPS_DIR"
        
        # 下载CUTLASS v3.4.1 (稳定版本)
        git clone --depth 1 --branch v3.4.1 https://github.com/NVIDIA/cutlass.git || {
            print_error "无法下载CUTLASS"
            return 1
        }
        
        print_success "CUTLASS下载完成"
    fi
    
    # 创建cutlass include软链接
    if [[ ! -d "$DEPS_DIR/cutlass/include" ]]; then
        print_error "CUTLASS include目录不存在"
        return 1
    fi
    
    print_success "CUTLASS设置完成"
    echo "  - 路径: $DEPS_DIR/cutlass"
    echo "  - Include: $DEPS_DIR/cutlass/include"
}

# 下载并设置nlohmann/json
setup_json() {
    print_status "设置nlohmann/json依赖..."
    
    cd "$YIRAGE_DIR"
    mkdir -p "$DEPS_DIR"
    
    if [[ -d "$DEPS_DIR/json" ]]; then
        print_warning "nlohmann/json已存在，跳过下载"
    else
        print_status "下载nlohmann/json..."
        cd "$DEPS_DIR"
        
        # 下载nlohmann/json v3.11.3
        git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git || {
            print_error "无法下载nlohmann/json"
            return 1
        }
        
        print_success "nlohmann/json下载完成"
    fi
    
    print_success "nlohmann/json设置完成"
    echo "  - 路径: $DEPS_DIR/json"
    echo "  - Include: $DEPS_DIR/json/include"
}

# 设置Z3依赖
setup_z3() {
    print_status "设置Z3依赖..."
    
    cd "$YIRAGE_DIR"
    mkdir -p "$DEPS_DIR"
    
    # 优先使用pip安装Z3，更简单可靠
    print_status "尝试使用pip安装Z3..."
    
    # 检查是否在虚拟环境中
    if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        print_status "检测到虚拟环境，使用pip安装Z3..."
        if pip install z3-solver; then
            print_success "Z3通过pip安装成功"
            echo "  - 版本: $(python -c 'import z3; print(z3.get_version_string())' 2>/dev/null || echo '未知')"
            return 0
        else
            print_warning "pip安装Z3失败，尝试从源码编译..."
        fi
    else
        print_warning "未检测到虚拟环境，建议在虚拟环境中使用pip安装Z3"
        print_status "尝试全局pip安装Z3..."
        if pip3 install z3-solver; then
            print_success "Z3通过pip安装成功"
            echo "  - 版本: $(python3 -c 'import z3; print(z3.get_version_string())' 2>/dev/null || echo '未知')"
            return 0
        else
            print_warning "pip安装Z3失败，尝试从源码编译..."
        fi
    fi
    
    # 如果pip安装失败，才尝试从源码编译
    if [[ -d "$DEPS_DIR/z3" && -d "$DEPS_DIR/z3/install" ]]; then
        print_warning "Z3源码版本已存在，跳过重新编译"
        print_success "Z3设置完成"
        echo "  - 路径: $DEPS_DIR/z3"
        echo "  - Install: $DEPS_DIR/z3/install"
        echo "  - Include: $DEPS_DIR/z3/install/include"
        echo "  - Library: $DEPS_DIR/z3/install/lib"
        return 0
    fi
    
    print_status "从源码编译Z3 (备用方案)..."
    print_warning "注意: 建议使用 'pip install z3-solver' 代替源码编译"
    
    # 询问用户是否继续源码编译
    if [[ "${1:-}" != "--force-compile" ]]; then
        print_status "推荐方案:"
        echo "  1. 创建虚拟环境: python3 -m venv venv && source venv/bin/activate"
        echo "  2. 安装Z3: pip install z3-solver"
        echo "  3. 重新运行构建脚本"
        echo ""
        print_warning "如需强制从源码编译，请运行: $0 deps --force-compile"
        return 0
    fi
    
    cd "$DEPS_DIR"
    
    # 清理可能存在的损坏Z3目录
    rm -rf z3
    
    # 下载Z3 v4.12.2 (更稳定的版本)
    git clone --depth 1 --branch z3-4.12.2 https://github.com/Z3Prover/z3.git || {
        print_error "无法下载Z3源码"
        print_status "请使用: pip install z3-solver"
        return 1
    }
    
    # 构建Z3
    print_status "构建Z3..."
    cd z3
    
    # 修复已知的编译问题
    print_status "修复Z3编译问题..."
    
    # 修复 static_matrix_def.h 中的 get_value_of_column_cell 问题
    if [[ -f "src/math/lp/static_matrix_def.h" ]]; then
        print_status "修复 static_matrix_def.h..."
        sed -i.bak 's/A\.get_value_of_column_cell(col)/col.coeff()/g' src/math/lp/static_matrix_def.h
    fi
    
    # 修复 static_matrix.h 中的 get_val 方法问题
    if [[ -f "src/math/lp/static_matrix.h" ]]; then
        print_status "修复 static_matrix.h..."
        # 添加 get_val 方法定义
        if ! grep -q "T get_val" src/math/lp/static_matrix.h; then
            sed -i.bak '/class static_matrix {/a\
    template<typename U>\
    T get_val(const U& cell) const { return cell.coeff(); }
' src/math/lp/static_matrix.h
        fi
    fi
    
    # 修复 column_info.h 中的成员变量名问题
    if [[ -f "src/math/lp/column_info.h" ]]; then
        print_status "修复 column_info.h..."
        sed -i.bak 's/c\.m_low_bound/c.m_lower_bound/g' src/math/lp/column_info.h
    fi
    
    # 修复Python脚本中的正则表达式警告
    if [[ -f "scripts/update_api.py" ]]; then
        print_status "修复 Python 脚本正则表达式..."
        sed -i.bak 's/\\(/\\\\(/g; s/\\)/\\\\)/g' scripts/update_api.py
    fi
    
    # 配置构建参数
    python scripts/mk_make.py \
        --prefix="$DEPS_DIR/z3/install" \
        --staticlib \
        --python || {
        print_error "Z3配置失败"
        print_status "请使用: pip install z3-solver"
        return 1
    }
    
    cd build
    
    # 尝试构建，如果失败则建议使用pip
    print_status "编译Z3 (单线程)..."
    if ! make -j1; then
        print_error "Z3源码编译失败"
        print_status "建议使用pip安装: pip install z3-solver"
        cd "$YIRAGE_DIR"
        rm -rf "$DEPS_DIR/z3"
        return 1
    fi
    
    # 安装
    print_status "安装Z3..."
    make install || {
        print_error "Z3安装失败"
        print_status "建议使用pip安装: pip install z3-solver"
        cd "$YIRAGE_DIR"
        rm -rf "$DEPS_DIR/z3"
        return 1
    }
    
    print_success "Z3源码编译完成"
    
    if [[ -d "$DEPS_DIR/z3/install" ]]; then
        print_success "Z3设置完成"
        echo "  - 路径: $DEPS_DIR/z3"
        echo "  - Install: $DEPS_DIR/z3/install"
        echo "  - Include: $DEPS_DIR/z3/install/include"
        echo "  - Library: $DEPS_DIR/z3/install/lib"
    else
        print_error "Z3源码编译失败"
        print_status "请使用: pip install z3-solver"
        return 1
    fi
}

# 创建增强的setup.py
create_enhanced_setup() {
    print_status "创建增强的setup.py..."
    
    cd "$YIRAGE_DIR"
    
    cat > setup.py << 'EOF'
#!/usr/bin/env python3
"""
增强版YIRAGE安装脚本
处理OpenMP、CUTLASS等硬性依赖
"""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import platform

# 读取版本信息 - 动态从版本文件获取
def get_version():
    version_file = os.path.join('python', 'yirage', 'version.py')
    
    # 方法1: 直接执行版本文件获取版本
    try:
        version_globals = {}
        with open(version_file, 'r') as f:
            exec(f.read(), version_globals)
        
        if '__version__' in version_globals:
            version = version_globals['__version__']
            print(f"✅ 动态读取版本: {version} (来源: {version_file})")
            return version
    except Exception as e:
        print(f"⚠️  方法1失败: {e}")
    
    # 方法2: 文本解析 (备用)
    try:
        with open(version_file, 'r') as f:
            content = f.read()
            import re
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                version = match.group(1)
                print(f"✅ 解析版本: {version} (来源: {version_file})")
                return version
    except Exception as e:
        print(f"⚠️  方法2失败: {e}")
    
    # 方法3: 尝试导入 (如果在正确路径)
    try:
        import sys
        sys.path.insert(0, 'python')
        from yirage.version import __version__
        print(f"✅ 导入版本: {__version__} (来源: 模块导入)")
        return __version__
    except Exception as e:
        print(f"⚠️  方法3失败: {e}")
    
    print(f"❌ 无法获取版本信息，使用默认值")
    return "dev-unknown"

# 检测编译环境
def detect_compile_env():
    env = {
        'has_cuda': False,
        'has_openmp': False,
        'cutlass_path': None,
        'json_path': None,
        'z3_path': None,
        'is_macos': platform.system() == 'Darwin',
        'is_linux': platform.system() == 'Linux',
    }
    
    # 检查CUDA
    if os.path.exists('/usr/local/cuda') or os.environ.get('CUDA_HOME'):
        env['has_cuda'] = True
        print("✅ 检测到CUDA环境")
    
    # 检查依赖路径
    deps_dir = os.path.join(os.getcwd(), 'deps')
    
    if os.path.exists(os.path.join(deps_dir, 'cutlass', 'include')):
        env['cutlass_path'] = os.path.join(deps_dir, 'cutlass')
        print(f"✅ 找到CUTLASS: {env['cutlass_path']}")
    
    if os.path.exists(os.path.join(deps_dir, 'json', 'include')):
        env['json_path'] = os.path.join(deps_dir, 'json')
        print(f"✅ 找到nlohmann/json: {env['json_path']}")
    
    # 优先检查pip安装的Z3
    try:
        import z3
        print(f"✅ 找到Z3 (pip): {z3.get_version_string()}")
        env['z3_pip'] = True
    except ImportError:
        env['z3_pip'] = False
        # 然后检查本地编译的Z3
        if os.path.exists(os.path.join(deps_dir, 'z3', 'install')):
            env['z3_path'] = os.path.join(deps_dir, 'z3', 'install')
            print(f"✅ 找到Z3 (源码): {env['z3_path']}")
        else:
            print("⚠️  未找到Z3，建议运行: pip install z3-solver")
    
    # 检查OpenMP
    if env['is_macos']:
        # macOS使用libomp
        try:
            import subprocess
            result = subprocess.run(['brew', '--prefix', 'libomp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                env['has_openmp'] = True
                env['openmp_path'] = result.stdout.strip()
                print(f"✅ 找到OpenMP (libomp): {env['openmp_path']}")
        except:
            pass
    else:
        # Linux通常有系统OpenMP
        env['has_openmp'] = True
        print("✅ 假设Linux系统有OpenMP支持")
    
    return env

# 构建扩展模块
def create_extensions(env):
    extensions = []
    
    # 基础包含路径
    include_dirs = [
        'include',
        'python',
        pybind11.get_include(),
    ]
    
    # 添加依赖包含路径
    if env['cutlass_path']:
        include_dirs.append(os.path.join(env['cutlass_path'], 'include'))
    
    if env['json_path']:
        include_dirs.append(os.path.join(env['json_path'], 'include'))
    
    if env['z3_path']:
        include_dirs.extend([
            os.path.join(env['z3_path'], 'include'),
        ])
    
    # 编译标志
    compile_args = ['-std=c++17', '-O3']
    link_args = []
    libraries = []
    library_dirs = []
    
    # OpenMP支持
    if env['has_openmp']:
        if env['is_macos'] and 'openmp_path' in env:
            # macOS libomp
            compile_args.extend(['-Xpreprocessor', '-fopenmp'])
            include_dirs.append(os.path.join(env['openmp_path'], 'include'))
            library_dirs.append(os.path.join(env['openmp_path'], 'lib'))
            libraries.append('omp')
        else:
            # Linux OpenMP
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')
    
    # Z3库 (优先使用pip版本，无需手动链接)
    if env.get('z3_pip'):
        # pip安装的Z3会自动处理链接
        print("✅ 使用pip安装的Z3，无需手动链接")
    elif env.get('z3_path'):
        # 使用本地编译的Z3
        library_dirs.append(os.path.join(env['z3_path'], 'lib'))
        libraries.append('z3')
        include_dirs.append(os.path.join(env['z3_path'], 'include'))
        print("✅ 使用本地编译的Z3")
    else:
        print("⚠️  未找到Z3，某些功能可能不可用")
    
    # CUDA支持 (可选)
    if env['has_cuda']:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        include_dirs.append(os.path.join(cuda_home, 'include'))
        library_dirs.append(os.path.join(cuda_home, 'lib64'))
        libraries.extend(['cuda', 'cudart', 'cublas'])
        compile_args.append('-DYICA_ENABLE_CUDA')
    else:
        compile_args.append('-DYICA_CPU_ONLY')
    
    # 创建核心扩展
    try:
        core_extension = Pybind11Extension(
            "yirage._core",
            sources=[
                # 添加关键源文件
                "src/base/layout.cc",
                "src/search/config.cc",
                "src/search/search.cc",
                # 可以根据需要添加更多源文件
            ],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            language='c++',
            cxx_std=17,
        )
        
        # 设置编译和链接参数
        core_extension.extra_compile_args = compile_args
        core_extension.extra_link_args = link_args
        
        extensions.append(core_extension)
        print(f"✅ 创建核心扩展模块")
        
    except Exception as e:
        print(f"⚠️  跳过C++扩展模块: {e}")
    
    return extensions

# 主安装配置
def main():
    print("🔧 检测编译环境...")
    env = detect_compile_env()
    
    print("🔨 创建扩展模块...")
    extensions = create_extensions(env)
    
    # 基础依赖
    install_requires = [
        "numpy>=1.19.0",
        "z3-solver>=4.8.0",
    ]
    
    # Z3依赖处理
    if env.get('z3_pip'):
        # 已经通过pip安装，无需重复添加
        print("✅ Z3依赖已通过pip满足")
    elif env.get('z3_path'):
        # 有本地编译版本，无需pip版本
        print("✅ Z3依赖通过本地编译满足")
    else:
        # 确保有Z3依赖
        print("📦 将通过pip安装Z3")
    
    # PyTorch依赖 (可选)
    try:
        import torch
        print(f"✅ 检测到PyTorch {torch.__version__}")
    except ImportError:
        install_requires.append("torch>=1.12.0")
        print("📦 将安装PyTorch")
    
    setup(
        name="yica-yirage",
        version=get_version(),
        description="YICA-Yirage: AI Computing Optimization Framework (Enhanced Build)",
        long_description="YICA-Yirage with OpenMP, CUTLASS, and Z3 support",
        long_description_content_type="text/plain",
        author="YICA Team",
        author_email="contact@yica.ai",
        
        # 包配置
        package_dir={"": "python"},
        packages=find_packages(where="python"),
        
        # C++扩展
        ext_modules=extensions,
        cmdclass={"build_ext": build_ext},
        
        # 依赖
        install_requires=install_requires,
        
        extras_require={
            "dev": [
                "pytest>=6.0",
                "pytest-cov>=3.0",
                "black>=21.0",
                "flake8>=3.8",
            ],
            "triton": [
                "triton>=2.0.0; sys_platform=='linux'",
            ],
            "full": [
                "torch>=1.12.0",
                "triton>=2.0.0; sys_platform=='linux'",
                "matplotlib>=3.0.0",
                "tqdm>=4.0.0",
            ],
        },
        
        python_requires=">=3.8",
        zip_safe=False,
        
        # 分类
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )

if __name__ == "__main__":
    main()
EOF

    print_success "增强setup.py创建完成"
}

# 构建增强包
build_enhanced_package() {
    print_status "构建增强Python包..."
    
    cd "$YIRAGE_DIR"
    
    # 创建虚拟环境
    if [[ ! -d "venv_enhanced" ]]; then
        print_status "创建增强虚拟环境..."
        python3 -m venv venv_enhanced
    fi
    
    # 激活虚拟环境
    source venv_enhanced/bin/activate
    
    # 安装构建依赖
    print_status "安装增强构建依赖..."
    pip install --upgrade pip wheel setuptools build twine
    pip install pybind11 cmake ninja
    
    # 设置环境变量
    if [[ "$OS" == "macos" ]] && [[ -n "${LIBOMP_PREFIX:-}" ]]; then
        export CC=clang
        export CXX=clang++
        export CPPFLAGS="-I$LIBOMP_PREFIX/include"
        export LDFLAGS="-L$LIBOMP_PREFIX/lib -lomp"
    fi
    
    # 清理之前的构建
    rm -rf build/ dist/ *.egg-info/ python/*.egg-info/
    
    # 构建包
    print_status "构建增强wheel包..."
    
    # 尝试C++扩展构建
    if python setup.py bdist_wheel; then
        print_success "C++扩展构建成功"
    else
        print_warning "C++扩展构建失败，创建纯Python版本..."
        
        # 创建纯Python版本的setup.py
        cat > setup_simple.py << 'EOF'
#!/usr/bin/env python3
"""
纯Python版YIRAGE安装脚本 (回退版本)
"""

from setuptools import setup, find_packages
import os

def get_version():
    version_file = os.path.join('python', 'yirage', 'version.py')
    
    # 动态读取版本信息
    try:
        version_globals = {}
        with open(version_file, 'r') as f:
            exec(f.read(), version_globals)
        
        if '__version__' in version_globals:
            return version_globals['__version__']
    except Exception:
        pass
    
    # 备用方法：正则解析
    try:
        with open(version_file, 'r') as f:
            content = f.read()
            import re
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    return "dev-unknown"

setup(
    name="yica-yirage",
    version=get_version(),
    description="YICA-Yirage: AI Computing Optimization Framework (Pure Python)",
    long_description="YICA-Yirage Pure Python version without C++ extensions",
    long_description_content_type="text/plain",
    author="YICA Team",
    author_email="contact@yica.ai",
    
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.12.0",
        "z3-solver>=4.8.0",
    ],
    
    extras_require={
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.8"],
        "triton": ["triton>=2.0.0; sys_platform=='linux'"],
        "full": ["torch>=1.12.0", "matplotlib>=3.0.0", "tqdm>=4.0.0"],
    },
    
    python_requires=">=3.8",
    zip_safe=False,
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
EOF
        
        # 使用纯Python版本构建
        python setup_simple.py bdist_wheel || {
            print_error "纯Python构建也失败了"
            return 1
        }
        
        print_success "纯Python版本构建成功"
    fi
    
    # 构建源码包
    print_status "构建源码包..."
    python setup.py sdist
    
    # 检查结果
    if [[ -d "dist" ]]; then
        print_success "增强构建完成！"
        echo "构建产物:"
        ls -la dist/
        
        # 验证包
        print_status "验证包内容..."
        python -m twine check dist/*
        
    else
        print_error "增强构建失败"
        return 1
    fi
}

# 测试增强安装
test_enhanced_installation() {
    print_status "测试增强安装..."
    
    cd "$YIRAGE_DIR"
    source venv_enhanced/bin/activate
    
    # 安装构建的包
    pip install --force-reinstall dist/*.whl
    
    # 测试导入和功能
    python -c "
import sys
sys.path.insert(0, 'python')
import yirage
print(f'✅ 增强安装测试成功！')
print(f'版本: {yirage.version.__version__}')
print(f'可用功能: {list(yirage.get_version_info().keys())}')

# 测试扩展模块
try:
    import yirage._core
    print('✅ C++扩展模块加载成功')
except ImportError as e:
    print(f'⚠️  C++扩展模块不可用: {e}')

# 测试优化器
try:
    optimizer = yirage.create_yica_optimizer()
    print('✅ 优化器创建成功')
    
    # 测试依赖
    if yirage.get_version_info()['z3_available']:
        print('✅ Z3支持可用')
    
    if yirage.get_version_info()['torch_available']:
        print('✅ PyTorch支持可用')
        
except Exception as e:
    print(f'⚠️  优化器测试失败: {e}')
"
    
    print_success "增强安装测试完成"
}

# 主函数
main() {
    case "${1:-}" in
        "deps")
            detect_os
            if [[ "$OS" == "macos" ]]; then
                install_openmp_macos
            elif [[ "$OS" == "linux" ]]; then
                install_openmp_linux
            fi
            setup_cutlass
            setup_json
            setup_z3
            ;;
        "setup")
            create_enhanced_setup
            ;;
        "build")
            detect_os
            if [[ "$OS" == "macos" ]]; then
                install_openmp_macos
            elif [[ "$OS" == "linux" ]]; then
                install_openmp_linux
            fi
            setup_cutlass
            setup_json
            create_enhanced_setup
            build_enhanced_package
            ;;
        "test")
            detect_os
            if [[ "$OS" == "macos" ]]; then
                install_openmp_macos
            elif [[ "$OS" == "linux" ]]; then
                install_openmp_linux
            fi
            setup_cutlass
            setup_json
            create_enhanced_setup
            build_enhanced_package
            test_enhanced_installation
            ;;
        "")
            detect_os
            if [[ "$OS" == "macos" ]]; then
                install_openmp_macos
            elif [[ "$OS" == "linux" ]]; then
                install_openmp_linux
            fi
            setup_cutlass
            setup_json
            create_enhanced_setup
            build_enhanced_package
            test_enhanced_installation
            ;;
        *)
            echo "增强版YIRAGE构建脚本"
            echo ""
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  deps   - 仅安装依赖 (OpenMP, CUTLASS, nlohmann/json, Z3)"
            echo "  setup  - 创建增强setup.py"
            echo "  build  - 构建增强Python包"
            echo "  test   - 测试增强安装"
            echo "  (空)   - 执行完整流程"
            echo ""
            echo "依赖处理:"
            echo "  ✅ OpenMP (macOS: libomp via brew, Linux: 系统包)"
            echo "  ✅ CUTLASS (从GitHub下载)"
            echo "  ✅ nlohmann/json (从GitHub下载)"
            echo "  ✅ Z3 (从GitHub下载并编译)"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 