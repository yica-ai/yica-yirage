#!/bin/bash

# Build YICA C++ Backend and Cython Bindings
# This script compiles the real YICA hardware acceleration backend

set -e  # Exit on error

echo "==================================================================="
echo "🚀 Building YICA Hardware Acceleration Backend"
echo "==================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect platform
PLATFORM=$(uname -s)
echo "Platform: $PLATFORM"

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ CMake not found. Please install CMake.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ CMake found: $(cmake --version | head -n1)${NC}"

# Check for C++ compiler
if ! command -v c++ &> /dev/null; then
    echo -e "${RED}❌ C++ compiler not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ C++ compiler found${NC}"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found: $(python3 --version)${NC}"

# Check for Cython
if ! python3 -c "import Cython" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Cython not found. Installing...${NC}"
    pip3 install cython>=0.29.32
fi
echo -e "${GREEN}✅ Cython found${NC}"

# Check for Z3
if [ "$PLATFORM" == "Darwin" ]; then
    # macOS
    if ! [ -d "/opt/homebrew/lib" ] || ! ls /opt/homebrew/lib/libz3* 1> /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Z3 not found. Installing via Homebrew...${NC}"
        if ! command -v brew &> /dev/null; then
            echo -e "${RED}❌ Homebrew not found. Please install Homebrew first.${NC}"
            exit 1
        fi
        brew install z3
    fi
    echo -e "${GREEN}✅ Z3 found${NC}"
else
    # Linux
    if ! ldconfig -p | grep -q libz3; then
        echo -e "${YELLOW}⚠️  Z3 not found. Please install Z3.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install libz3-dev"
        echo "  Fedora: sudo dnf install z3-devel"
        exit 1
    fi
    echo -e "${GREEN}✅ Z3 found${NC}"
fi

# Build directory
BUILD_DIR="yirage/build"
echo -e "\n${YELLOW}Setting up build directory: $BUILD_DIR${NC}"

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

# Step 1: Build C++ Runtime with YICA enabled
echo -e "\n${YELLOW}Step 1: Building C++ Runtime with YICA support...${NC}"
cd "$BUILD_DIR"

# Configure with YICA enabled and no CUDA
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_YICA=ON \
    -DYICA_HARDWARE_ACCELERATION=ON \
    -DBUILD_YICA_CYTHON_BINDINGS=OFF \
    -DUSE_CUDA=OFF \
    -DBUILD_CPP_EXAMPLES=OFF \
    -DYIRAGE_BUILD_UNIT_TEST=OFF

# Build the runtime library
echo -e "${YELLOW}Compiling C++ runtime...${NC}"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) yirage_runtime

if [ ! -f "libyirage_runtime.a" ] && [ ! -f "libyirage_runtime.so" ] && [ ! -f "libyirage_runtime.dylib" ]; then
    echo -e "${RED}❌ Failed to build yirage_runtime library${NC}"
    exit 1
fi
echo -e "${GREEN}✅ C++ runtime built successfully${NC}"

# Step 2: Build Cython extensions
echo -e "\n${YELLOW}Step 2: Building Cython extensions...${NC}"
cd ../python

# Update the cython_setup.py to not skip yica_kernels.pyx
echo -e "${YELLOW}Updating cython_setup.py to enable YICA kernels...${NC}"
cat > cython_setup_yica.py << 'EOF'
import os
from os import path
import sys
import sysconfig
from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:                                                
    from distutils.core import setup
    from distutils.extension import Extension                              
else:
    from setuptools import setup
    from setuptools.extension import Extension

def config_cython():
    """Configure Cython extensions with YICA hardware acceleration support"""
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        cython_path = path.join(path.dirname(__file__), "yirage/_cython")
        yirage_path = path.join(path.dirname(__file__), "..")
        
        # Include directories for YICA support
        include_dirs = [
            path.join(yirage_path, "include"),
            path.join(yirage_path, "include", "yirage", "yica"),
            "/opt/homebrew/include",  # For Z3 on macOS
            "/usr/local/include"      # Alternative location
        ]
        
        # Libraries including YICA support
        libraries = [
            "yirage_runtime", 
            "z3"
        ]
        
        # Library directories
        library_dirs = [
            path.join(yirage_path, "build"),
            "/opt/homebrew/lib",  # For Z3 on macOS
            "/usr/local/lib"      # Alternative location
        ]
        
        # Compile flags
        extra_compile_args = [
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-DYIRAGE_ENABLE_YICA_HARDWARE",  # Enable YICA
            "-DYICA_TARGET_YZ_G100"            # Target hardware
        ]
        
        # Link flags
        if sys.platform == 'darwin':
            extra_link_args = ["-fPIC", "-undefined", "dynamic_lookup"]
        else:
            extra_link_args = ["-fPIC"]
        
        # Build YICA kernels extension
        print("🚀 Building YICA kernels Cython extension...")
        ret.append(Extension(
            "yirage._cython.yica_kernels",
            ["yirage/_cython/yica_kernels.pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"
        ))
        
        # Build YICA operators extension
        if path.exists("yirage/_cython/yica_operators.pyx"):
            print("🚀 Building YICA operators Cython extension...")
            ret.append(Extension(
                "yirage._cython.yica_operators",
                ["yirage/_cython/yica_operators.pyx"],
                include_dirs=include_dirs,
                libraries=libraries,
                library_dirs=library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++"
            ))
        
        # Build core extension if needed
        if path.exists("yirage/_cython/core.pyx"):
            print("🚀 Building core Cython extension...")
            ret.append(Extension(
                "yirage._cython.core",
                ["yirage/_cython/core.pyx"],
                include_dirs=include_dirs,
                libraries=libraries,
                library_dirs=library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++"
            ))
        
        # Cython compiler directives for performance
        compiler_directives = {
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,  
            "initializedcheck": False,
            "cdivision": True,
            "embedsignature": True,
        }
        
        print(f"✅ Configured {len(ret)} Cython extensions")
        return cythonize(ret, compiler_directives=compiler_directives)
        
    except ImportError as e:
        print(f"❌ ERROR: Cython is not installed: {e}")
        raise SystemExit(1)

setup(
    name='yirage',
    version="1.0.5",
    description="Yirage with YICA Hardware Acceleration",
    zip_safe=False,
    install_requires=[],
    packages=find_packages(),
    ext_modules=config_cython(),
)
EOF

# Build the Cython extensions
echo -e "${YELLOW}Building Cython extensions...${NC}"
python3 cython_setup_yica.py build_ext --inplace

# Check if the extensions were built
if ls yirage/_cython/*.so 2>/dev/null || ls yirage/_cython/*.pyd 2>/dev/null; then
    echo -e "${GREEN}✅ Cython extensions built successfully:${NC}"
    ls -la yirage/_cython/*.so 2>/dev/null || ls -la yirage/_cython/*.pyd 2>/dev/null
else
    echo -e "${RED}❌ Failed to build Cython extensions${NC}"
    exit 1
fi

# Step 3: Test the installation
echo -e "\n${YELLOW}Step 3: Testing YICA backend...${NC}"
cd ../..

# Create test script
cat > test_yica_backend.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'yirage/python')

print("Testing YICA backend integration...")
print("-" * 50)

# Test 1: Check if YICA C++ backend is available
try:
    from yirage.yica.yica_backend_integration import YICA_CPP_AVAILABLE
    print(f"1. YICA C++ Backend Available: {YICA_CPP_AVAILABLE}")
    if YICA_CPP_AVAILABLE:
        print("   ✅ C++ kernels are available")
    else:
        print("   ❌ C++ kernels are NOT available")
except Exception as e:
    print(f"   ❌ Error checking C++ backend: {e}")

# Test 2: Check Cython extensions
try:
    from yirage._cython.yica_kernels import YICAMatMulOp
    print("2. ✅ Cython YICA kernels loaded successfully")
except ImportError as e:
    print(f"2. ❌ Cython YICA kernels not found: {e}")

try:
    from yirage._cython.yica_operators import YICAOperator
    print("3. ✅ Cython YICA operators loaded successfully")
except ImportError as e:
    print(f"3. ❌ Cython YICA operators not found: {e}")

# Test 3: Check core module
try:
    from yirage._cython import core
    print("4. ✅ YICA core module loaded successfully")
except ImportError as e:
    print(f"4. ❌ YICA core module not found: {e}")

# Test 4: Check if we can import the main YICA optimizer
try:
    from yirage.yica.yica_real_optimizer import YICAOptimizer
    optimizer = YICAOptimizer()
    print("5. ✅ YICA optimizer instantiated successfully")
    
    # Check if hardware acceleration is available
    if hasattr(optimizer, 'is_hardware_accelerated'):
        print(f"   Hardware acceleration: {optimizer.is_hardware_accelerated()}")
except Exception as e:
    print(f"5. ❌ Error with YICA optimizer: {e}")

print("-" * 50)
print("Testing complete!")
EOF

python3 test_yica_backend.py

echo -e "\n${GREEN}==================================================================="
echo "✅ YICA Backend Build Complete!"
echo "==================================================================="
echo ""
echo "The YICA C++ backend and Cython bindings have been built."
echo ""
echo "To use YICA in your Python code:"
echo "  export PYTHONPATH=yirage/python:\$PYTHONPATH"
echo "  python3 -m yirage.yica.yica_real_optimizer benchmark"
echo ""
echo "Note: The optimizations should now use real C++ kernels"
echo "instead of simulated performance numbers."
echo "===================================================================${NC}"
