#!/usr/bin/env python3
"""
YICA-Mirage: High-Performance AI Computing Optimization Framework
Supporting In-Memory Computing Architecture with Code Optimization and Triton Conversion
"""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Version information
VERSION = "1.0.0"
DESCRIPTION = "YICA-Mirage: AI Computing Optimization Framework for In-Memory Computing Architecture"
LONG_DESCRIPTION = """
YICA-Mirage is a high-performance AI computing optimization framework designed for in-memory computing architectures.

Core Features:
- 🚀 Mirage-based universal code optimization
- 🧠 YICA in-memory computing architecture specific optimizations
- ⚡ Automatic Triton code generation
- 🔧 Multi-backend support (CPU/GPU/YICA)
- 📊 Intelligent performance tuning

Supported Platforms:
- Linux (x86_64, aarch64)
- macOS (x86_64, arm64)
- Windows (x86_64)

Installation:
```bash
pip install yica-mirage
```
"""

# Dependencies
REQUIREMENTS = [
    "numpy>=1.19.0",
    "torch>=1.12.0",
    "triton>=2.0.0",
    "z3-solver>=4.8.0",
    "pybind11>=2.6.0",
    "cmake>=3.18.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.0",
        "black>=21.0",
        "flake8>=3.8",
        "mypy>=0.900",
        "sphinx>=4.0",
        "sphinx-rtd-theme>=1.0",
    ],
    "cuda": [
        "cupy>=9.0.0",
        "nvidia-ml-py>=11.0.0",
    ],
    "rocm": [
        "torch-rocm>=1.12.0",
    ],
}

# Platform detection
def get_platform_tag():
    """Get platform tag"""
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "linux":
        if machine in ["x86_64", "amd64"]:
            return "linux_x86_64"
        elif machine in ["aarch64", "arm64"]:
            return "linux_aarch64"
    elif system == "darwin":
        if machine in ["x86_64", "amd64"]:
            return "macosx_10_14_x86_64"
        elif machine in ["arm64", "aarch64"]:
            return "macosx_11_0_arm64"
    elif system == "windows":
        return "win_amd64"
    
    return "any"

# CMake extension build
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(_build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF",
        ]

        build_args = ["--config", cfg]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

# Main configuration
setup(
    name="yica-mirage",
    version=VERSION,
    author="YICA Team",
    author_email="contact@yica.ai",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yica-ai/yica-mirage",
    project_urls={
        "Bug Tracker": "https://github.com/yica-ai/yica-mirage/issues",
        "Documentation": "https://yica-mirage.readthedocs.io/",
        "Source Code": "https://github.com/yica-ai/yica-mirage",
    },
    
    # Package configuration
    packages=find_packages(where="mirage/python"),
    package_dir={"": "mirage/python"},
    
    # C++ extensions
    ext_modules=[
        CMakeExtension("yica_mirage._core"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Classifications
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Hardware",
    ],
    
    # Keywords
    keywords="ai, optimization, compiler, triton, yica, mirage, deep-learning, in-memory-computing",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "yica-optimizer=yica_mirage.cli:main",
            "yica-benchmark=yica_mirage.benchmark:main",
            "yica-analyze=yica_mirage.analyzer:main",
        ],
    },
    
    # Include data files
    include_package_data=True,
    package_data={
        "yica_mirage": [
            "kernels/*.cu",
            "kernels/*.h",
            "configs/*.yaml",
            "templates/*.j2",
        ],
    },
    
    # Platform tags
    options={
        "bdist_wheel": {
            "plat_name": get_platform_tag(),
        }
    },
    
    zip_safe=False,
) 