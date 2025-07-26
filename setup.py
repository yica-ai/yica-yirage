#!/usr/bin/env python3
"""
YICA-Yirage: High-Performance AI Computing Optimization Framework
Supporting In-Memory Computing Architecture with Code Optimization and Triton Conversion
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Version information
VERSION = "1.0.1"
DESCRIPTION = "Yirage: AI Computing Optimization Framework for In-Memory Computing Architecture"
LONG_DESCRIPTION = """
YICA-Yirage is a high-performance AI computing optimization framework designed for in-memory computing architectures.

Core Features:
- 🚀 Yirage-based universal code optimization
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
pip install yica-yirage
```
"""

# Dependencies
REQUIREMENTS = [
    "numpy>=1.19.0",
    "torch>=1.12.0",
    "triton>=2.0.0; sys_platform=='linux'",
    "z3-solver>=4.8.0",
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

# Main configuration
setup(
    name="yica-yirage",
    version=VERSION,
    author="YICA Team",
    author_email="contact@yica.ai",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yica-ai/yica-yirage",
    project_urls={
        "Bug Tracker": "https://github.com/yica-ai/yica-yirage/issues",
        "Documentation": "https://yica-yirage.readthedocs.io/",
        "Source Code": "https://github.com/yica-ai/yica-yirage",
    },
    
    # Package configuration
    packages=find_packages(where="yirage/python"),
    package_dir={"": "yirage/python"},
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Classifications
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    keywords="ai, optimization, compiler, triton, yica, yirage, deep-learning, in-memory-computing",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "yica-optimizer=yirage.yica_optimizer:main",
            "yica-benchmark=yirage.yica_performance_monitor:main",
            "yica-analyze=yirage.yica_advanced:main",
        ],
    },
    
    # Include data files
    include_package_data=True,
    package_data={
        "yirage": [
            "*.py",
            "_cython/*.pyx",
            "_cython/*.pxd",
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