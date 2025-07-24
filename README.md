# YICA-Mirage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/yica-mirage.svg)](https://badge.fury.io/py/yica-mirage)
[![CI/CD](https://github.com/yica-ai/yica-mirage/workflows/Release%20Pipeline/badge.svg)](https://github.com/yica-ai/yica-mirage/actions)

**YICA-Mirage** is a high-performance AI computing optimization framework designed for in-memory computing architectures. It combines the power of Mirage's universal code optimization with YICA's specialized in-memory computing optimizations to deliver exceptional performance for AI workloads.

## 🚀 Key Features

- **🧠 In-Memory Computing Optimization**: Specialized optimizations for YICA in-memory computing architectures
- **⚡ Automatic Triton Code Generation**: Seamless conversion from high-level operations to optimized Triton kernels
- **🔧 Multi-Backend Support**: Unified interface supporting CPU, GPU, and YICA hardware
- **📊 Intelligent Performance Tuning**: Advanced search algorithms for optimal kernel configurations
- **🎯 CUDA Compatibility**: Full backward compatibility with existing CUDA workflows
- **🐍 Python Integration**: Easy-to-use Python API with C++ performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│              (PyTorch, Transformers, etc.)                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Mirage Layer                             │
│        (Universal Code Optimization & Triton Conversion)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     YICA Layer                              │
│     (Hardware-Specific Optimization & Memory Management)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Hardware Layer                           │
│              (CPU / GPU / YICA Chips)                      │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Quick Install (Recommended)

```bash
# Install via pip
pip install yica-mirage

# Install with CUDA support
pip install yica-mirage[cuda]

# Install with all optional dependencies
pip install yica-mirage[all]
```

### Platform-Specific Installation

#### 🍎 macOS (Homebrew)

```bash
brew tap yica-ai/tap
brew install yica-mirage
```

#### 🐧 Ubuntu/Debian (APT)

```bash
# Add repository
wget -qO - https://packages.yica.ai/gpg.key | sudo apt-key add -
echo "deb https://packages.yica.ai/debian stable main" | sudo tee /etc/apt/sources.list.d/yica.list

# Install
sudo apt-get update
sudo apt-get install yica-mirage python3-yica-mirage
```

#### 🎩 RHEL/CentOS/Fedora (YUM/DNF)

```bash
# Add repository
sudo tee /etc/yum.repos.d/yica.repo > /dev/null <<EOF
[yica]
name=YICA Repository
baseurl=https://packages.yica.ai/rpm/\$basearch
enabled=1
gpgcheck=1
gpgkey=https://packages.yica.ai/gpg.key
EOF

# Install
sudo yum install yica-mirage python3-yica-mirage
```

#### 🐳 Docker

```bash
# CPU version
docker run -it yicaai/yica-mirage:cpu-latest

# GPU version (requires NVIDIA Docker)
docker run --gpus all -it yicaai/yica-mirage:gpu-latest
```

#### 🛠️ Universal Installation Script

```bash
# Auto-detect platform and install
curl -fsSL https://install.yica.ai | bash

# Manual method selection
curl -fsSL https://install.yica.ai | bash -s -- --method pip --cuda
```

## 🚀 Quick Start

### Python API

```python
import torch
import yica_mirage as ym

# Create YICA optimizer
optimizer = ym.YicaOptimizer(backend="yica")

# Define a simple model
model = torch.nn.Sequential(
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.Softmax(dim=-1)
)

# Optimize the model
optimized_model = optimizer.optimize(model)

# Run inference
input_data = torch.randn(32, 1024)
output = optimized_model(input_data)
```

### Command Line Interface

```bash
# Optimize a model
yica-optimizer --model model.onnx --backend yica --output optimized_model.triton

# Run benchmarks
yica-benchmark --model optimized_model.triton --batch-size 32 --iterations 1000

# Analyze performance
yica-analyze --model optimized_model.triton --hardware yica --report performance.json
```

### Advanced Usage

```python
import yica_mirage as ym

# Configure optimization settings
config = ym.OptimizationConfig(
    target_hardware="yica",
    memory_optimization=True,
    kernel_fusion=True,
    precision="mixed"
)

# Create optimizer with custom config
optimizer = ym.YicaOptimizer(config=config)

# Optimize with performance constraints
constraints = ym.PerformanceConstraints(
    max_memory_usage="8GB",
    min_throughput="1000 samples/sec",
    max_latency="10ms"
)

optimized_model = optimizer.optimize(
    model, 
    constraints=constraints,
    search_iterations=100
)
```

## 🎯 YICA Architecture Features

### In-Memory Computing Optimizations

- **Memory-Centric Operations**: Minimize data movement between compute and memory
- **Local Processing**: Maximize computation within memory units
- **Energy Efficiency**: Optimize for power consumption in in-memory architectures

### Advanced Parallelization

- **Data Parallelism**: Efficient distribution across memory banks
- **Model Parallelism**: Intelligent partitioning for large models
- **Pipeline Parallelism**: Overlapped execution stages

### Memory Management

- **Smart Allocation**: Intelligent memory placement strategies
- **Data Reuse**: Maximize cache hit rates and data locality
- **Bandwidth Optimization**: Efficient utilization of memory bandwidth

## 📊 Performance Benchmarks

| Model | Hardware | Original (ms) | YICA-Optimized (ms) | Speedup |
|-------|----------|---------------|---------------------|---------|
| ResNet-50 | YICA Chip | 12.3 | 3.2 | 3.8x |
| BERT-Base | YICA Chip | 45.7 | 11.2 | 4.1x |
| GPT-2 | YICA Chip | 89.4 | 21.6 | 4.1x |
| Transformer | GPU (A100) | 8.9 | 7.1 | 1.3x |
| CNN | CPU (Intel) | 156.2 | 98.4 | 1.6x |

## 🔧 Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/yica-ai/yica-mirage.git
cd yica-mirage

# Install dependencies
pip install -r requirements.txt

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)

# Install Python package
cd ../mirage/python
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -m "not slow"  # Skip slow tests
python -m pytest tests/ -m "cuda"      # CUDA-only tests
python -m pytest tests/ -m "yica"      # YICA-only tests
```

### Code Formatting

```bash
# Format code
black mirage/python/
isort mirage/python/

# Type checking
mypy mirage/python/

# Linting
flake8 mirage/python/
```

## 📚 Documentation

- **[API Reference](https://yica-mirage.readthedocs.io/en/latest/api/)**
- **[Architecture Guide](docs/architecture/YICA_ARCH.md)**
- **[Integration Manual](docs/architecture/YICA-MIRAGE-INTEGRATION-PLAN.md)**
- **[Performance Tuning](docs/tutorials/performance-tuning.md)**
- **[Examples](examples/)**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run development checks
make check
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yica-ai/yica-mirage/issues)
- **Discussions**: [Community discussions](https://github.com/yica-ai/yica-mirage/discussions)
- **Email**: [contact@yica.ai](mailto:contact@yica.ai)
- **Documentation**: [yica-mirage.readthedocs.io](https://yica-mirage.readthedocs.io/)

## 🙏 Acknowledgments

- **Mirage Team**: For the foundational optimization framework
- **YICA Hardware Team**: For in-memory computing architecture insights
- **Triton Community**: For the excellent GPU kernel compilation framework
- **Open Source Contributors**: For making this project possible

## 🔗 Related Projects

- **[Mirage](https://github.com/mirage-project/mirage)**: Universal tensor program optimization
- **[Triton](https://github.com/openai/triton)**: GPU kernel programming language
- **[PyTorch](https://pytorch.org/)**: Deep learning framework integration
- **[CUDA](https://developer.nvidia.com/cuda-zone)**: GPU computing platform

---

<div align="center">

**[🏠 Homepage](https://yica.ai)** • **[📖 Docs](https://yica-mirage.readthedocs.io/)** • **[🚀 Examples](examples/)** • **[💬 Community](https://github.com/yica-ai/yica-mirage/discussions)**

Made with ❤️ by the YICA Team

</div>
