<div align="center">

![YICA Logo](docs/_static/images/yica-logo.svg)

# YICA-YiRage: AI Computing Optimization Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/yica-ai/yica-yirage)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://yica-yirage.readthedocs.io/)
[![GitHub Stars](https://img.shields.io/github/stars/yica-ai/yica-yirage?style=social)](https://github.com/yica-ai/yica-yirage)

</div>

**YICA-YiRage** is an AI computing optimization framework specifically designed for in-memory computing architectures. It extends the YiRage superoptimization engine with YICA (Yet another In-memory Computing Architecture) support, providing automated GPU kernel generation and optimization for deep learning workloads on specialized hardware.

## 🌟 Key Features

- **🚀 Automated Kernel Generation**: Automatically generates optimized GPU kernels without manual CUDA/Triton programming
- **🧠 In-Memory Computing Support**: Specialized optimizations for in-memory computing architectures 
- **⚡ Superoptimization**: Multi-level optimization techniques for maximum performance
- **🔄 PyTorch Integration**: Seamless integration with existing PyTorch workflows
- **🎯 Production Ready**: Comprehensive testing and validation framework
- **📊 Performance Monitoring**: Built-in profiling and performance analysis tools

## 🚀 Quick Installation

### From PyPI (Recommended)
```bash
pip install yica-yirage
```

### From Source
```bash
git clone --recursive https://github.com/yica-ai/yica-yirage.git
cd yica-yirage
pip install -e . -v
```

### Prerequisites
- Python 3.8+
- PyTorch 1.12.0+
- CUDA 11.0+ (optional, for GPU acceleration)
- CMake 3.18.0+
- Triton 2.0.0+ (Linux only)

## 💻 Quick Start

### Basic Usage

```python
import yirage as yr

# Create a kernel graph
graph = yr.new_kernel_graph()

# Define input tensors
X = graph.new_input(dims=(1024, 512), dtype=yr.float16)
W = graph.new_input(dims=(512, 256), dtype=yr.float16)

# Add operations
Y = graph.rms_norm(X, normalized_shape=(512,))
Z = graph.matmul(Y, W)

# Mark outputs
graph.mark_output(Z)

# Generate optimized kernel
kernel = graph.superoptimize()

# Use in PyTorch
import torch
x = torch.randn(1024, 512, dtype=torch.float16, device='cuda')
w = torch.randn(512, 256, dtype=torch.float16, device='cuda')
output = kernel(inputs=[x, w])
```

### Advanced Example: Transformer Layer Optimization

```python
def get_optimized_transformer_layer(batch_size, seq_len, hidden_dim):
    graph = yr.new_kernel_graph()
    
    # Input tensors
    X = graph.new_input(dims=(batch_size, seq_len, hidden_dim), dtype=yr.float16)
    Wqkv = graph.new_input(dims=(hidden_dim, 3 * hidden_dim), dtype=yr.float16)
    
    # Fused RMSNorm + Linear
    Y = graph.rms_norm(X, normalized_shape=(hidden_dim,))
    QKV = graph.matmul(Y, Wqkv)
    
    graph.mark_output(QKV)
    return graph.superoptimize()

# Generate kernel
kernel = get_optimized_transformer_layer(32, 2048, 4096)

# Use in training/inference
qkv_output = kernel(inputs=[hidden_states, qkv_weights])
```

## 🏗️ Architecture

<div align="center">

![Architecture Diagram](https://img.shields.io/badge/Architecture-YICA--YiRage-blue?style=for-the-badge)

</div>

YICA-YiRage consists of three main components:

1. **🧠 YiRage Core**: Multi-level superoptimization engine
2. **⚡ YICA Backend**: In-memory computing architecture support  
3. **🐍 Python Interface**: High-level API for easy integration

```
┌─────────────────────────────────────────────────────────────┐
│                    YICA-YiRage Framework                    │
├─────────────────────────────────────────────────────────────┤
│  🐍 Python API Layer                                       │
│  ├── Graph Construction (yr.new_kernel_graph())           │
│  ├── Tensor Operations (matmul, rms_norm, etc.)           │
│  └── PyTorch Integration                                   │
├─────────────────────────────────────────────────────────────┤
│  🧠 YiRage Optimization Engine                             │
│  ├── Search-based Optimization                            │
│  ├── Multi-level Code Generation                          │
│  ├── Triton/CUDA Backend                                  │
│  └── Performance Profiling                                │
├─────────────────────────────────────────────────────────────┤
│  ⚡ YICA Hardware Abstraction                              │
│  ├── In-Memory Computing Support                          │
│  ├── Hardware-Specific Optimizations                      │
│  └── Memory Management                                     │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Documentation

- **[Getting Started Guide](docs/getting-started/README.md)**: Basic setup and first steps
- **[API Reference](docs/api/README.md)**: Complete API documentation
- **[Architecture Guide](docs/architecture/README.md)**: System design and internals
- **[Development Guide](docs/development/README.md)**: Contributing and development setup
- **[Deployment Guide](docs/deployment/README.md)**: Production deployment instructions

## 🧪 Testing

The project includes comprehensive testing suites:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/yica/           # YICA-specific tests
python -m pytest tests/cpu/            # CPU tests
python -m pytest tests/gpu/            # GPU tests

# Run performance benchmarks
python scripts/run_yica_benchmarks.sh
```

## 🚀 Performance

<div align="center">

![Performance Badge](https://img.shields.io/badge/Performance-Up%20to%201.7x%20Faster-brightgreen?style=for-the-badge)
![Memory Badge](https://img.shields.io/badge/Memory-Up%20to%2020%25%20Reduction-blue?style=for-the-badge)

</div>

YICA-YiRage achieves significant performance improvements over baseline PyTorch:

| 🔥 Operation | ⚡ Speedup | 💾 Memory Reduction |
|--------------|-----------|---------------------|
| **RMSNorm + Linear** | `1.5-1.7x` | `15-20%` |
| **Attention (Fused)** | `1.3-1.5x` | `10-15%` |
| **Transformer Layer** | `1.4-1.6x` | `12-18%` |

> 📊 *Results measured on NVIDIA A100 with mixed precision training*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone --recursive https://github.com/yica-ai/yica-yirage.git
cd yica-yirage

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use YICA-YiRage in your research, please cite:

```bibtex
@software{yica_yirage_2024,
  title={YICA-YiRage: AI Computing Optimization Framework for In-Memory Computing Architecture},
  author={YICA Team},
  year={2024},
  url={https://github.com/yica-ai/yica-yirage},
  version={1.0.1}
}
```

## 🔗 Related Projects

- **[YiRage](https://github.com/yirage-project/yirage)**: Original superoptimization engine
- **[Triton](https://github.com/openai/triton)**: GPU kernel language
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

## 🆘 Support

<div align="center">

| 📖 Documentation | 🐛 Issues | 💬 Discussions | ✉️ Contact |
|:----------------:|:---------:|:-------------:|:---------:|
| [Read the Docs](https://yica-yirage.readthedocs.io/) | [GitHub Issues](https://github.com/yica-ai/yica-yirage/issues) | [GitHub Discussions](https://github.com/yica-ai/yica-yirage/discussions) | contact@yica.ai |

</div>

---

<div align="center">

![YICA Logo](docs/_static/images/yica-logo.svg)

**Made with ❤️ by the YICA Team**

[![YICA](https://img.shields.io/badge/Powered%20by-YICA-blue?style=for-the-badge)](https://github.com/yica-ai)
[![YiRage](https://img.shields.io/badge/Built%20with-YiRage-orange?style=for-the-badge)](https://github.com/yirage-project)

*Unleashing the power of AI computing on in-memory architectures* 🚀

</div>
