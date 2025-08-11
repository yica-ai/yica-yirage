# Getting Started Guide

Welcome to YICA/YiRage! This section provides essential information to get you started.

## 📖 Documentation Overview

### [Design Philosophy](design-philosophy.md)
Understand the core design principles of YICA/YiRage:
- Self-Contained Architecture
- Environment Agnostic Design
- The True Purpose of Backend Separation
- Code Transformation and Optimization Tool Philosophy

### [Quick Reference](quick-reference.md)
Quick reference for common commands and operations:
- One-click deployment commands
- Access addresses and ports
- Container management commands
- Troubleshooting guide

## 🚀 Recommended Reading Order

1. **[Design Philosophy](design-philosophy.md)** - Understand the project's design philosophy
2. **[Quick Reference](quick-reference.md)** - Master basic operation commands
3. **[Architecture Design](../architecture/)** - Deep dive into system architecture
4. **[Deployment Operations](../deployment/)** - Learn deployment and operations

## 💡 Core Concepts

### YICA Architecture
- **Compute-in-Memory (CIM)**: Computing units directly integrated into memory
- **YIS Instruction Set**: Custom instruction set designed specifically for YICA
- **Three-tier Memory Hierarchy**: Register files, SPM, and DRAM

### YiRage Engine
- **Multi-backend Support**: CUDA, Triton, and YICA backends
- **Automatic Optimization**: Intelligent search for optimal computation graphs
- **Transformation Tool**: Code transformation and optimization capabilities

## 🎯 Use Cases

- **AI Model Inference**: Significantly improve inference performance
- **Operator Optimization**: Automatically generate optimized computation kernels
- **Cross-platform Deployment**: Support multiple hardware backends
- **Research and Development**: Algorithm validation and performance analysis

## ❓ Frequently Asked Questions

**Q: What is Compute-in-Memory architecture?**
A: It integrates computing units directly into memory, reducing data movement and improving computational efficiency.

**Q: How is YiRage different from other optimization tools?**  
A: YiRage is a transformation optimization tool that focuses on code transformation and multi-backend support, rather than depending on specific hardware.

**Q: How to choose the appropriate backend?**
A: The system automatically selects based on the hardware environment, or you can manually specify `backend="yica"` etc.

## 🔗 Next Steps

- Read [YiRage Architecture Documentation](../architecture/yirage-architecture.md)
- Check [Deployment Guide](../deployment/)
- Try [API Examples](../api/)

## 🛠️ Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space

### Optional Requirements
- **CUDA**: For GPU backend support
- **Docker**: For containerized deployment
- **OpenMP**: For parallel optimization (automatically detected)

## 📋 Installation Overview

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/your-org/yica-yirage.git
cd yica-yirage

# Build and install
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Docker Installation
```bash
# One-click deployment
./scripts/docker_yica_deployment.sh
```

For detailed installation instructions, see the [Deployment Guide](../deployment/).

## 🎓 Learning Path

### For Beginners
```
1. Getting Started → 2. Environment Setup → 3. Basic Concepts → 4. Simple Examples
```

### For Developers
```
1. Architecture Understanding → 2. API Learning → 3. Advanced Features → 4. Custom Development
```

### For DevOps
```
1. Deployment Basics → 2. Environment Configuration → 3. Monitoring Management → 4. Troubleshooting
```

### For Researchers
```
1. Theoretical Foundation → 2. Algorithm Principles → 3. Performance Analysis → 4. Optimization Research
```