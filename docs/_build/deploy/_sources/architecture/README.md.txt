# Architecture Documentation

This directory contains detailed architectural design documentation for YICA/YiRage.

## 📖 Documentation Overview

### Core Architecture
- **[YiRage Architecture](yirage-architecture.md)** - Complete design of the AI kernel super optimizer
- **[YICA Architecture Detailed](yica-architecture-detailed.md)** - In-depth technical analysis of YICA hardware and software integration
- **[Modular Architecture](modular-architecture.md)** - System modular design approach
- **[Implementation Summary](implementation-summary.md)** - Overview of overall architecture implementation

### Integration and Extensions
- **[YiRage Integration Plan](yirage-integration-plan.md)** - System integration design
- **[YiRage Extension](yirage-extension.md)** - Feature extension design
- **[YiRage Updates](yirage-updates.md)** - Version update documentation

## 🏗️ Architecture Overview

### YICA Hardware Architecture
```
YICA Computing System
├── 8 Dies
│   ├── 4 Clusters (per Die)
│   │   ├── 16 CIM Arrays (per Cluster)
│   │   └── SPM Memory Management
│   └── Internal High-Speed Interconnect
├── Three-tier Memory Hierarchy
│   ├── Register File Layer (Fastest)
│   ├── SPM Layer (Programmable)
│   └── DRAM Layer (High Capacity)
└── YIS Instruction Set Support
```

### YiRage Software Architecture
```
YiRage AI Kernel Super Optimizer
├── Frontend Interfaces
│   ├── Python API
│   ├── C++ API
│   └── Command Line Tools
├── Core Engine
│   ├── Graph Search Algorithms
│   ├── Code Generator
│   └── Performance Evaluator
├── Backend Support
│   ├── CUDA Backend
│   ├── Triton Backend
│   ├── YICA Backend
│   └── Generic Backend
└── Optimization Strategies
    ├── Operator Fusion
    ├── Memory Optimization
    └── Parallelization
```

## 🎯 Design Principles

### 1. Self-Contained Architecture
- All necessary components built-in
- No dependency on external complex source files
- One-click build and deployment

### 2. Environment Agnostic Design
- Compiles in any environment
- Works even with hardware mismatches
- Cross-platform compatibility

### 3. Backend Separation
- Reduces compilation time
- Flexible backend selection
- Easy to extend and maintain

### 4. High-Performance Design
- Compute-in-Memory architecture
- Multi-level parallel optimization
- Intelligent memory management

## 📊 Key Specifications

### Hardware Specifications
- **CIM Arrays**: 512 units (8×4×16)
- **Memory Bandwidth**: High-speed SPM + DRAM hierarchical structure
- **Instruction Set**: YIS specialized instruction set
- **Precision Support**: FP16/FP32/INT8

### Performance Targets
- **Matrix Multiplication**: 2.2x speedup vs CUDA
- **Attention Mechanism**: 1.5x speedup vs Triton
- **End-to-End Inference**: 2.5x speedup vs PyTorch
- **Energy Efficiency**: 3x improvement vs traditional architectures

## 🔄 System Components

### Core Optimization Engine
- **Multi-objective Search**: Balances latency, energy efficiency, and memory utilization
- **Architecture Awareness**: Deep integration with YICA CIM characteristics
- **Hierarchical Optimization**: Algorithm, operator, kernel, and instruction level optimization

### Backend Abstraction Layer
- **Unified Interface**: Common API across all backends
- **Dynamic Selection**: Runtime backend switching
- **Extensible Design**: Easy addition of new backends

### Performance Analysis Framework
- **Real-time Profiling**: Live performance monitoring
- **Comparative Analysis**: Multi-backend performance comparison
- **Optimization Guidance**: Automated optimization recommendations

## 🛠️ Development Architecture

### Modular Design
- **Independent Components**: Each module can be developed and tested separately
- **Clear Interfaces**: Well-defined APIs between modules
- **Extensible Framework**: Easy to add new features and capabilities

### Testing Strategy
- **Unit Testing**: Comprehensive test coverage for individual components
- **Integration Testing**: End-to-end system validation
- **Performance Testing**: Benchmark-driven development approach

### Documentation Framework
- **API Documentation**: Complete reference for all interfaces
- **Architecture Guides**: Detailed design documentation
- **Tutorial System**: Step-by-step learning materials

## 🔗 Related Documentation

- [Getting Started](../getting-started/) - Basic concepts and setup
- [Development Guide](../development/) - Development environment setup
- [Deployment Operations](../deployment/) - Deployment and operations
- [API Documentation](../api/) - Programming interface reference

## 📈 Architecture Evolution

### Current Version (v2.0)
- Complete YICA hardware architecture implementation
- YiRage engine core functionality
- Multi-backend support
- Production-ready deployment

### Next Version Roadmap
- Extended operator support
- Further performance optimization
- Enhanced ecosystem toolchain
- Advanced debugging capabilities

## 🎛️ Configuration Management

### Architecture Configuration
- **Hardware Profiles**: Different YICA hardware configurations
- **Optimization Profiles**: Predefined optimization strategies
- **Backend Configurations**: Backend-specific settings

### Runtime Configuration
- **Dynamic Tuning**: Runtime parameter adjustment
- **Performance Monitoring**: Live performance tracking
- **Adaptive Optimization**: Self-tuning based on workload characteristics

---

*These architecture documents provide a complete technical perspective for understanding the YICA/YiRage system. It is recommended to read them in the order: Core Architecture → Integration and Extensions.*