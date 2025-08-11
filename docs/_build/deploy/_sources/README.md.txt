# YICA/YiRage Documentation

Welcome to the YICA (YICA Intelligence Computing Architecture) and YiRage (AI Kernel Super Optimizer) documentation center.

## 🚀 Quick Start

### [Getting Started Guide](getting-started/README.md)
Essential information to get started with YICA/YiRage:
- [Design Philosophy](getting-started/design-philosophy.md) - Core design principles and concepts
- [Quick Reference](getting-started/quick-reference.md) - Common commands and operations

## 🏗️ Architecture

### [Architecture Overview](architecture/README.md) 
Detailed architectural design documentation:
- [YiRage Architecture](architecture/yirage-architecture.md) - AI kernel super optimizer architecture
- [Modular Architecture](architecture/modular-architecture.md) - System modular design
- [Implementation Summary](architecture/implementation-summary.md) - Architecture implementation overview
- [YiRage Integration Plan](architecture/yirage-integration-plan.md) - System integration design
- [YiRage Extension](architecture/yirage-extension.md) - Feature extension design
- [YiRage Updates](architecture/yirage-updates.md) - Version update documentation

## 📖 User Guide

### [Usage Documentation](USAGE.md)
Comprehensive usage guide and tutorials:
- [Tutorials](tutorials/README.md) - Step-by-step tutorials and examples

## 🔌 API Reference

### [API Documentation](api/README.md)
Complete API reference and examples:
- [Analyzer API](api/analyzer.md) - YICA analyzer API reference

## 🛠️ Development

### [Development Guide](development/README.md)
Development environment and contribution guidelines

### Production Design
Enterprise-grade system design documentation:
- [Build System Redesign](design/build_system_redesign.md) - Robust build system design
- [Compatibility Layer Enhancement](design/compatibility_layer_enhancement.md) - Enhanced compatibility solutions
- [Configuration Management System](design/configuration_management_system.md) - Production-grade configuration management
- [Deployment Packaging Strategy](design/deployment_packaging_strategy.md) - Professional deployment and packaging
- [Error Handling Logging System](design/error_handling_logging_system.md) - Enterprise-level error handling
- [Testing Framework Design](design/testing_framework_design.md) - Comprehensive testing framework

## 🚀 Deployment

### [Deployment Guide](deployment/README.md)
Deployment and operations documentation:
- [Docker Deployment](deployment/docker-deployment.md) - Deploy YICA environment using Docker
- [Deployment Report](deployment/deployment-report.md) - Deployment implementation report

## 📈 Project Management

### [Project Management](project-management/README.md)
Project planning, analysis, and management documentation:
- [Backend Integration](project-management/backend-integration.md) - YICA backend integration design
- [Implementation Analysis](project-management/implementation-analysis.md) - C++ kernel implementation analysis
- [Roadmap](project-management/roadmap.md) - Development roadmap and milestones
- [Execution Plan](project-management/execution-plan.md) - Task execution planning

## 🎯 Project Overview

### YICA (YICA Intelligence Computing Architecture)
YICA is a revolutionary Compute-in-Memory (CIM) architecture designed specifically for AI computing optimization. By integrating computing units directly into memory, it significantly reduces data movement and provides exceptional performance and energy efficiency.

### YiRage (AI Kernel Super Optimizer)  
YiRage is a high-performance AI operator optimization engine that supports multiple backends (CUDA, Triton, YICA). It can automatically search and optimize computation graphs of AI models, achieving significant performance improvements.

## 🚀 Key Features

- **Compute-in-Memory Architecture**: 512 CIM arrays for highly parallel computing
- **Three-tier Memory Hierarchy**: Optimized memory management with register files, SPM, and DRAM
- **YIS Instruction Set**: Custom instruction set designed specifically for CIM architecture
- **Multi-backend Support**: Seamless switching between CUDA, Triton, and YICA backends
- **Automatic Optimization**: Intelligent search for optimal computation graphs
- **High Performance**: 2-3x performance improvement compared to traditional solutions

## 📊 Performance Metrics

| Operator Type | vs PyTorch | vs CUDA | vs Triton |
|---------------|------------|---------|-----------|
| Matrix Multiplication | 3.0x | 2.2x | - |
| Attention Mechanism | 2.8x | 1.9x | 1.5x |
| End-to-End Inference | 2.5x | 1.7x | - |

## 🔗 Related Links

- **Source Code** - YiRage core source code (located at `../yirage/`)
- **Examples** - Usage examples and demonstrations (located at `../yirage/demo/`)
- **Test Suite** - Complete test cases (located at `../tests/`)

## 📞 Getting Help

If you encounter issues while using the system, please:

1. Consult the relevant documentation
2. Check [FAQ](getting-started/quick-reference.md)
3. View [Error Handling Guide](design/error_handling_logging_system.md)
4. Submit an Issue or contact the maintenance team

---

**Documentation Version**: v2.0  
**Last Updated**: December 2024  
**Maintenance Team**: YICA Development Team