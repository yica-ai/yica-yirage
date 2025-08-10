# Development Guide

This directory contains development-related documentation and guides for YICA/YiRage.

## 📖 Documentation List

### Testing and Performance
- **[Performance Testing](performance-testing.md)** - Performance testing methods, tools, and benchmarks

### Planned Documentation
- **Build Guide** - Detailed steps for building the project from source
- **Debugging Guide** - Debugging techniques and troubleshooting
- **Contribution Guide** - How to contribute code to the project
- **API Development** - How to develop new API interfaces
- **Backend Extensions** - How to add new computation backends

## 🛠️ Development Environment

### Basic Requirements
- **C++17** or higher
- **Python 3.8+**
- **CMake 3.16+**
- **CUDA** (optional, for CUDA backend)

### Recommended Tools
- **IDE**: VSCode, CLion, or other C++/Python IDEs
- **Debugger**: GDB, LLDB
- **Performance Analysis**: Perf, VTune, Nsight
- **Version Control**: Git

## 🔧 Build System

### Quick Build
```bash
# Basic build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Enable all backends
cmake -DBUILD_ALL_BACKENDS=ON ..
make -j$(nproc)

# Build only specific backend
cmake -DBUILD_CPU_BACKEND=ON -DBUILD_GPU_BACKEND=OFF ..
make -j$(nproc)
```

### Python Package Build
```bash
# Standard installation
cd yirage/python
pip install -e .

# Simplified installation (compatibility mode)
python simple_cython_setup.py build_ext --inplace
```

## 🧪 Testing Framework

### Unit Tests
```bash
# C++ tests
cd build
ctest

# Python tests
cd yirage/python
python -m pytest tests/
```

### Performance Tests
```bash
# Run benchmarks
python yirage/benchmark/run_benchmarks.py

# Compare different backends
python yirage/benchmark/compare_backends.py
```

## 📊 Performance Analysis

### Built-in Performance Tools
- **YiRage Profiler**: Built-in performance analyzer
- **YICA Monitor**: YICA architecture-specific monitoring tools
- **Backend Comparator**: Multi-backend performance comparison

### External Tool Integration
- **CUDA Profiler**: Nsight Systems/Compute
- **CPU Profiler**: Intel VTune, perf
- **Memory Profiler**: Valgrind, AddressSanitizer

## 🔍 Debugging Techniques

### Common Issues
1. **Compilation Errors**: Check dependencies and compiler versions
2. **Runtime Errors**: Use debug mode and breakpoints
3. **Performance Issues**: Use profiler to analyze bottlenecks
4. **Memory Issues**: Use memory checking tools

### Debug Commands
```bash
# Debug mode build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Debug with GDB
gdb ./yirage_optimizer
(gdb) run --backend yica input.py

# Memory checking
valgrind --tool=memcheck ./yirage_optimizer
```

## 🚀 Best Practices

### Code Quality
- Follow C++17 standards
- Use smart pointers for memory management
- Write unit tests
- Add detailed comments

### Performance Optimization
- Prioritize algorithm complexity
- Use parallelization appropriately
- Pay attention to memory access patterns
- Measure before optimizing

### Compatibility Design
- Support multiple compilers
- Compatible with different operating systems
- Gracefully handle missing dependencies
- Provide fallback mechanisms

## 🔗 Related Resources

### Internal Documentation
- [Architecture Design](../architecture/) - System architecture details
- [Production Design](../design/) - Production environment design
- [API Documentation](../api/) - Programming interface reference

### External Resources
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Cython User Guide](https://cython.readthedocs.io/)

## 📈 Contribution Process

### Code Contributions
1. Fork the project repository
2. Create a feature branch
3. Write code and tests
4. Submit Pull Request
5. Code review and merge

### Documentation Contributions
1. Identify documentation improvements
2. Write or update documentation
3. Check formatting and links
4. Submit documentation PR

---

*This development guide will be continuously updated. Contributions for more development-related documentation and best practices are welcome.*
