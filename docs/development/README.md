# Development Documentation

This directory contains development environment setup, contribution guidelines, and developer resources for YICA/YiRage.

## 📖 Documentation Overview

### Current Documentation
- **Development Environment Setup** - Setting up local development environment
- **Contribution Guidelines** - How to contribute to the project
- **Testing Framework** - Running and writing tests
- **Performance Testing** - Benchmarking and performance analysis
- **[Troubleshooting Guide](troubleshooting-guide.md)** - Comprehensive debugging and issue resolution

### Planned Documentation
- **Backend Development Guide** - Creating custom optimization backends
- **API Development** - Extending and modifying APIs
- **Integration Guide** - Integrating with external frameworks

## 🛠️ Development Environment Setup

### Prerequisites
- **Operating System**: Linux (Ubuntu 20.04+), macOS (11+), or Windows 10+ with WSL2
- **Compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **Python**: 3.8 or higher
- **CMake**: 3.16 or higher
- **Git**: Latest version

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yica-ai/yica-yirage.git
cd yica-yirage

# Set up development environment
./scripts/setup-dev-env.sh

# Build in debug mode
mkdir build-debug && cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
make -j$(nproc)

# Run tests
make test
```

### Docker Development Environment
```bash
# Build development container
docker build -f docker/Dockerfile.dev -t yica-dev .

# Run development container
docker run -it --rm \
  -v $(pwd):/workspace \
  -v ~/.gitconfig:/home/dev/.gitconfig:ro \
  yica-dev

# Inside container
cd /workspace
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## 🏗️ Project Structure

```
yica-yirage/
├── yirage/                    # Main source code
│   ├── src/                   # C++ source files
│   ├── include/               # Header files
│   ├── python/                # Python bindings
│   └── tests/                 # Unit tests
├── docs/                      # Documentation
├── scripts/                   # Build and utility scripts
├── docker/                    # Docker configurations
├── cmake/                     # CMake modules
├── tests/                     # Integration tests
└── examples/                  # Usage examples
```

### Core Components
- **Core Engine** (`yirage/src/core/`) - Main optimization algorithms
- **Backend Abstraction** (`yirage/src/backends/`) - Backend interface implementations
- **Python Bindings** (`yirage/python/`) - Python API implementation
- **Testing Framework** (`tests/`) - Comprehensive test suite

## 💻 Development Workflow

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-optimization-algorithm

# Make changes and commit
git add .
git commit -m "feat: add new optimization algorithm

- Implement genetic algorithm for graph optimization
- Add unit tests for new algorithm
- Update documentation"

# Push and create pull request
git push origin feature/new-optimization-algorithm
```

### Code Style Guidelines

#### C++ Code Style
```cpp
// Use camelCase for variables and functions
int optimizationLevel = 3;
void optimizeGraph(const Graph& graph);

// Use PascalCase for classes and namespaces
class GraphOptimizer {
public:
    // Public interface
    OptimizationResult optimize(const Graph& graph);
    
private:
    // Private implementation
    void performOptimization();
};

namespace yirage {
namespace optimization {
    // Namespace content
}
}
```

#### Python Code Style
```python
# Follow PEP 8 guidelines
class YirageOptimizer:
    """Main optimizer class for YiRage engine."""
    
    def __init__(self, backend: str = "yica"):
        """Initialize optimizer with specified backend."""
        self.backend = backend
        self._config = self._load_default_config()
    
    def optimize(self, model: Any) -> OptimizationResult:
        """Optimize the given model."""
        return self._run_optimization(model)
```

### Testing Guidelines

#### Unit Testing
```cpp
// C++ unit tests using Google Test
#include <gtest/gtest.h>
#include "yirage/optimizer.h"

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        optimizer = std::make_unique<Optimizer>("yica");
    }
    
    std::unique_ptr<Optimizer> optimizer;
};

TEST_F(OptimizerTest, BasicOptimization) {
    Graph input_graph = CreateTestGraph();
    auto result = optimizer->optimize(input_graph);
    
    EXPECT_GT(result.speedup, 1.0);
    EXPECT_TRUE(result.is_valid);
}
```

```python
# Python unit tests using pytest
import pytest
import yirage

class TestYirageOptimizer:
    def setup_method(self):
        self.optimizer = yirage.Optimizer(backend="yica")
    
    def test_basic_optimization(self):
        model = create_test_model()
        result = self.optimizer.optimize(model)
        
        assert result.speedup > 1.0
        assert result.is_valid
```

#### Integration Testing
```bash
# Run full test suite
make test

# Run specific test categories
ctest -L unit
ctest -L integration
ctest -L performance

# Run with coverage
make coverage
```

## 🔧 Build System

### CMake Configuration
```cmake
# Main CMakeLists.txt structure
cmake_minimum_required(VERSION 3.16)
project(YiRage VERSION 2.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_CUDA_BACKEND "Build CUDA backend" ON)
option(BUILD_YICA_BACKEND "Build YICA backend" ON)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Add subdirectories
add_subdirectory(yirage)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

### Build Configurations
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..

# Release build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

# Custom configuration
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CUDA_BACKEND=OFF \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

## 🧪 Testing Framework

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Benchmark and performance validation
- **End-to-End Tests**: Complete workflow testing

### Running Tests
```bash
# Build and run all tests
make test

# Run specific test suites
./build/tests/unit_tests
./build/tests/integration_tests
./build/tests/performance_tests

# Run with detailed output
ctest --verbose

# Run tests in parallel
ctest -j$(nproc)
```

### Writing New Tests
```cpp
// Add new test file: tests/test_new_feature.cpp
#include <gtest/gtest.h>
#include "yirage/new_feature.h"

TEST(NewFeatureTest, BasicFunctionality) {
    NewFeature feature;
    EXPECT_TRUE(feature.is_initialized());
    
    auto result = feature.process(test_input);
    EXPECT_EQ(result.status, Status::SUCCESS);
}
```

## 🚀 Performance Testing

### Benchmarking Framework
```cpp
// Performance test example
#include <benchmark/benchmark.h>
#include "yirage/optimizer.h"

static void BM_OptimizeGraph(benchmark::State& state) {
    auto optimizer = CreateOptimizer("yica");
    auto graph = CreateBenchmarkGraph(state.range(0));
    
    for (auto _ : state) {
        auto result = optimizer->optimize(graph);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_OptimizeGraph)
    ->Range(8, 8<<10)
    ->Complexity(benchmark::oNLogN);
```

### Continuous Performance Monitoring
```yaml
# .github/workflows/performance.yml
name: Performance Tests
on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build benchmarks
      run: |
        mkdir build && cd build
        cmake -DBUILD_BENCHMARKS=ON ..
        make -j$(nproc)
    - name: Run benchmarks
      run: |
        cd build
        ./benchmarks/yirage_benchmarks --benchmark_format=json > results.json
    - name: Store results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'googlecpp'
        output-file-path: build/results.json
```

## 🔍 Debugging Tools

### Debug Build Configuration
```bash
# Build with debug symbols and sanitizers
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DENABLE_SANITIZERS=ON \
      -DENABLE_DEBUG_LOGGING=ON ..
make -j$(nproc)
```

### Using GDB
```bash
# Debug with GDB
gdb ./build/yirage_optimizer
(gdb) set args --input test.py --backend yica
(gdb) run
(gdb) bt  # backtrace on crash
```

### Memory Debugging
```bash
# Use Valgrind for memory checking
valgrind --tool=memcheck --leak-check=full ./build/tests/unit_tests

# Use AddressSanitizer
export ASAN_OPTIONS=abort_on_error=1:halt_on_error=1
./build/tests/unit_tests
```

### Performance Profiling
```bash
# Profile with perf
perf record -g ./build/yirage_optimizer --input large_model.py
perf report

# Profile with gperftools
export CPUPROFILE=yirage.prof
./build/yirage_optimizer --input model.py
google-pprof --web ./build/yirage_optimizer yirage.prof
```

## 📚 Contributing Guidelines

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Write** tests for new functionality
5. **Update** documentation
6. **Submit** a pull request

### Pull Request Process
1. Ensure all tests pass
2. Update documentation if needed
3. Follow code style guidelines
4. Provide clear commit messages
5. Request review from maintainers

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Breaking changes are documented

## 🔗 Development Resources

### Internal Resources
- [Architecture Documentation](../architecture/) - System design details
- [API Documentation](../api/) - Complete API reference
- [Project Management](../project-management/) - Development planning

### External Resources
- [CMake Documentation](https://cmake.org/documentation/)
- [Google Test Guide](https://google.github.io/googletest/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/)

### Development Tools
- **IDEs**: VS Code, CLion, Qt Creator
- **Debuggers**: GDB, LLDB, Visual Studio Debugger
- **Profilers**: perf, Valgrind, Intel VTune
- **Static Analysis**: Clang Static Analyzer, PVS-Studio

## 📞 Developer Support

### Getting Help
- **GitHub Discussions**: Technical questions and discussions
- **Discord**: Real-time chat with developers
- **Stack Overflow**: Tag questions with `yica-yirage`
- **Email**: dev-support@yica-yirage.org

### Reporting Issues
When reporting bugs or issues:
1. Use the issue template
2. Provide minimal reproduction case
3. Include system information
4. Attach relevant logs
5. Specify expected vs actual behavior

---

*This development documentation is maintained by the core development team. For updates and contributions, please submit pull requests or contact the development team.*