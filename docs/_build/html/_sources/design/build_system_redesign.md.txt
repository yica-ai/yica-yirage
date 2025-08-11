# YICA/YiRage Production Build System Design

## Core Principles

### 1. Graceful Degradation
- **Never fail completely** due to missing optional dependencies
- **Feature-based building** - enable/disable features based on available dependencies
- **Clear capability reporting** - users know exactly what's available

### 2. Dependency Hierarchy

```
├── Core (Required)
│   ├── C++17 compiler
│   ├── CMake 3.18+
│   ├── Python 3.8+
│   └── Basic math libraries
├── Enhanced (Strongly Recommended)
│   ├── Z3 Solver (SMT solving)
│   ├── OpenMP (Parallel optimization)
│   └── PyTorch (Model integration)
└── Accelerated (Optional)
    ├── CUDA (GPU acceleration)
    ├── ROCm (AMD GPU support)
    ├── Triton (Kernel compilation)
    └── CUTLASS (Matrix operations)
```

### 3. Build Modes

#### Mode 1: Core Build (Always Works)
```bash
cmake -DYICA_BUILD_MODE=CORE
# - CPU-only computation
# - Basic optimization algorithms
# - Pure Python fallbacks
# - No external GPU dependencies
```

#### Mode 2: Enhanced Build (Recommended)
```bash
cmake -DYICA_BUILD_MODE=ENHANCED
# - Parallel optimization with OpenMP
# - Advanced SMT solving with Z3
# - PyTorch model integration
# - Automatic fallback to Core if dependencies missing
```

#### Mode 3: Full Build (Maximum Performance)
```bash
cmake -DYICA_BUILD_MODE=FULL
# - All acceleration features
# - GPU computation support
# - Triton kernel compilation
# - Professional performance monitoring
```

## Implementation Strategy

### 1. Smart Dependency Detection

```cmake
# yica_dependencies.cmake
function(detect_yica_capabilities)
    # Initialize capability flags
    set(YICA_HAS_OPENMP OFF PARENT_SCOPE)
    set(YICA_HAS_CUDA OFF PARENT_SCOPE)
    set(YICA_HAS_Z3 OFF PARENT_SCOPE)
    set(YICA_HAS_TRITON OFF PARENT_SCOPE)
    
    # Detect OpenMP
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(YICA_HAS_OPENMP ON PARENT_SCOPE)
        message(STATUS "✅ OpenMP detected - Parallel optimization enabled")
    else()
        message(STATUS "⚠️  OpenMP not found - Using serial optimization")
    endif()
    
    # Detect CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(YICA_HAS_CUDA ON PARENT_SCOPE)
        message(STATUS "✅ CUDA detected - GPU acceleration enabled")
    else()
        message(STATUS "⚠️  CUDA not found - CPU-only mode")
    endif()
    
    # Detect Z3
    find_package(Z3 QUIET)
    if(Z3_FOUND)
        set(YICA_HAS_Z3 ON PARENT_SCOPE)
        message(STATUS "✅ Z3 detected - Advanced SMT solving enabled")
    else()
        # Try pip-installed Z3
        execute_process(
            COMMAND python3 -c "import z3; print(z3.get_version_string())"
            RESULT_VARIABLE Z3_PIP_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(Z3_PIP_RESULT EQUAL 0)
            set(YICA_HAS_Z3 ON PARENT_SCOPE)
            message(STATUS "✅ Z3 (pip) detected - SMT solving enabled")
        else()
            message(STATUS "⚠️  Z3 not found - Basic solving only")
        endif()
    endif()
    
    # Generate capability report
    generate_capability_report()
endfunction()
```

### 2. Feature-Based Configuration

```cmake
# Configure based on detected capabilities
if(YICA_HAS_OPENMP)
    target_link_libraries(yirage_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(yirage_core PRIVATE YICA_ENABLE_OPENMP)
endif()

if(YICA_HAS_CUDA)
    enable_language(CUDA)
    target_link_libraries(yirage_core PRIVATE CUDA::cudart)
    target_compile_definitions(yirage_core PRIVATE YICA_ENABLE_CUDA)
endif()

if(YICA_HAS_Z3)
    target_link_libraries(yirage_core PRIVATE ${Z3_LIBRARIES})
    target_compile_definitions(yirage_core PRIVATE YICA_ENABLE_Z3)
endif()
```

### 3. Comprehensive Error Handling

```cmake
# Error handling for missing critical components
function(validate_core_requirements)
    # Check C++17 support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
    if(NOT COMPILER_SUPPORTS_CXX17)
        message(FATAL_ERROR "❌ C++17 support required but not available")
    endif()
    
    # Check Python compatibility
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(Python3_VERSION VERSION_LESS "3.8")
        message(FATAL_ERROR "❌ Python 3.8+ required, found ${Python3_VERSION}")
    endif()
    
    # Validate CMake version
    if(CMAKE_VERSION VERSION_LESS "3.18")
        message(FATAL_ERROR "❌ CMake 3.18+ required, found ${CMAKE_VERSION}")
    endif()
endfunction()
```

## Build Targets Structure

### 1. Core Targets
```cmake
# Always available
add_library(yirage_core SHARED ${CORE_SOURCES})
add_library(yirage_compat SHARED ${COMPAT_SOURCES})  # Compatibility layer

# Python bindings (always built)
add_library(yirage_python MODULE ${PYTHON_BINDING_SOURCES})
```

### 2. Optional Targets
```cmake
# Only built if dependencies available
if(YICA_HAS_OPENMP)
    add_library(yirage_parallel SHARED ${PARALLEL_SOURCES})
endif()

if(YICA_HAS_CUDA)
    add_library(yirage_cuda SHARED ${CUDA_SOURCES})
endif()

if(YICA_HAS_Z3)
    add_library(yirage_solver SHARED ${SOLVER_SOURCES})
endif()
```

## Installation Strategy

### 1. Modular Installation
```cmake
# Core components (always installed)
install(TARGETS yirage_core yirage_compat yirage_python
        DESTINATION lib/yirage)

# Optional components
if(YICA_HAS_OPENMP)
    install(TARGETS yirage_parallel DESTINATION lib/yirage)
endif()

if(YICA_HAS_CUDA)
    install(TARGETS yirage_cuda DESTINATION lib/yirage)
endif()
```

### 2. Capability Configuration File
```cmake
# Generate runtime capability configuration
configure_file(
    ${PROJECT_SOURCE_DIR}/config/yica_capabilities.h.in
    ${PROJECT_BINARY_DIR}/include/yica_capabilities.h
)
```

## User Experience

### 1. Clear Build Status
```
🔧 YICA/YiRage Build Configuration
=====================================

Core Features:
✅ C++ Optimization Engine    ENABLED
✅ Python Integration         ENABLED
✅ Basic CPU Kernels         ENABLED

Enhanced Features:
✅ OpenMP Parallelization    ENABLED
✅ Z3 SMT Solver            ENABLED
⚠️  Triton Code Generation   DISABLED (not found)

Acceleration Features:
❌ CUDA GPU Support         DISABLED (not found)
❌ ROCm AMD Support         DISABLED (not found)

Build Mode: ENHANCED (7/9 features enabled)
=====================================
```

### 2. Runtime Feature Detection
```python
import yirage

# Check available features at runtime
print(f"OpenMP parallel optimization: {yirage.has_openmp}")
print(f"CUDA GPU acceleration: {yirage.has_cuda}")
print(f"Z3 advanced solving: {yirage.has_z3}")

# Graceful feature usage
if yirage.has_openmp:
    optimizer = yirage.ParallelOptimizer()
else:
    optimizer = yirage.SerialOptimizer()
```

## Testing Strategy

### 1. Multi-Configuration Testing
```bash
# Test all build modes
ctest -C Core -V
ctest -C Enhanced -V
ctest -C Full -V
```

### 2. Capability-Specific Tests
```bash
# Test OpenMP functionality
ctest -R "openmp" -V

# Test CUDA functionality (if available)
ctest -R "cuda" -V

# Test fallback mechanisms
ctest -R "fallback" -V
```

This design ensures that YICA/YiRage can build and run in any environment while maximizing performance when dependencies are available.
