# GPU CUDA后端测试配置
cmake_minimum_required(VERSION 3.16)

message(STATUS "配置GPU CUDA后端测试")

# 确保CUDA语言已启用
enable_language(CUDA)

# 查找CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found, cannot build GPU tests")
endif()

# 查找测试框架
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest未找到，使用内置测试框架")
    set(USE_BUILTIN_TEST_FRAMEWORK ON)
else()
    message(STATUS "使用GTest测试框架")
    set(USE_BUILTIN_TEST_FRAMEWORK OFF)
endif()

# GPU后端测试源文件
set(GPU_TEST_SRCS
    # 基础功能测试
    test_gpu_basic.cc
    test_gpu_memory_manager.cc
    test_gpu_cuda_runtime.cc
    test_gpu_device_properties.cc
    
    # CUDA kernel测试
    test_gpu_matmul_kernel.cu
    test_gpu_all_reduce_kernel.cu
    test_gpu_reduction_kernel.cu
    test_gpu_element_ops_kernel.cu
    
    # GPU算子测试
    test_gpu_matmul.cc
    test_gpu_all_reduce.cc
    test_gpu_rms_norm.cc
    test_gpu_reduction.cc
    test_gpu_element_ops.cc
    
    # GPU优化测试
    test_gpu_memory_optimization.cc
    test_gpu_kernel_fusion.cc
    test_gpu_mixed_precision.cc
    test_gpu_tensor_core.cc
    
    # GPU性能测试
    test_gpu_performance.cc
    test_gpu_benchmark.cc
    test_gpu_memory_bandwidth.cc
    test_gpu_compute_throughput.cc
)

# GPU后端测试可执行文件
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_executable(yica_gpu_tests ${GPU_TEST_SRCS} test_main.cc)
    target_compile_definitions(yica_gpu_tests PRIVATE USE_BUILTIN_TEST)
else()
    add_executable(yica_gpu_tests ${GPU_TEST_SRCS})
    target_link_libraries(yica_gpu_tests GTest::GTest GTest::Main)
endif()

# 链接GPU后端库
target_link_libraries(yica_gpu_tests yica_gpu yica_core)
target_link_libraries(yica_gpu_tests ${CUDA_LIBRARIES})

# GPU特定编译标志
target_compile_definitions(yica_gpu_tests PRIVATE 
    YICA_GPU_BACKEND
    USE_CUDA
    YICA_TEST_MODE
)

# CUDA编译标志
set_target_properties(yica_gpu_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# CUDA编译选项
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(yica_gpu_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O0 -g -Xcompiler=-fPIC>
    )
else()
    target_compile_options(yica_gpu_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 -Xcompiler=-fPIC>
    )
endif()

# 包含目录
target_include_directories(yica_gpu_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/mirage/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# 查找额外的GPU库
find_library(CUBLAS_LIBRARY cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUFFT_LIBRARY cufft PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

if(CUBLAS_LIBRARY)
    target_link_libraries(yica_gpu_tests ${CUBLAS_LIBRARY})
    target_compile_definitions(yica_gpu_tests PRIVATE HAS_CUBLAS)
    message(STATUS "链接cuBLAS: ${CUBLAS_LIBRARY}")
endif()

if(CUDNN_LIBRARY)
    target_link_libraries(yica_gpu_tests ${CUDNN_LIBRARY})
    target_compile_definitions(yica_gpu_tests PRIVATE HAS_CUDNN)
    message(STATUS "链接cuDNN: ${CUDNN_LIBRARY}")
endif()

if(CUFFT_LIBRARY)
    target_link_libraries(yica_gpu_tests ${CUFFT_LIBRARY})
    target_compile_definitions(yica_gpu_tests PRIVATE HAS_CUFFT)
    message(STATUS "链接cuFFT: ${CUFFT_LIBRARY}")
endif()

# 添加GPU测试
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_test(NAME gpu_basic_test COMMAND yica_gpu_tests basic)
    add_test(NAME gpu_memory_test COMMAND yica_gpu_tests memory)
    add_test(NAME gpu_kernel_test COMMAND yica_gpu_tests kernel)
    add_test(NAME gpu_matmul_test COMMAND yica_gpu_tests matmul)
    add_test(NAME gpu_all_reduce_test COMMAND yica_gpu_tests all_reduce)
    add_test(NAME gpu_optimization_test COMMAND yica_gpu_tests optimization)
else()
    add_test(NAME yica_gpu_tests COMMAND yica_gpu_tests)
endif()

# GPU测试属性
set_tests_properties(gpu_basic_test PROPERTIES
    LABELS "gpu;basic"
    TIMEOUT 120
    REQUIRED_FILES "${CMAKE_CURRENT_BINARY_DIR}/yica_gpu_tests"
)

set_tests_properties(gpu_memory_test PROPERTIES
    LABELS "gpu;memory"
    TIMEOUT 180
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

set_tests_properties(gpu_kernel_test PROPERTIES
    LABELS "gpu;kernel"
    TIMEOUT 240
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

set_tests_properties(gpu_matmul_test PROPERTIES
    LABELS "gpu;matmul"
    TIMEOUT 300
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

set_tests_properties(gpu_all_reduce_test PROPERTIES
    LABELS "gpu;collective"
    TIMEOUT 360
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0,1"  # 多GPU测试
)

set_tests_properties(gpu_optimization_test PROPERTIES
    LABELS "gpu;optimization"
    TIMEOUT 240
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
)

# GPU性能测试（需要特定GPU）
option(BUILD_GPU_PERFORMANCE_TESTS "Build GPU performance tests" OFF)
if(BUILD_GPU_PERFORMANCE_TESTS)
    add_executable(yica_gpu_perf_tests
        test_gpu_performance_detailed.cc
        test_gpu_memory_bandwidth_detailed.cc
        test_gpu_compute_throughput_detailed.cc
        test_gpu_multi_gpu_scaling.cc
        test_gpu_mixed_precision_performance.cc
    )
    
    target_link_libraries(yica_gpu_perf_tests yica_gpu yica_core ${CUDA_LIBRARIES})
    
    if(CUBLAS_LIBRARY)
        target_link_libraries(yica_gpu_perf_tests ${CUBLAS_LIBRARY})
    endif()
    
    target_compile_definitions(yica_gpu_perf_tests PRIVATE 
        YICA_GPU_BACKEND
        YICA_PERFORMANCE_TEST
        USE_CUDA
    )
    
    set_target_properties(yica_gpu_perf_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
    
    add_test(NAME gpu_performance_test COMMAND yica_gpu_perf_tests)
    set_tests_properties(gpu_performance_test PROPERTIES
        LABELS "gpu;performance"
        TIMEOUT 1200
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
endif()

# GPU压力测试（长时间运行）
option(BUILD_GPU_STRESS_TESTS "Build GPU stress tests" OFF)
if(BUILD_GPU_STRESS_TESTS)
    add_executable(yica_gpu_stress_tests
        test_gpu_memory_stress.cc
        test_gpu_thermal_stress.cc
        test_gpu_long_running.cc
    )
    
    target_link_libraries(yica_gpu_stress_tests yica_gpu yica_core ${CUDA_LIBRARIES})
    
    target_compile_definitions(yica_gpu_stress_tests PRIVATE 
        YICA_GPU_BACKEND
        YICA_STRESS_TEST
        USE_CUDA
    )
    
    set_target_properties(yica_gpu_stress_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
    
    add_test(NAME gpu_stress_test COMMAND yica_gpu_stress_tests)
    set_tests_properties(gpu_stress_test PROPERTIES
        LABELS "gpu;stress"
        TIMEOUT 3600  # 1小时
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
endif()

# GPU兼容性测试（不同CUDA版本）
option(BUILD_GPU_COMPATIBILITY_TESTS "Build GPU compatibility tests" OFF)
if(BUILD_GPU_COMPATIBILITY_TESTS)
    # 检测CUDA版本
    execute_process(COMMAND nvcc --version OUTPUT_VARIABLE NVCC_OUTPUT)
    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" CUDA_VERSION_MATCH "${NVCC_OUTPUT}")
    set(DETECTED_CUDA_VERSION ${CMAKE_MATCH_1})
    
    message(STATUS "检测到CUDA版本: ${DETECTED_CUDA_VERSION}")
    
    add_executable(yica_gpu_compat_tests
        test_gpu_cuda_version_compatibility.cc
        test_gpu_driver_compatibility.cc
        test_gpu_architecture_compatibility.cc
    )
    
    target_link_libraries(yica_gpu_compat_tests yica_gpu yica_core ${CUDA_LIBRARIES})
    
    target_compile_definitions(yica_gpu_compat_tests PRIVATE 
        YICA_GPU_BACKEND
        YICA_COMPATIBILITY_TEST
        USE_CUDA
        "DETECTED_CUDA_VERSION=\"${DETECTED_CUDA_VERSION}\""
    )
    
    add_test(NAME gpu_compatibility_test COMMAND yica_gpu_compat_tests)
    set_tests_properties(gpu_compatibility_test PROPERTIES
        LABELS "gpu;compatibility"
        TIMEOUT 300
    )
endif()

# 安装GPU测试可执行文件（可选）
option(INSTALL_TESTS "Install test executables" OFF)
if(INSTALL_TESTS)
    install(TARGETS yica_gpu_tests RUNTIME DESTINATION bin/tests)
    if(BUILD_GPU_PERFORMANCE_TESTS)
        install(TARGETS yica_gpu_perf_tests RUNTIME DESTINATION bin/tests)
    endif()
    if(BUILD_GPU_STRESS_TESTS)
        install(TARGETS yica_gpu_stress_tests RUNTIME DESTINATION bin/tests)
    endif()
    if(BUILD_GPU_COMPATIBILITY_TESTS)
        install(TARGETS yica_gpu_compat_tests RUNTIME DESTINATION bin/tests)
    endif()
endif()

message(STATUS "GPU CUDA后端测试配置完成") 