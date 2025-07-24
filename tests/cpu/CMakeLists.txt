# CPU后端测试配置
cmake_minimum_required(VERSION 3.16)

message(STATUS "配置CPU后端测试")

# 查找测试框架
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # 如果没有找到GTest，使用内置的简单测试框架
    message(STATUS "GTest未找到，使用内置测试框架")
    set(USE_BUILTIN_TEST_FRAMEWORK ON)
else()
    message(STATUS "使用GTest测试框架")
    set(USE_BUILTIN_TEST_FRAMEWORK OFF)
endif()

# CPU后端测试源文件
set(CPU_TEST_SRCS
    # 基础功能测试
    test_cpu_basic.cc
    test_cpu_memory_manager.cc
    test_cpu_openmp_parallel.cc
    
    # 算子测试
    test_cpu_matmul.cc
    test_cpu_all_reduce.cc
    test_cpu_rms_norm.cc
    test_cpu_reduction.cc
    test_cpu_element_ops.cc
    
    # 优化测试
    test_cpu_simd_optimization.cc
    test_cpu_cache_optimization.cc
    test_cpu_code_generation.cc
    
    # 性能测试
    test_cpu_performance.cc
    test_cpu_benchmark.cc
)

# CPU后端测试可执行文件
if(USE_BUILTIN_TEST_FRAMEWORK)
    # 使用内置测试框架
    add_executable(yica_cpu_tests ${CPU_TEST_SRCS} test_main.cc)
    target_compile_definitions(yica_cpu_tests PRIVATE USE_BUILTIN_TEST)
else()
    # 使用GTest
    add_executable(yica_cpu_tests ${CPU_TEST_SRCS})
    target_link_libraries(yica_cpu_tests GTest::GTest GTest::Main)
endif()

# 链接CPU后端库
target_link_libraries(yica_cpu_tests yica_cpu yica_core)

# CPU特定编译标志
target_compile_definitions(yica_cpu_tests PRIVATE 
    YICA_CPU_BACKEND
    NO_CUDA
    YICA_TEST_MODE
)

# OpenMP支持（可选）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yica_cpu_tests OpenMP::OpenMP_CXX)
    message(STATUS "CPU测试将支持OpenMP并行化")
else()
    message(STATUS "CPU测试将在单线程模式下运行")
endif()

# 包含目录
target_include_directories(yica_cpu_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/mirage/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CPU测试配置
set_target_properties(yica_cpu_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 添加测试
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_test(NAME cpu_basic_test COMMAND yica_cpu_tests basic)
    add_test(NAME cpu_memory_test COMMAND yica_cpu_tests memory)
    add_test(NAME cpu_parallel_test COMMAND yica_cpu_tests parallel)
    add_test(NAME cpu_matmul_test COMMAND yica_cpu_tests matmul)
    add_test(NAME cpu_all_reduce_test COMMAND yica_cpu_tests all_reduce)
    add_test(NAME cpu_optimization_test COMMAND yica_cpu_tests optimization)
    
    # 测试属性（仅在使用内置测试框架时设置）
    set_tests_properties(cpu_basic_test PROPERTIES
        LABELS "cpu;basic"
        TIMEOUT 60
    )
    
    set_tests_properties(cpu_memory_test PROPERTIES
        LABELS "cpu;memory"
        TIMEOUT 120
    )
    
    set_tests_properties(cpu_parallel_test PROPERTIES
        LABELS "cpu;parallel"
        TIMEOUT 180
    )
    
    set_tests_properties(cpu_matmul_test PROPERTIES
        LABELS "cpu;matmul"
        TIMEOUT 300
    )
    
    set_tests_properties(cpu_all_reduce_test PROPERTIES
        LABELS "cpu;collective"
        TIMEOUT 240
    )
    
    set_tests_properties(cpu_optimization_test PROPERTIES
        LABELS "cpu;optimization"
        TIMEOUT 180
    )
else()
    add_test(NAME yica_cpu_tests COMMAND yica_cpu_tests)
    
    # GTest测试属性
    set_tests_properties(yica_cpu_tests PROPERTIES
        LABELS "cpu;gtest"
        TIMEOUT 600
    )
endif()

# CPU性能测试（可选）
option(BUILD_CPU_PERFORMANCE_TESTS "Build CPU performance tests" OFF)
if(BUILD_CPU_PERFORMANCE_TESTS)
    add_executable(yica_cpu_perf_tests
        test_cpu_performance_detailed.cc
        test_cpu_memory_bandwidth.cc
        test_cpu_compute_throughput.cc
    )
    
    target_link_libraries(yica_cpu_perf_tests yica_cpu yica_core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(yica_cpu_perf_tests OpenMP::OpenMP_CXX)
    endif()
    
    target_compile_definitions(yica_cpu_perf_tests PRIVATE 
        YICA_CPU_BACKEND
        YICA_PERFORMANCE_TEST
    )
    
    add_test(NAME cpu_performance_test COMMAND yica_cpu_perf_tests)
    set_tests_properties(cpu_performance_test PROPERTIES
        LABELS "cpu;performance"
        TIMEOUT 600
    )
endif()

# 安装测试可执行文件（可选）
option(INSTALL_TESTS "Install test executables" OFF)
if(INSTALL_TESTS)
    install(TARGETS yica_cpu_tests RUNTIME DESTINATION bin/tests)
    if(BUILD_CPU_PERFORMANCE_TESTS)
        install(TARGETS yica_cpu_perf_tests RUNTIME DESTINATION bin/tests)
    endif()
endif()

message(STATUS "CPU后端测试配置完成") 