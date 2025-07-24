# 混合多后端测试配置
cmake_minimum_required(VERSION 3.16)

message(STATUS "配置混合多后端测试")

# 检查启用的后端
set(ENABLED_BACKENDS "")
if(BUILD_CPU_BACKEND)
    list(APPEND ENABLED_BACKENDS "CPU")
endif()
if(BUILD_GPU_BACKEND)
    list(APPEND ENABLED_BACKENDS "GPU")
endif()
if(BUILD_YICA_BACKEND)
    list(APPEND ENABLED_BACKENDS "YICA")
endif()

message(STATUS "混合测试将覆盖后端: ${ENABLED_BACKENDS}")

if(LENGTH ENABLED_BACKENDS LESS 2)
    message(FATAL_ERROR "混合后端测试需要至少启用两个硬件后端")
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

# 混合后端测试源文件
set(HYBRID_TEST_SRCS
    # 基础功能测试
    test_hybrid_basic.cc
    test_backend_manager.cc
    test_backend_selector.cc
    test_performance_monitor.cc
    
    # 自动调度器测试
    test_auto_scheduler.cc
    test_load_balancer.cc
    test_workload_distribution.cc
    test_dynamic_scheduling.cc
    
    # 跨后端优化测试
    test_cross_backend_optimizer.cc
    test_memory_coordinator.cc
    test_data_migration.cc
    test_computation_offloading.cc
    
    # 后端切换测试
    test_backend_switching.cc
    test_runtime_adaptation.cc
    test_fallback_mechanisms.cc
    test_error_recovery.cc
    
    # 混合算子测试
    test_hybrid_matmul.cc
    test_hybrid_all_reduce.cc
    test_hybrid_rms_norm.cc
    test_hybrid_reduction.cc
    test_hybrid_element_ops.cc
    
    # 性能对比测试
    test_backend_performance_comparison.cc
    test_energy_efficiency_comparison.cc
    test_scalability_analysis.cc
    
    # 集成测试
    test_hybrid_integration.cc
    test_end_to_end_workflows.cc
    test_multi_backend_coordination.cc
)

# 混合后端测试可执行文件
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_executable(yica_hybrid_tests ${HYBRID_TEST_SRCS} test_main.cc)
    target_compile_definitions(yica_hybrid_tests PRIVATE USE_BUILTIN_TEST)
else()
    add_executable(yica_hybrid_tests ${HYBRID_TEST_SRCS})
    target_link_libraries(yica_hybrid_tests GTest::GTest GTest::Main)
endif()

# 链接混合后端库和核心库
target_link_libraries(yica_hybrid_tests yica_hybrid yica_core)

# 根据启用的后端链接对应库
if(BUILD_CPU_BACKEND)
    target_link_libraries(yica_hybrid_tests yica_cpu)
    target_compile_definitions(yica_hybrid_tests PRIVATE HYBRID_HAS_CPU)
endif()

if(BUILD_GPU_BACKEND)
    target_link_libraries(yica_hybrid_tests yica_gpu)
    target_compile_definitions(yica_hybrid_tests PRIVATE HYBRID_HAS_GPU)
    
    # 如果有GPU后端，需要CUDA支持
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    target_link_libraries(yica_hybrid_tests ${CUDA_LIBRARIES})
endif()

if(BUILD_YICA_BACKEND)
    target_link_libraries(yica_hybrid_tests yica_hardware)
    target_compile_definitions(yica_hybrid_tests PRIVATE HYBRID_HAS_YICA)
    
    # 检查YICA SDK
    find_path(YICA_INCLUDE_DIR yica/yica.h PATHS /opt/yica/include)
    find_library(YICA_LIBRARY yica PATHS /opt/yica/lib)
    
    if(YICA_INCLUDE_DIR AND YICA_LIBRARY)
        target_include_directories(yica_hybrid_tests PRIVATE ${YICA_INCLUDE_DIR})
        target_link_libraries(yica_hybrid_tests ${YICA_LIBRARY})
        target_compile_definitions(yica_hybrid_tests PRIVATE YICA_HARDWARE_MODE)
    else()
        target_compile_definitions(yica_hybrid_tests PRIVATE YICA_SIMULATION_MODE)
    endif()
endif()

# 混合后端特定编译标志
target_compile_definitions(yica_hybrid_tests PRIVATE 
    YICA_HYBRID_BACKEND
    YICA_TEST_MODE
)

# 包含目录
target_include_directories(yica_hybrid_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/mirage/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 混合测试配置
set_target_properties(yica_hybrid_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 如果有GPU后端，设置CUDA属性
if(BUILD_GPU_BACKEND)
    set_target_properties(yica_hybrid_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
endif()

# 添加混合测试
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_test(NAME hybrid_basic_test COMMAND yica_hybrid_tests basic)
    add_test(NAME hybrid_backend_manager_test COMMAND yica_hybrid_tests backend_manager)
    add_test(NAME hybrid_scheduler_test COMMAND yica_hybrid_tests scheduler)
    add_test(NAME hybrid_optimizer_test COMMAND yica_hybrid_tests optimizer)
    add_test(NAME hybrid_switching_test COMMAND yica_hybrid_tests switching)
    add_test(NAME hybrid_operators_test COMMAND yica_hybrid_tests operators)
    add_test(NAME hybrid_performance_test COMMAND yica_hybrid_tests performance)
    add_test(NAME hybrid_integration_test COMMAND yica_hybrid_tests integration)
else()
    add_test(NAME yica_hybrid_tests COMMAND yica_hybrid_tests)
endif()

# 混合测试属性
set_tests_properties(hybrid_basic_test PROPERTIES
    LABELS "hybrid;basic"
    TIMEOUT 120
)

set_tests_properties(hybrid_backend_manager_test PROPERTIES
    LABELS "hybrid;backend_manager"
    TIMEOUT 180
)

set_tests_properties(hybrid_scheduler_test PROPERTIES
    LABELS "hybrid;scheduler;auto"
    TIMEOUT 240
)

set_tests_properties(hybrid_optimizer_test PROPERTIES
    LABELS "hybrid;optimizer;cross_backend"
    TIMEOUT 300
)

set_tests_properties(hybrid_switching_test PROPERTIES
    LABELS "hybrid;switching;adaptation"
    TIMEOUT 180
)

set_tests_properties(hybrid_operators_test PROPERTIES
    LABELS "hybrid;operators"
    TIMEOUT 360
)

set_tests_properties(hybrid_performance_test PROPERTIES
    LABELS "hybrid;performance;comparison"
    TIMEOUT 600
)

set_tests_properties(hybrid_integration_test PROPERTIES
    LABELS "hybrid;integration;end_to_end"
    TIMEOUT 480
)

# 根据启用的后端设置环境变量
if(BUILD_GPU_BACKEND)
    set_tests_properties(hybrid_basic_test PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
    set_tests_properties(hybrid_performance_test PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
    )
endif()

if(BUILD_YICA_BACKEND)
    set_tests_properties(hybrid_basic_test PROPERTIES
        ENVIRONMENT "YICA_DEVICE_PATH=/dev/yica0"
    )
endif()

# 混合性能基准测试（详细对比）
option(BUILD_HYBRID_BENCHMARK_TESTS "Build hybrid benchmark tests" OFF)
if(BUILD_HYBRID_BENCHMARK_TESTS)
    add_executable(yica_hybrid_benchmark_tests
        test_hybrid_benchmark_detailed.cc
        test_backend_performance_matrix.cc
        test_workload_characterization.cc
        test_optimal_backend_selection.cc
        test_energy_performance_tradeoff.cc
        test_scalability_across_backends.cc
    )
    
    target_link_libraries(yica_hybrid_benchmark_tests yica_hybrid yica_core)
    
    # 链接所有启用的后端
    if(BUILD_CPU_BACKEND)
        target_link_libraries(yica_hybrid_benchmark_tests yica_cpu)
    endif()
    if(BUILD_GPU_BACKEND)
        target_link_libraries(yica_hybrid_benchmark_tests yica_gpu ${CUDA_LIBRARIES})
    endif()
    if(BUILD_YICA_BACKEND)
        target_link_libraries(yica_hybrid_benchmark_tests yica_hardware)
        if(YICA_LIBRARY)
            target_link_libraries(yica_hybrid_benchmark_tests ${YICA_LIBRARY})
        endif()
    endif()
    
    target_compile_definitions(yica_hybrid_benchmark_tests PRIVATE 
        YICA_HYBRID_BACKEND
        YICA_BENCHMARK_TEST
    )
    
    # 复制编译定义
    if(BUILD_CPU_BACKEND)
        target_compile_definitions(yica_hybrid_benchmark_tests PRIVATE HYBRID_HAS_CPU)
    endif()
    if(BUILD_GPU_BACKEND)
        target_compile_definitions(yica_hybrid_benchmark_tests PRIVATE HYBRID_HAS_GPU)
    endif()
    if(BUILD_YICA_BACKEND)
        target_compile_definitions(yica_hybrid_benchmark_tests PRIVATE HYBRID_HAS_YICA)
    endif()
    
    add_test(NAME hybrid_benchmark_test COMMAND yica_hybrid_benchmark_tests)
    set_tests_properties(hybrid_benchmark_test PROPERTIES
        LABELS "hybrid;benchmark;detailed"
        TIMEOUT 1800  # 30分钟
    )
endif()

# 混合压力测试（长时间运行）
option(BUILD_HYBRID_STRESS_TESTS "Build hybrid stress tests" OFF)
if(BUILD_HYBRID_STRESS_TESTS)
    add_executable(yica_hybrid_stress_tests
        test_hybrid_long_running.cc
        test_backend_switching_stress.cc
        test_memory_pressure_handling.cc
        test_concurrent_workloads.cc
        test_fault_injection.cc
    )
    
    target_link_libraries(yica_hybrid_stress_tests yica_hybrid yica_core)
    
    # 链接所有启用的后端
    if(BUILD_CPU_BACKEND)
        target_link_libraries(yica_hybrid_stress_tests yica_cpu)
    endif()
    if(BUILD_GPU_BACKEND)
        target_link_libraries(yica_hybrid_stress_tests yica_gpu ${CUDA_LIBRARIES})
    endif()
    if(BUILD_YICA_BACKEND)
        target_link_libraries(yica_hybrid_stress_tests yica_hardware)
        if(YICA_LIBRARY)
            target_link_libraries(yica_hybrid_stress_tests ${YICA_LIBRARY})
        endif()
    endif()
    
    target_compile_definitions(yica_hybrid_stress_tests PRIVATE 
        YICA_HYBRID_BACKEND
        YICA_STRESS_TEST
    )
    
    add_test(NAME hybrid_stress_test COMMAND yica_hybrid_stress_tests)
    set_tests_properties(hybrid_stress_test PROPERTIES
        LABELS "hybrid;stress;long_running"
        TIMEOUT 7200  # 2小时
    )
endif()

# 混合兼容性测试（不同后端组合）
option(BUILD_HYBRID_COMPATIBILITY_TESTS "Build hybrid compatibility tests" OFF)
if(BUILD_HYBRID_COMPATIBILITY_TESTS)
    add_executable(yica_hybrid_compat_tests
        test_backend_combinations.cc
        test_version_compatibility.cc
        test_driver_compatibility.cc
        test_cross_platform_compatibility.cc
    )
    
    target_link_libraries(yica_hybrid_compat_tests yica_hybrid yica_core)
    
    # 链接所有启用的后端
    if(BUILD_CPU_BACKEND)
        target_link_libraries(yica_hybrid_compat_tests yica_cpu)
    endif()
    if(BUILD_GPU_BACKEND)
        target_link_libraries(yica_hybrid_compat_tests yica_gpu ${CUDA_LIBRARIES})
    endif()
    if(BUILD_YICA_BACKEND)
        target_link_libraries(yica_hybrid_compat_tests yica_hardware)
        if(YICA_LIBRARY)
            target_link_libraries(yica_hybrid_compat_tests ${YICA_LIBRARY})
        endif()
    endif()
    
    target_compile_definitions(yica_hybrid_compat_tests PRIVATE 
        YICA_HYBRID_BACKEND
        YICA_COMPATIBILITY_TEST
    )
    
    add_test(NAME hybrid_compatibility_test COMMAND yica_hybrid_compat_tests)
    set_tests_properties(hybrid_compatibility_test PROPERTIES
        LABELS "hybrid;compatibility"
        TIMEOUT 600
    )
endif()

# 安装混合测试可执行文件（可选）
option(INSTALL_TESTS "Install test executables" OFF)
if(INSTALL_TESTS)
    install(TARGETS yica_hybrid_tests RUNTIME DESTINATION bin/tests)
    if(BUILD_HYBRID_BENCHMARK_TESTS)
        install(TARGETS yica_hybrid_benchmark_tests RUNTIME DESTINATION bin/tests)
    endif()
    if(BUILD_HYBRID_STRESS_TESTS)
        install(TARGETS yica_hybrid_stress_tests RUNTIME DESTINATION bin/tests)
    endif()
    if(BUILD_HYBRID_COMPATIBILITY_TESTS)
        install(TARGETS yica_hybrid_compat_tests RUNTIME DESTINATION bin/tests)
    endif()
endif()

message(STATUS "混合多后端测试配置完成") 