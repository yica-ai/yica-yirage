# YICA硬件后端测试配置
cmake_minimum_required(VERSION 3.16)
project(YICAHardwareTests)

message(STATUS "配置YICA硬件后端测试")

# 查找YICA SDK
find_path(YICA_INCLUDE_DIR yica/yica.h PATHS /opt/yica/include)
find_library(YICA_LIBRARY yica PATHS /opt/yica/lib)

if(NOT YICA_INCLUDE_DIR OR NOT YICA_LIBRARY)
    message(WARNING "YICA SDK not found, using simulation mode for tests")
    set(YICA_SIMULATION_MODE ON)
else()
    message(STATUS "Found YICA SDK: ${YICA_INCLUDE_DIR}")
    set(YICA_SIMULATION_MODE OFF)
endif()

# 查找测试框架
option(USE_BUILTIN_TEST_FRAMEWORK "Use builtin test framework instead of GTest" ON)

if(USE_BUILTIN_TEST_FRAMEWORK)
    message(STATUS "使用内置测试框架")
    set(USE_BUILTIN_TEST_FRAMEWORK ON)
else()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest未找到，使用内置测试框架")
        set(USE_BUILTIN_TEST_FRAMEWORK ON)
    else()
        message(STATUS "使用GTest测试框架")
        set(USE_BUILTIN_TEST_FRAMEWORK OFF)
    endif()
endif()

# YICA后端测试源文件
set(YICA_TEST_SRCS
    # 基础功能测试
    test_yica_basic.cc
    test_yica_hardware_abstraction.cc
    test_yica_device_manager.cc
    test_yica_memory_manager.cc
    
    # YIS指令测试
    test_yis_instruction_generator.cc
    test_yis_optimizer.cc
    test_yis_execution.cc
    test_yis_verification.cc
    
    # CIM阵列测试
    test_cim_array_optimizer.cc
    test_cim_operations.cc
    test_cim_mapping.cc
    test_cim_scheduling.cc
    
    # SPM测试
    test_spm_optimizer.cc
    test_spm_operations.cc
    test_spm_allocation.cc
    test_spm_coherence.cc
    
    # YICA算子测试
    test_yica_matmul.cc
    test_yica_all_reduce.cc
    test_yica_rms_norm.cc
    test_yica_reduction.cc
    test_yica_element_ops.cc
    
    # 存算一体优化测试
    test_compute_in_memory.cc
    test_energy_optimization.cc
    test_data_locality.cc
    test_bandwidth_optimization.cc
    
    # YCCL通信测试
    test_yccl_operations.cc
    test_yccl_collective.cc
    test_yccl_p2p.cc
    
    # 性能测试
    test_yica_performance.cc
    test_yica_benchmark.cc
    test_yica_power_efficiency.cc
)

# YICA后端测试可执行文件
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_executable(yica_hardware_tests ${YICA_TEST_SRCS} test_main.cc)
    target_compile_definitions(yica_hardware_tests PRIVATE USE_BUILTIN_TEST)
else()
    add_executable(yica_hardware_tests ${YICA_TEST_SRCS})
    target_link_libraries(yica_hardware_tests GTest::GTest GTest::Main)
endif()

# YICA特定编译标志
target_compile_definitions(yica_hardware_tests PRIVATE 
    YICA_HARDWARE_BACKEND
    YICA_TEST_MODE
)

if(YICA_SIMULATION_MODE)
    target_compile_definitions(yica_hardware_tests PRIVATE YICA_SIMULATION_MODE)
    message(STATUS "YICA测试将在模拟模式下运行")
else()
    target_compile_definitions(yica_hardware_tests PRIVATE YICA_HARDWARE_MODE)
    target_include_directories(yica_hardware_tests PRIVATE ${YICA_INCLUDE_DIR})
    target_link_libraries(yica_hardware_tests ${YICA_LIBRARY})
    message(STATUS "YICA测试将在硬件模式下运行")
endif()

# 设置target属性
set_target_properties(yica_hardware_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 启用测试
enable_testing()

# 添加YICA测试
if(USE_BUILTIN_TEST_FRAMEWORK)
    add_test(NAME yica_basic_test COMMAND yica_hardware_tests basic)
    add_test(NAME yica_hardware_test COMMAND yica_hardware_tests hardware)
    add_test(NAME yica_yis_test COMMAND yica_hardware_tests yis)
    add_test(NAME yica_cim_test COMMAND yica_hardware_tests cim)
    add_test(NAME yica_spm_test COMMAND yica_hardware_tests spm)
    add_test(NAME yica_operators_test COMMAND yica_hardware_tests operators)
    add_test(NAME yica_optimization_test COMMAND yica_hardware_tests optimization)
    add_test(NAME yica_communication_test COMMAND yica_hardware_tests communication)
    
    # 设置测试属性
    set_tests_properties(yica_basic_test PROPERTIES
        LABELS "yica;basic"
        TIMEOUT 60
    )

    set_tests_properties(yica_hardware_test PROPERTIES
        LABELS "yica;hardware"
        TIMEOUT 120
    )

    set_tests_properties(yica_yis_test PROPERTIES
        LABELS "yica;yis;instruction"
        TIMEOUT 180
    )

    set_tests_properties(yica_cim_test PROPERTIES
        LABELS "yica;cim;compute_in_memory"
        TIMEOUT 240
    )

    set_tests_properties(yica_spm_test PROPERTIES
        LABELS "yica;spm;memory"
        TIMEOUT 180
    )

    set_tests_properties(yica_operators_test PROPERTIES
        LABELS "yica;operators"
        TIMEOUT 300
    )

    set_tests_properties(yica_optimization_test PROPERTIES
        LABELS "yica;optimization;energy"
        TIMEOUT 240
    )

    set_tests_properties(yica_communication_test PROPERTIES
        LABELS "yica;yccl;communication"
        TIMEOUT 300
    )
else()
    add_test(NAME yica_hardware_tests COMMAND yica_hardware_tests)
endif()

# 如果在硬件模式下，设置硬件访问权限
if(NOT YICA_SIMULATION_MODE)
    if(TEST yica_hardware_test)
        set_tests_properties(yica_hardware_test PROPERTIES
            ENVIRONMENT "YICA_DEVICE_PATH=/dev/yica0"
        )
    endif()
    
    if(TEST yica_cim_test)
        set_tests_properties(yica_cim_test PROPERTIES
            ENVIRONMENT "YICA_DEVICE_PATH=/dev/yica0;YICA_CIM_ARRAYS=4"
        )
    endif()
    
    if(TEST yica_spm_test)
        set_tests_properties(yica_spm_test PROPERTIES
            ENVIRONMENT "YICA_DEVICE_PATH=/dev/yica0;YICA_SPM_SIZE=2147483648"  # 2GB
        )
    endif()
endif()

# YICA性能测试（需要真实硬件）
option(BUILD_YICA_PERFORMANCE_TESTS "Build YICA performance tests" OFF)
if(BUILD_YICA_PERFORMANCE_TESTS)
    add_executable(yica_hardware_perf_tests
        test_yica_performance_detailed.cc
        test_yica_energy_efficiency.cc
        test_yica_throughput_scaling.cc
        test_yica_latency_analysis.cc
        test_yica_memory_bandwidth.cc
        test_yica_cim_utilization.cc
    )
    
    target_compile_definitions(yica_hardware_perf_tests PRIVATE 
        YICA_HARDWARE_BACKEND
        YICA_PERFORMANCE_TEST
    )
    
    if(YICA_SIMULATION_MODE)
        target_compile_definitions(yica_hardware_perf_tests PRIVATE YICA_SIMULATION_MODE)
    else()
        target_compile_definitions(yica_hardware_perf_tests PRIVATE YICA_HARDWARE_MODE)
        target_link_libraries(yica_hardware_perf_tests ${YICA_LIBRARY})
    endif()
    
    add_test(NAME yica_performance_test COMMAND yica_hardware_perf_tests)
    set_tests_properties(yica_performance_test PROPERTIES
        LABELS "yica;performance"
        TIMEOUT 1800  # 30分钟
    )
    
    if(NOT YICA_SIMULATION_MODE)
        set_tests_properties(yica_performance_test PROPERTIES
            ENVIRONMENT "YICA_DEVICE_PATH=/dev/yica0"
        )
    endif()
endif()

# YICA功耗测试（需要特殊权限）
option(BUILD_YICA_POWER_TESTS "Build YICA power measurement tests" OFF)
if(BUILD_YICA_POWER_TESTS)
    # 这些测试文件目前不存在，仅作为示例
    message(STATUS "Power tests are disabled - test files not implemented")
endif()

# YICA可靠性测试（长时间运行）
option(BUILD_YICA_RELIABILITY_TESTS "Build YICA reliability tests" OFF)
if(BUILD_YICA_RELIABILITY_TESTS)
    # 这些测试文件目前不存在，仅作为示例
    message(STATUS "Reliability tests are disabled - test files not implemented")
endif()

# YICA多设备测试（如果有多个YICA设备）
option(BUILD_YICA_MULTI_DEVICE_TESTS "Build YICA multi-device tests" OFF)
if(BUILD_YICA_MULTI_DEVICE_TESTS)
    # 这些测试文件目前不存在，仅作为示例
    message(STATUS "Multi-device tests are disabled - test files not implemented")
endif()

# 安装YICA测试可执行文件（可选）
option(INSTALL_TESTS "Install test executables" OFF)
if(INSTALL_TESTS)
    install(TARGETS yica_hardware_tests RUNTIME DESTINATION bin/tests)
endif()

message(STATUS "YICA硬件后端测试配置完成") 