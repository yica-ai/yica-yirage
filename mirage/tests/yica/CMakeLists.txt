# YICA测试配置

# 设置测试源文件
set(YICA_TEST_SOURCES
    test_yica_analyzer.cc
    test_strategy_library.cc
    test_code_generator.cc
)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/deps/json/include)

# 创建测试可执行文件
add_executable(yica_tests ${YICA_TEST_SOURCES})

# 链接必要的库
target_link_libraries(yica_tests
    gtest
    gtest_main
    pthread
)

# 如果需要链接其他Mirage库
# target_link_libraries(yica_tests mirage_core)

# 添加编译选项
target_compile_options(yica_tests PRIVATE -std=c++17 -Wall -Wextra)

# 添加测试到CTest
add_test(NAME YICATests COMMAND yica_tests) 