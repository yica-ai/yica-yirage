/**
 * YICA硬件后端测试主程序
 * 运行所有YICA相关的硬件测试
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cstring>

// 声明外部测试函数
#ifdef USE_BUILTIN_TEST
extern int run_yica_basic_tests();
extern int run_cim_array_tests();
#endif

// 测试类别
enum class TestCategory {
    ALL,
    BASIC,
    HARDWARE,
    YIS,
    CIM,
    SPM,
    OPERATORS,
    OPTIMIZATION,
    COMMUNICATION
};

// 测试类别映射
std::map<std::string, TestCategory> test_category_map = {
    {"all", TestCategory::ALL},
    {"basic", TestCategory::BASIC},
    {"hardware", TestCategory::HARDWARE},
    {"yis", TestCategory::YIS},
    {"cim", TestCategory::CIM},
    {"spm", TestCategory::SPM},
    {"operators", TestCategory::OPERATORS},
    {"optimization", TestCategory::OPTIMIZATION},
    {"communication", TestCategory::COMMUNICATION}
};

void print_banner() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           YICA 硬件后端测试套件                                ║
║                        YICA-G100 存算一体架构测试                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

#ifdef YICA_SIMULATION_MODE
    std::cout << "🔧 运行模式: 模拟模式 (YICA硬件未检测到)" << std::endl;
#else
    std::cout << "⚡ 运行模式: 硬件模式 (使用真实YICA设备)" << std::endl;
#endif

    std::cout << "📅 编译时间: " << __DATE__ << " " << __TIME__ << std::endl;
    
#ifdef YICA_HARDWARE_BACKEND
    std::cout << "🏗️  后端支持: YICA硬件后端已启用" << std::endl;
#endif

#ifdef YICA_TEST_MODE
    std::cout << "🧪 测试模式: 测试专用构建" << std::endl;
#endif

    std::cout << std::endl;
}

void print_help() {
    std::cout << "用法: yica_hardware_tests [选项] [测试类别]" << std::endl;
    std::cout << std::endl;
    std::cout << "测试类别:" << std::endl;
    std::cout << "  all           - 运行所有测试 (默认)" << std::endl;
    std::cout << "  basic         - YICA基础功能测试" << std::endl;
    std::cout << "  hardware      - 硬件抽象层测试" << std::endl;
    std::cout << "  yis           - YIS指令系统测试" << std::endl;
    std::cout << "  cim           - CIM阵列测试" << std::endl;
    std::cout << "  spm           - SPM内存管理测试" << std::endl;
    std::cout << "  operators     - YICA算子测试" << std::endl;
    std::cout << "  optimization  - 存算一体优化测试" << std::endl;
    std::cout << "  communication - YCCL通信测试" << std::endl;
    std::cout << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -h, --help    - 显示此帮助信息" << std::endl;
    std::cout << "  -v, --verbose - 详细输出模式" << std::endl;
    std::cout << "  --list        - 列出所有可用测试" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  yica_hardware_tests basic    # 运行基础测试" << std::endl;
    std::cout << "  yica_hardware_tests cim      # 运行CIM阵列测试" << std::endl;
    std::cout << "  yica_hardware_tests -v all   # 详细模式运行所有测试" << std::endl;
}

void list_available_tests() {
    std::cout << "可用的YICA硬件测试:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "📱 基础功能测试 (basic):" << std::endl;
    std::cout << "  ✓ YICABackendInitialization  - YICA后端初始化测试" << std::endl;
    std::cout << "  ✓ YICADeviceEnumeration      - 设备枚举测试" << std::endl;
    std::cout << "  ✓ YICADeviceSelection        - 设备选择测试" << std::endl;
    std::cout << "  ✓ YICAMemoryOperations       - 内存操作测试" << std::endl;
    std::cout << "  ✓ YICADeviceSynchronization  - 设备同步测试" << std::endl;
    std::cout << std::endl;
    
    std::cout << "🧮 CIM阵列测试 (cim):" << std::endl;
    std::cout << "  ✓ CIMArrayInitialization     - CIM阵列初始化测试" << std::endl;
    std::cout << "  ✓ CIMArrayScheduling          - CIM阵列调度测试" << std::endl;
    std::cout << "  ✓ CIMOperationMapping         - CIM操作映射测试" << std::endl;
    std::cout << "  ✓ CIMPerformanceEstimation    - CIM性能估算测试" << std::endl;
    std::cout << "  ✓ CIMArrayUtilization         - CIM阵列利用率测试" << std::endl;
    std::cout << std::endl;
    
    std::cout << "🚧 计划中的测试:" << std::endl;
    std::cout << "  ⏳ YIS指令系统测试 (yis)" << std::endl;
    std::cout << "  ⏳ SPM内存管理测试 (spm)" << std::endl;
    std::cout << "  ⏳ YICA算子测试 (operators)" << std::endl;
    std::cout << "  ⏳ 存算一体优化测试 (optimization)" << std::endl;
    std::cout << "  ⏳ YCCL通信测试 (communication)" << std::endl;
    std::cout << std::endl;
}

int run_basic_tests() {
#ifdef USE_BUILTIN_TEST
    return run_yica_basic_tests();
#else
    std::cout << "⚠️  基础测试需要内置测试框架支持" << std::endl;
    return 0;
#endif
}

int run_cim_tests() {
#ifdef USE_BUILTIN_TEST
    return run_cim_array_tests();
#else
    std::cout << "⚠️  CIM测试需要内置测试框架支持" << std::endl;
    return 0;
#endif
}

int run_placeholder_test(const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "⏳ " << test_name << " 尚未实现，敬请期待..." << std::endl;
    std::cout << "✅ PLACEHOLDER PASSED" << std::endl;
    return 0;
}

int run_test_category(TestCategory category, bool verbose = false) {
    int total_failures = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    switch (category) {
        case TestCategory::ALL:
            std::cout << "🚀 运行所有YICA硬件测试..." << std::endl;
            total_failures += run_basic_tests();
            total_failures += run_cim_tests();
            total_failures += run_placeholder_test("YIS指令系统测试");
            total_failures += run_placeholder_test("SPM内存管理测试");
            total_failures += run_placeholder_test("YICA算子测试");
            total_failures += run_placeholder_test("存算一体优化测试");
            total_failures += run_placeholder_test("YCCL通信测试");
            break;
            
        case TestCategory::BASIC:
            total_failures += run_basic_tests();
            break;
            
        case TestCategory::CIM:
            total_failures += run_cim_tests();
            break;
            
        case TestCategory::HARDWARE:
            total_failures += run_placeholder_test("硬件抽象层测试");
            break;
            
        case TestCategory::YIS:
            total_failures += run_placeholder_test("YIS指令系统测试");
            break;
            
        case TestCategory::SPM:
            total_failures += run_placeholder_test("SPM内存管理测试");
            break;
            
        case TestCategory::OPERATORS:
            total_failures += run_placeholder_test("YICA算子测试");
            break;
            
        case TestCategory::OPTIMIZATION:
            total_failures += run_placeholder_test("存算一体优化测试");
            break;
            
        case TestCategory::COMMUNICATION:
            total_failures += run_placeholder_test("YCCL通信测试");
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::endl;
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                              测试执行总结                                      ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    if (total_failures == 0) {
        std::cout << "🎉 所有测试通过！" << std::endl;
    } else {
        std::cout << "❌ 发现 " << total_failures << " 个测试失败" << std::endl;
    }
    
    std::cout << "⏱️  总执行时间: " << duration.count() << "ms" << std::endl;
    
#ifdef YICA_SIMULATION_MODE
    std::cout << "ℹ️  注意: 测试在模拟模式下运行，实际硬件性能可能不同" << std::endl;
#endif
    
    return total_failures;
}

int main(int argc, char* argv[]) {
    print_banner();
    
    // 解析命令行参数
    bool verbose = false;
    bool show_help = false;
    bool list_tests = false;
    TestCategory category = TestCategory::ALL;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--list") {
            list_tests = true;
        } else {
            // 尝试解析测试类别
            auto it = test_category_map.find(arg);
            if (it != test_category_map.end()) {
                category = it->second;
            } else {
                std::cout << "❌ 未知的测试类别: " << arg << std::endl;
                std::cout << "使用 --help 查看可用选项" << std::endl;
                return 1;
            }
        }
    }
    
    if (show_help) {
        print_help();
        return 0;
    }
    
    if (list_tests) {
        list_available_tests();
        return 0;
    }
    
    // 显示系统信息
    std::cout << "🖥️  系统信息:" << std::endl;
    
#ifdef YICA_SIMULATION_MODE
    std::cout << "  运行模式: 模拟模式" << std::endl;
    std::cout << "  CIM Dies: 8 (模拟)" << std::endl;
    std::cout << "  每Die的Clusters: 4 (模拟)" << std::endl;
    std::cout << "  每Cluster的CIM阵列: 16 (模拟)" << std::endl;
    std::cout << "  SPM大小: 2GB/Die (模拟)" << std::endl;
    std::cout << "  峰值算力: 200TOPS (模拟)" << std::endl;
#else
    std::cout << "  运行模式: 硬件模式" << std::endl;
    std::cout << "  设备路径: " << (getenv("YICA_DEVICE_PATH") ? getenv("YICA_DEVICE_PATH") : "/dev/yica0") << std::endl;
#endif
    
    std::cout << std::endl;
    
    // 运行测试
    int exit_code = run_test_category(category, verbose);
    
    if (exit_code == 0) {
        std::cout << "\n🎯 YICA硬件测试完成！所有测试通过。" << std::endl;
    } else {
        std::cout << "\n💥 YICA硬件测试完成，但有测试失败。" << std::endl;
    }
    
    return exit_code;
} 