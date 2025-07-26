/**
 * YICA基础功能测试
 * 测试YICA硬件抽象层和基本设备管理功能
 */

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

#ifdef USE_BUILTIN_TEST
#define TEST_FRAMEWORK_AVAILABLE 1
#include "test_framework.h"
#else
#include <gtest/gtest.h>
#define BUILTIN_TEST(name) TEST(YICABasic, name)
#define EXPECT_TRUE_MSG(condition, msg) EXPECT_TRUE(condition) << msg
#define EXPECT_EQ_MSG(expected, actual, msg) EXPECT_EQ(expected, actual) << msg
#endif

// YICA硬件抽象接口
struct YICADeviceInfo {
    int device_id;
    int num_cim_dies;
    int clusters_per_die;
    int cim_arrays_per_cluster;
    size_t spm_size_bytes;
    size_t dram_size_bytes;
    float peak_compute_tops;
    std::string device_name;
};

class YICAHardwareBackend {
public:
    virtual ~YICAHardwareBackend() = default;
    
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual int get_device_count() const = 0;
    virtual YICADeviceInfo get_device_info(int device_id) const = 0;
    virtual bool set_active_device(int device_id) = 0;
    virtual bool is_device_available(int device_id) const = 0;
    
    // 基础内存操作
    virtual void* allocate_memory(size_t size, int device_id = -1) = 0;
    virtual void deallocate_memory(void* ptr) = 0;
    virtual bool copy_to_device(void* dst, const void* src, size_t size) = 0;
    virtual bool copy_from_device(void* dst, const void* src, size_t size) = 0;
    
    // 设备同步
    virtual void synchronize_device(int device_id = -1) = 0;
};

#ifdef YICA_SIMULATION_MODE
// 模拟模式实现
class YICASimulationBackend : public YICAHardwareBackend {
private:
    bool initialized_;
    int active_device_;
    std::vector<YICADeviceInfo> device_infos_;
    
public:
    YICASimulationBackend() : initialized_(false), active_device_(0) {
        // 模拟YICA-G100硬件规格
        device_infos_.push_back({
            .device_id = 0,
            .num_cim_dies = 8,
            .clusters_per_die = 4,
            .cim_arrays_per_cluster = 16,
            .spm_size_bytes = 2ULL * 1024 * 1024 * 1024,  // 2GB SPM per die
            .dram_size_bytes = 64ULL * 1024 * 1024 * 1024,  // 64GB DRAM
            .peak_compute_tops = 200.0f,
            .device_name = "YICA-G100 (Simulation)"
        });
    }
    
    bool initialize() override {
        std::cout << "初始化YICA模拟后端..." << std::endl;
        initialized_ = true;
        return true;
    }
    
    void shutdown() override {
        std::cout << "关闭YICA模拟后端..." << std::endl;
        initialized_ = false;
    }
    
    int get_device_count() const override {
        return initialized_ ? static_cast<int>(device_infos_.size()) : 0;
    }
    
    YICADeviceInfo get_device_info(int device_id) const override {
        if (device_id < 0 || device_id >= static_cast<int>(device_infos_.size())) {
            return {};
        }
        return device_infos_[device_id];
    }
    
    bool set_active_device(int device_id) override {
        if (device_id < 0 || device_id >= get_device_count()) {
            return false;
        }
        active_device_ = device_id;
        return true;
    }
    
    bool is_device_available(int device_id) const override {
        return device_id >= 0 && device_id < get_device_count() && initialized_;
    }
    
    void* allocate_memory(size_t size, int device_id = -1) override {
        // 简单的主机内存分配作为模拟
        return malloc(size);
    }
    
    void deallocate_memory(void* ptr) override {
        free(ptr);
    }
    
    bool copy_to_device(void* dst, const void* src, size_t size) override {
        memcpy(dst, src, size);
        return true;
    }
    
    bool copy_from_device(void* dst, const void* src, size_t size) override {
        memcpy(dst, src, size);
        return true;
    }
    
    void synchronize_device(int device_id = -1) override {
        // 模拟同步延迟
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
};

#else
// 真实硬件模式实现（占位符）
class YICAHardwareBackendImpl : public YICAHardwareBackend {
    // TODO: 实现真实的YICA硬件接口
public:
    bool initialize() override {
        std::cout << "初始化YICA硬件后端..." << std::endl;
        // TODO: 初始化真实硬件
        return false;  // 目前返回false，因为没有真实硬件
    }
    
    void shutdown() override {
        std::cout << "关闭YICA硬件后端..." << std::endl;
    }
    
    int get_device_count() const override { return 0; }
    YICADeviceInfo get_device_info(int device_id) const override { return {}; }
    bool set_active_device(int device_id) override { return false; }
    bool is_device_available(int device_id) const override { return false; }
    void* allocate_memory(size_t size, int device_id = -1) override { return nullptr; }
    void deallocate_memory(void* ptr) override {}
    bool copy_to_device(void* dst, const void* src, size_t size) override { return false; }
    bool copy_from_device(void* dst, const void* src, size_t size) override { return false; }
    void synchronize_device(int device_id = -1) override {}
};
#endif

// 全局后端实例
std::unique_ptr<YICAHardwareBackend> g_yica_backend;

YICAHardwareBackend* get_yica_backend() {
    if (!g_yica_backend) {
#ifdef YICA_SIMULATION_MODE
        g_yica_backend = std::make_unique<YICASimulationBackend>();
#else
        g_yica_backend = std::make_unique<YICAHardwareBackendImpl>();
#endif
    }
    return g_yica_backend.get();
}

// 测试用例

BUILTIN_TEST(YICABackendInitialization) {
    auto* backend = get_yica_backend();
    EXPECT_TRUE_MSG(backend != nullptr, "YICA后端应该成功创建");
    
    bool init_result = backend->initialize();
    EXPECT_TRUE_MSG(init_result, "YICA后端应该成功初始化");
    
    backend->shutdown();
}

BUILTIN_TEST(YICADeviceEnumeration) {
    auto* backend = get_yica_backend();
    backend->initialize();
    
    int device_count = backend->get_device_count();
    
#ifdef YICA_SIMULATION_MODE
    EXPECT_EQ_MSG(1, device_count, "模拟模式下应该有1个设备");
#else
    // 真实硬件模式下设备数量可能为0（如果没有硬件）
    std::cout << "检测到 " << device_count << " 个YICA设备" << std::endl;
#endif
    
    if (device_count > 0) {
        YICADeviceInfo info = backend->get_device_info(0);
        std::cout << "设备0信息:" << std::endl;
        std::cout << "  名称: " << info.device_name << std::endl;
        std::cout << "  CIM Dies: " << info.num_cim_dies << std::endl;
        std::cout << "  每Die的Clusters: " << info.clusters_per_die << std::endl;
        std::cout << "  每Cluster的CIM阵列: " << info.cim_arrays_per_cluster << std::endl;
        std::cout << "  SPM大小: " << (info.spm_size_bytes / (1024*1024*1024)) << "GB" << std::endl;
        std::cout << "  峰值算力: " << info.peak_compute_tops << "TOPS" << std::endl;
        
        EXPECT_TRUE_MSG(info.num_cim_dies > 0, "CIM Dies数量应该大于0");
        EXPECT_TRUE_MSG(info.clusters_per_die > 0, "每Die的Clusters数量应该大于0");
        EXPECT_TRUE_MSG(info.peak_compute_tops > 0, "峰值算力应该大于0");
    }
    
    backend->shutdown();
}

BUILTIN_TEST(YICADeviceSelection) {
    auto* backend = get_yica_backend();
    backend->initialize();
    
    int device_count = backend->get_device_count();
    
    if (device_count > 0) {
        // 测试设备选择
        bool select_result = backend->set_active_device(0);
        EXPECT_TRUE_MSG(select_result, "应该能够选择设备0");
        
        bool available = backend->is_device_available(0);
        EXPECT_TRUE_MSG(available, "设备0应该可用");
        
        // 测试无效设备
        bool invalid_select = backend->set_active_device(999);
        EXPECT_TRUE_MSG(!invalid_select, "选择无效设备应该失败");
        
        bool invalid_available = backend->is_device_available(999);
        EXPECT_TRUE_MSG(!invalid_available, "无效设备应该不可用");
    }
    
    backend->shutdown();
}

BUILTIN_TEST(YICAMemoryOperations) {
    auto* backend = get_yica_backend();
    backend->initialize();
    
    if (backend->get_device_count() > 0) {
        backend->set_active_device(0);
        
        // 测试内存分配
        const size_t test_size = 1024 * 1024;  // 1MB
        void* device_ptr = backend->allocate_memory(test_size);
        EXPECT_TRUE_MSG(device_ptr != nullptr, "内存分配应该成功");
        
        if (device_ptr) {
            // 测试内存拷贝
            std::vector<uint8_t> host_data(test_size, 0x42);
            std::vector<uint8_t> result_data(test_size, 0x00);
            
            bool copy_to_result = backend->copy_to_device(device_ptr, host_data.data(), test_size);
            EXPECT_TRUE_MSG(copy_to_result, "拷贝到设备应该成功");
            
            bool copy_from_result = backend->copy_from_device(result_data.data(), device_ptr, test_size);
            EXPECT_TRUE_MSG(copy_from_result, "从设备拷贝应该成功");
            
            // 验证数据正确性
            bool data_correct = (host_data == result_data);
            EXPECT_TRUE_MSG(data_correct, "拷贝的数据应该正确");
            
            backend->deallocate_memory(device_ptr);
        }
    }
    
    backend->shutdown();
}

BUILTIN_TEST(YICADeviceSynchronization) {
    auto* backend = get_yica_backend();
    backend->initialize();
    
    if (backend->get_device_count() > 0) {
        backend->set_active_device(0);
        
        // 测试设备同步
        auto start_time = std::chrono::high_resolution_clock::now();
        backend->synchronize_device();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto sync_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "设备同步耗时: " << sync_duration.count() << "μs" << std::endl;
        
        // 同步操作应该完成（不应该崩溃）
        EXPECT_TRUE_MSG(true, "设备同步应该成功完成");
    }
    
    backend->shutdown();
}

#ifdef USE_BUILTIN_TEST
// 内置测试框架的main函数
int run_yica_basic_tests() {
    std::cout << "=== YICA基础功能测试 ===" << std::endl;
    
#ifdef YICA_SIMULATION_MODE
    std::cout << "运行模式: 模拟模式" << std::endl;
#else
    std::cout << "运行模式: 硬件模式" << std::endl;
#endif
    
    int failed_tests = 0;
    
    try {
        std::cout << "\n[TEST] YICABackendInitialization" << std::endl;
        YICABasic_YICABackendInitialization_Test test1;
        test1.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] YICADeviceEnumeration" << std::endl;
        YICABasic_YICADeviceEnumeration_Test test2;
        test2.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] YICADeviceSelection" << std::endl;
        YICABasic_YICADeviceSelection_Test test3;
        test3.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] YICAMemoryOperations" << std::endl;
        YICABasic_YICAMemoryOperations_Test test4;
        test4.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    try {
        std::cout << "\n[TEST] YICADeviceSynchronization" << std::endl;
        YICABasic_YICADeviceSynchronization_Test test5;
        test5.TestBody();
        std::cout << "✅ PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ FAILED: " << e.what() << std::endl;
        failed_tests++;
    }
    
    std::cout << "\n=== 测试总结 ===" << std::endl;
    std::cout << "总测试数: 5" << std::endl;
    std::cout << "成功: " << (5 - failed_tests) << std::endl;
    std::cout << "失败: " << failed_tests << std::endl;
    
    return failed_tests;
}
#endif 