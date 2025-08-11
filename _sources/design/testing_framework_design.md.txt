# Comprehensive Testing Framework Design

## Current Testing Gaps

### 1. Insufficient Test Coverage
- **Limited unit tests**: Only basic functionality tested
- **No integration tests**: Components not tested together
- **Missing performance tests**: No automated performance regression detection
- **No compatibility testing**: Compatibility layers not systematically tested

### 2. Test Infrastructure Problems
- **Manual test execution**: No automated CI/CD testing pipeline
- **Environment dependencies**: Tests fail in different environments
- **Poor test isolation**: Tests interfere with each other
- **Inconsistent test data**: No standardized test datasets

### 3. Quality Assurance Gaps
- **No stress testing**: System behavior under load not tested
- **Missing failure scenario testing**: Error handling paths not covered
- **No security testing**: Potential vulnerabilities not identified
- **Insufficient documentation testing**: Examples and documentation not validated

## Comprehensive Testing Framework Design

### 1. Multi-Level Testing Architecture

```cpp
// yirage/include/yirage/testing/testing_framework.h
#ifndef YIRAGE_TESTING_FRAMEWORK_H
#define YIRAGE_TESTING_FRAMEWORK_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <optional>

namespace yirage {
namespace testing {

    // Test result and reporting
    enum class TestResult {
        PASSED,
        FAILED,
        SKIPPED,
        ERROR,
        TIMEOUT
    };
    
    struct TestMetrics {
        std::chrono::milliseconds execution_time{0};
        size_t memory_used_mb = 0;
        double cpu_usage_percent = 0.0;
        size_t assertions_checked = 0;
        std::map<std::string, double> custom_metrics;
    };
    
    struct TestOutcome {
        TestResult result = TestResult::FAILED;
        std::string message;
        std::string error_details;
        TestMetrics metrics;
        std::vector<std::string> log_messages;
        std::map<std::string, std::string> metadata;
        
        std::string to_json() const;
        std::string to_summary() const;
    };
    
    // Base test case class
    class TestCase {
    private:
        std::string name_;
        std::string description_;
        std::vector<std::string> tags_;
        std::chrono::milliseconds timeout_{30000}; // 30 seconds default
        
    protected:
        TestOutcome outcome_;
        std::shared_ptr<Logger> logger_;
        
    public:
        TestCase(const std::string& name, const std::string& description = "");
        virtual ~TestCase() = default;
        
        // Test lifecycle
        virtual void setup() {}
        virtual void teardown() {}
        virtual void run() = 0;
        
        // Configuration
        void set_timeout(std::chrono::milliseconds timeout) { timeout_ = timeout; }
        void add_tag(const std::string& tag) { tags_.push_back(tag); }
        void set_description(const std::string& desc) { description_ = desc; }
        
        // Execution
        TestOutcome execute();
        
        // Accessors
        const std::string& get_name() const { return name_; }
        const std::string& get_description() const { return description_; }
        const std::vector<std::string>& get_tags() const { return tags_; }
        std::chrono::milliseconds get_timeout() const { return timeout_; }
        
    protected:
        // Assertion helpers
        void assert_true(bool condition, const std::string& message = "");
        void assert_false(bool condition, const std::string& message = "");
        void assert_equals(const std::string& expected, const std::string& actual, const std::string& message = "");
        template<typename T>
        void assert_equals(const T& expected, const T& actual, const std::string& message = "");
        void assert_not_null(const void* ptr, const std::string& message = "");
        void assert_throws(std::function<void()> func, const std::string& message = "");
        template<typename ExceptionType>
        void assert_throws(std::function<void()> func, const std::string& message = "");
        
        // Performance assertions
        void assert_execution_time_less_than(std::chrono::milliseconds max_time);
        void assert_memory_usage_less_than(size_t max_memory_mb);
        
        // Custom metrics
        void record_metric(const std::string& name, double value);
        void add_metadata(const std::string& key, const std::string& value);
        
        // Test utilities
        void skip_test(const std::string& reason = "");
        void fail_test(const std::string& message = "");
        void log_info(const std::string& message);
        
    private:
        void start_metrics_collection();
        void stop_metrics_collection();
        TestMetrics collect_metrics();
    };
    
    // Specialized test case types
    class UnitTest : public TestCase {
    public:
        UnitTest(const std::string& name, const std::string& description = "")
            : TestCase(name, description) {
            add_tag("unit");
        }
    };
    
    class IntegrationTest : public TestCase {
    public:
        IntegrationTest(const std::string& name, const std::string& description = "")
            : TestCase(name, description) {
            add_tag("integration");
            set_timeout(std::chrono::minutes(5)); // Longer timeout
        }
    };
    
    class PerformanceTest : public TestCase {
    private:
        std::map<std::string, double> performance_thresholds_;
        
    public:
        PerformanceTest(const std::string& name, const std::string& description = "")
            : TestCase(name, description) {
            add_tag("performance");
            set_timeout(std::chrono::minutes(10)); // Even longer timeout
        }
        
        void set_performance_threshold(const std::string& metric, double threshold);
        void assert_performance_threshold(const std::string& metric, double value);
        
    protected:
        void run() override;
        virtual void run_performance_test() = 0;
    };
    
    class StressTest : public TestCase {
    private:
        int load_multiplier_ = 10;
        std::chrono::minutes duration_{5};
        
    public:
        StressTest(const std::string& name, const std::string& description = "")
            : TestCase(name, description) {
            add_tag("stress");
            set_timeout(std::chrono::minutes(30));
        }
        
        void set_load_multiplier(int multiplier) { load_multiplier_ = multiplier; }
        void set_duration(std::chrono::minutes duration) { duration_ = duration; }
        
    protected:
        int get_load_multiplier() const { return load_multiplier_; }
        std::chrono::minutes get_duration() const { return duration_; }
    };
    
    // Test suite management
    class TestSuite {
    private:
        std::string name_;
        std::vector<std::unique_ptr<TestCase>> test_cases_;
        std::shared_ptr<Logger> logger_;
        
    public:
        TestSuite(const std::string& name);
        
        // Test management
        void add_test(std::unique_ptr<TestCase> test_case);
        template<typename TestType, typename... Args>
        void add_test(Args&&... args);
        
        // Execution
        std::vector<TestOutcome> run_all();
        std::vector<TestOutcome> run_tagged(const std::vector<std::string>& tags);
        std::vector<TestOutcome> run_pattern(const std::string& name_pattern);
        
        // Reporting
        void generate_report(const std::vector<TestOutcome>& outcomes, const std::string& format = "text");
        void generate_coverage_report();
        
        // Accessors
        const std::string& get_name() const { return name_; }
        size_t get_test_count() const { return test_cases_.size(); }
        std::vector<std::string> get_test_names() const;
    };
    
    // Test registry and discovery
    class TestRegistry {
    private:
        static std::map<std::string, std::function<std::unique_ptr<TestCase>()>> test_factories_;
        static std::map<std::string, std::unique_ptr<TestSuite>> test_suites_;
        
    public:
        // Registration
        template<typename TestType>
        static void register_test(const std::string& name);
        static void register_test_suite(std::unique_ptr<TestSuite> suite);
        
        // Discovery
        static std::vector<std::string> discover_tests();
        static std::vector<std::string> discover_test_suites();
        
        // Execution
        static std::vector<TestOutcome> run_tests(const std::vector<std::string>& test_names);
        static std::vector<TestOutcome> run_test_suite(const std::string& suite_name);
        static std::vector<TestOutcome> run_all_tests();
        
        // Filtering
        static std::vector<TestOutcome> run_tests_with_tags(const std::vector<std::string>& tags);
        static std::vector<TestOutcome> run_tests_matching(const std::string& pattern);
    };
    
    // Test data management
    class TestDataManager {
    private:
        std::string test_data_dir_;
        std::map<std::string, std::vector<uint8_t>> cached_data_;
        
    public:
        TestDataManager(const std::string& data_directory = "tests/data");
        
        // Data loading
        std::vector<uint8_t> load_binary_data(const std::string& filename);
        std::string load_text_data(const std::string& filename);
        std::map<std::string, std::string> load_json_data(const std::string& filename);
        
        // Data generation
        std::vector<uint8_t> generate_random_data(size_t size);
        std::string generate_random_code(const std::string& language, size_t lines);
        std::vector<float> generate_test_matrix(size_t rows, size_t cols);
        
        // Data validation
        bool validate_test_data_integrity();
        void cleanup_temporary_data();
    };
    
    // Mock and fixture support
    template<typename T>
    class MockObject {
    private:
        std::map<std::string, std::function<void()>> method_expectations_;
        std::map<std::string, int> call_counts_;
        
    public:
        void expect_call(const std::string& method_name, int expected_calls = 1);
        void record_call(const std::string& method_name);
        bool verify_expectations() const;
        void reset_expectations();
        int get_call_count(const std::string& method_name) const;
    };
    
    // Test fixtures for common setup/teardown
    class YirageTestFixture {
    protected:
        std::shared_ptr<ConfigurationManager> config_manager_;
        std::shared_ptr<Logger> logger_;
        std::unique_ptr<TestDataManager> data_manager_;
        
    public:
        virtual void setup();
        virtual void teardown();
        
        // Common test utilities
        std::string create_temp_directory();
        void cleanup_temp_directory(const std::string& dir);
        std::string generate_test_config();
        void setup_minimal_environment();
    };

}} // namespace yirage::testing

// Test registration macros
#define YIRAGE_TEST(test_class, test_name) \
    class test_class : public yirage::testing::UnitTest { \
    public: \
        test_class() : UnitTest(test_name) {} \
        void run() override; \
    }; \
    static auto test_class##_registered = []() { \
        yirage::testing::TestRegistry::register_test<test_class>(test_name); \
        return true; \
    }(); \
    void test_class::run()

#define YIRAGE_INTEGRATION_TEST(test_class, test_name) \
    class test_class : public yirage::testing::IntegrationTest { \
    public: \
        test_class() : IntegrationTest(test_name) {} \
        void run() override; \
    }; \
    static auto test_class##_registered = []() { \
        yirage::testing::TestRegistry::register_test<test_class>(test_name); \
        return true; \
    }(); \
    void test_class::run()

#define YIRAGE_PERFORMANCE_TEST(test_class, test_name) \
    class test_class : public yirage::testing::PerformanceTest { \
    public: \
        test_class() : PerformanceTest(test_name) {} \
        void run_performance_test() override; \
    }; \
    static auto test_class##_registered = []() { \
        yirage::testing::TestRegistry::register_test<test_class>(test_name); \
        return true; \
    }(); \
    void test_class::run_performance_test()

#define ASSERT_TRUE(condition) assert_true(condition, #condition)
#define ASSERT_FALSE(condition) assert_false(condition, #condition)
#define ASSERT_EQ(expected, actual) assert_equals(expected, actual, #expected " == " #actual)
#define ASSERT_THROWS(func) assert_throws([&]() { func; }, #func " should throw")

#endif // YIRAGE_TESTING_FRAMEWORK_H
```

### 2. Specific Test Categories Implementation

#### Unit Tests for Core Components

```cpp
// tests/unit/test_compatibility_layer.cc
#include "yirage/testing/testing_framework.h"
#include "yirage/compat/cuda_runtime.h"
#include "yirage/compat/omp.h"
#include "yirage/compat/nlohmann/json.hpp"

YIRAGE_TEST(CudaCompatibilityTest, "CUDA Compatibility Layer") {
    // Test basic CUDA type definitions
    dim3 grid(8, 8, 1);
    ASSERT_EQ(8u, grid.x);
    ASSERT_EQ(8u, grid.y);
    ASSERT_EQ(1u, grid.z);
    
    // Test memory management
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 1024);
    ASSERT_EQ(cudaSuccess, err);
    ASSERT_TRUE(ptr != nullptr);
    
    err = cudaFree(ptr);
    ASSERT_EQ(cudaSuccess, err);
    
    // Test stream operations
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    ASSERT_EQ(cudaSuccess, err);
    
    err = cudaStreamDestroy(stream);
    ASSERT_EQ(cudaSuccess, err);
    
    log_info("CUDA compatibility layer basic functionality verified");
}

YIRAGE_TEST(OpenMPCompatibilityTest, "OpenMP Compatibility Layer") {
    // Test thread functions
    int num_threads = omp_get_num_threads();
    ASSERT_TRUE(num_threads >= 1);
    
    int thread_num = omp_get_thread_num();
    ASSERT_TRUE(thread_num >= 0);
    
    // Test timing functions
    double start_time = omp_get_wtime();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double end_time = omp_get_wtime();
    ASSERT_TRUE(end_time > start_time);
    
    // Test lock operations
    omp_lock_t lock;
    omp_init_lock(&lock);
    
    int test_result = omp_test_lock(&lock);
    ASSERT_EQ(1, test_result);
    
    omp_unset_lock(&lock);
    
    log_info("OpenMP compatibility layer basic functionality verified");
}

YIRAGE_TEST(JsonCompatibilityTest, "JSON Compatibility Layer") {
    // Test basic JSON operations
    nlohmann::json obj;
    obj["name"] = "test";
    obj["value"] = 42;
    obj["enabled"] = true;
    
    ASSERT_EQ("test", obj["name"].get<std::string>());
    ASSERT_EQ(42, obj["value"].get<int>());
    ASSERT_TRUE(obj["enabled"].get<bool>());
    
    // Test serialization
    std::string json_str = obj.dump();
    ASSERT_TRUE(json_str.find("\"name\":\"test\"") != std::string::npos);
    
    // Test array operations
    nlohmann::json arr = nlohmann::json::array();
    arr.push_back(1);
    arr.push_back(2);
    arr.push_back(3);
    
    ASSERT_EQ(3, arr.size());
    ASSERT_EQ(1, arr[0].get<int>());
    
    log_info("JSON compatibility layer basic functionality verified");
}
```

#### Integration Tests

```cpp
// tests/integration/test_build_system.cc
#include "yirage/testing/testing_framework.h"
#include "yirage/core/configuration.h"
#include "yirage/core/logging.h"

class BuildSystemIntegrationTest : public yirage::testing::IntegrationTest, 
                                 public yirage::testing::YirageTestFixture {
public:
    BuildSystemIntegrationTest() : IntegrationTest("Build System Integration") {}
    
    void setup() override {
        YirageTestFixture::setup();
        test_config_dir_ = create_temp_directory();
    }
    
    void teardown() override {
        cleanup_temp_directory(test_config_dir_);
        YirageTestFixture::teardown();
    }
    
    void run() override {
        test_configuration_loading();
        test_dependency_detection();
        test_compatibility_layer_integration();
        test_build_mode_selection();
    }
    
private:
    std::string test_config_dir_;
    
    void test_configuration_loading() {
        log_info("Testing configuration loading...");
        
        // Create test configuration
        std::string config_content = R"({
            "build": {
                "mode": "ENHANCED",
                "enable_cuda": false,
                "enable_openmp": true
            },
            "runtime": {
                "log_level": "DEBUG",
                "num_threads": 4
            }
        })";
        
        std::string config_file = test_config_dir_ + "/test_config.json";
        std::ofstream file(config_file);
        file << config_content;
        file.close();
        
        // Load configuration
        auto config_mgr = std::make_shared<yirage::core::ConfigurationManager>();
        config_mgr->add_source(std::make_shared<yirage::core::FileConfigSource>(
            config_file, "json", 50));
        
        bool loaded = config_mgr->load_configuration();
        ASSERT_TRUE(loaded);
        
        // Verify configuration values
        std::string build_mode = config_mgr->get<std::string>("build.mode");
        ASSERT_EQ("ENHANCED", build_mode);
        
        bool cuda_enabled = config_mgr->get<bool>("build.enable_cuda");
        ASSERT_FALSE(cuda_enabled);
        
        int num_threads = config_mgr->get<int>("runtime.num_threads");
        ASSERT_EQ(4, num_threads);
        
        log_info("Configuration loading test passed");
    }
    
    void test_dependency_detection() {
        log_info("Testing dependency detection...");
        
        // Test environment detection
        auto capabilities = yirage::core::EnvironmentDetector::detect_system_capabilities();
        
        ASSERT_TRUE(capabilities.cpu_cores > 0);
        ASSERT_TRUE(capabilities.total_memory_mb > 0);
        ASSERT_TRUE(!capabilities.os_type.empty());
        ASSERT_TRUE(!capabilities.arch.empty());
        
        log_info("System capabilities detected: " + capabilities.to_string());
        
        // Test dependency availability
        // These should work regardless of actual dependency availability
        // due to compatibility layers
        
        log_info("Dependency detection test passed");
    }
    
    void test_compatibility_layer_integration() {
        log_info("Testing compatibility layer integration...");
        
        // Test that all compatibility layers work together
        
        // CUDA + OpenMP compatibility
        void* cuda_ptr = nullptr;
        cudaError_t cuda_err = cudaMalloc(&cuda_ptr, 1024);
        ASSERT_EQ(cudaSuccess, cuda_err);
        
        omp_set_num_threads(2);
        int omp_threads = omp_get_num_threads();
        ASSERT_TRUE(omp_threads >= 1);
        
        // JSON + logging compatibility
        nlohmann::json config;
        config["test"] = "integration";
        
        auto logger = yirage::core::LoggerRegistry::get_logger("integration_test");
        logger->info("JSON config: " + config.dump());
        
        // Cleanup
        cudaFree(cuda_ptr);
        
        log_info("Compatibility layer integration test passed");
    }
    
    void test_build_mode_selection() {
        log_info("Testing build mode selection...");
        
        // Test different build modes
        std::vector<std::string> build_modes = {"CORE", "ENHANCED", "FULL"};
        
        for (const auto& mode : build_modes) {
            auto config_mgr = std::make_shared<yirage::core::ConfigurationManager>();
            
            // Set build mode
            config_mgr->set("build.mode", mode);
            
            // Verify mode is set correctly
            std::string actual_mode = config_mgr->get<std::string>("build.mode");
            ASSERT_EQ(mode, actual_mode);
            
            // Test that appropriate features are enabled/disabled
            // based on build mode
            // (Implementation would check feature availability)
            
            log_info("Build mode " + mode + " configuration verified");
        }
        
        log_info("Build mode selection test passed");
    }
};

// Register the integration test
static auto build_system_test_registered = []() {
    yirage::testing::TestRegistry::register_test<BuildSystemIntegrationTest>("BuildSystemIntegration");
    return true;
}();
```

#### Performance Tests

```cpp
// tests/performance/test_optimization_performance.cc
#include "yirage/testing/testing_framework.h"
#include "yirage/optimization/optimizer.h"

YIRAGE_PERFORMANCE_TEST(OptimizationPerformanceTest, "Optimization Performance") {
    // Set performance thresholds
    set_performance_threshold("optimization_time_ms", 1000.0); // 1 second max
    set_performance_threshold("memory_usage_mb", 100.0); // 100MB max
    set_performance_threshold("cpu_usage_percent", 80.0); // 80% CPU max
    
    // Prepare test data
    auto test_data = generate_large_test_input();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run optimization
    auto optimizer = create_optimizer();
    auto result = optimizer->optimize(test_data);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Record performance metrics
    record_metric("optimization_time_ms", duration.count());
    record_metric("output_size_kb", result.size() / 1024.0);
    record_metric("optimization_ratio", calculate_optimization_ratio(test_data, result));
    
    // Verify results
    ASSERT_TRUE(result.is_valid());
    ASSERT_TRUE(result.size() > 0);
    
    // Check performance thresholds
    assert_performance_threshold("optimization_time_ms", duration.count());
    
    log_info("Optimization completed in " + std::to_string(duration.count()) + "ms");
}

YIRAGE_PERFORMANCE_TEST(MemoryUsageTest, "Memory Usage Performance") {
    set_performance_threshold("peak_memory_mb", 500.0);
    set_performance_threshold("memory_leak_mb", 10.0);
    
    size_t initial_memory = get_current_memory_usage();
    
    // Perform memory-intensive operations
    std::vector<std::unique_ptr<LargeObject>> objects;
    for (int i = 0; i < 1000; ++i) {
        objects.push_back(std::make_unique<LargeObject>(1024 * 1024)); // 1MB each
    }
    
    size_t peak_memory = get_current_memory_usage();
    record_metric("peak_memory_mb", (peak_memory - initial_memory) / (1024.0 * 1024.0));
    
    // Cleanup
    objects.clear();
    force_garbage_collection(); // If applicable
    
    size_t final_memory = get_current_memory_usage();
    size_t memory_leak = final_memory - initial_memory;
    record_metric("memory_leak_mb", memory_leak / (1024.0 * 1024.0));
    
    // Verify memory usage
    assert_performance_threshold("peak_memory_mb", (peak_memory - initial_memory) / (1024.0 * 1024.0));
    assert_performance_threshold("memory_leak_mb", memory_leak / (1024.0 * 1024.0));
}
```

### 3. Automated Testing Infrastructure

```yaml
# .github/workflows/comprehensive_testing.yml
name: Comprehensive Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Nightly tests

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        build-mode: [CORE, ENHANCED, FULL]
        compiler: [gcc-11, clang-14]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Test Environment
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake ninja-build
        
    - name: Configure Build
      run: |
        cmake -S . -B build \
          -DYICA_BUILD_MODE=${{ matrix.build-mode }} \
          -DCMAKE_CXX_COMPILER=${{ matrix.compiler }} \
          -DYICA_ENABLE_TESTING=ON
          
    - name: Build
      run: cmake --build build --parallel
      
    - name: Run Unit Tests
      run: |
        cd build
        ctest -L unit --output-on-failure
        
    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: unit-test-results-${{ matrix.build-mode }}-${{ matrix.compiler }}
        path: build/test-results/

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Integration Environment
      run: |
        # Setup dependencies for integration testing
        sudo apt-get install -y z3 libz3-dev
        
    - name: Configure Build
      run: |
        cmake -S . -B build \
          -DYICA_BUILD_MODE=FULL \
          -DYICA_ENABLE_TESTING=ON \
          -DYICA_ENABLE_INTEGRATION_TESTS=ON
          
    - name: Build
      run: cmake --build build --parallel
      
    - name: Run Integration Tests
      run: |
        cd build
        ctest -L integration --output-on-failure --timeout 600
        
    - name: Generate Integration Report
      run: |
        cd build
        ./tests/generate_integration_report.sh

  performance-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Performance Environment
      run: |
        # Configure system for performance testing
        sudo sysctl -w kernel.perf_event_paranoid=1
        
    - name: Configure Build
      run: |
        cmake -S . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DYICA_BUILD_MODE=FULL \
          -DYICA_ENABLE_PERFORMANCE_TESTS=ON
          
    - name: Build
      run: cmake --build build --parallel
      
    - name: Run Performance Tests
      run: |
        cd build
        ctest -L performance --output-on-failure --timeout 1800
        
    - name: Generate Performance Report
      run: |
        cd build
        ./tests/generate_performance_report.sh
        
    - name: Upload Performance Results
      uses: actions/upload-artifact@v3
      with:
        name: performance-test-results
        path: build/performance-results/

  compatibility-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-20.04, ubuntu-22.04, macos-12, macos-13]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Compatibility Environment
      run: |
        if [[ "${{ matrix.os }}" == macos* ]]; then
          brew install cmake ninja z3
        else
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build libz3-dev
        fi
        
    - name: Test Compatibility Layers
      run: |
        cmake -S . -B build \
          -DYICA_BUILD_MODE=CORE \
          -DYICA_ENABLE_TESTING=ON \
          -DYICA_TEST_COMPATIBILITY_ONLY=ON
        cmake --build build --parallel
        cd build && ctest -L compatibility --output-on-failure

  stress-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[stress-test]')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Stress Test Environment
      run: |
        # Configure for heavy load testing
        ulimit -n 65536
        
    - name: Configure Build
      run: |
        cmake -S . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DYICA_BUILD_MODE=FULL \
          -DYICA_ENABLE_STRESS_TESTS=ON
          
    - name: Build
      run: cmake --build build --parallel
      
    - name: Run Stress Tests
      run: |
        cd build
        timeout 3600 ctest -L stress --output-on-failure
        
    - name: Collect System Metrics
      if: always()
      run: |
        dmesg | tail -100 > system_messages.log
        free -h > memory_status.log
        ps aux > process_status.log
        
    - name: Upload Stress Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: stress-test-results
        path: |
          build/stress-results/
          system_messages.log
          memory_status.log
          process_status.log
```

### 4. Test Execution and Reporting

```bash
#!/bin/bash
# scripts/run_comprehensive_tests.sh

set -e

echo "🧪 Starting Comprehensive YIRAGE Testing..."

# Parse command line arguments
BUILD_MODE=${1:-"ENHANCED"}
TEST_CATEGORIES=${2:-"unit,integration,performance"}
PARALLEL_JOBS=${3:-$(nproc)}

echo "Configuration:"
echo "  Build Mode: $BUILD_MODE"
echo "  Test Categories: $TEST_CATEGORIES"
echo "  Parallel Jobs: $PARALLEL_JOBS"

# Setup test environment
mkdir -p build/test-results
mkdir -p build/test-logs

# Configure build
echo "🔧 Configuring build..."
cmake -S . -B build \
  -DYICA_BUILD_MODE=$BUILD_MODE \
  -DYICA_ENABLE_TESTING=ON \
  -DYICA_ENABLE_ALL_TEST_TYPES=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build
echo "🔨 Building..."
cmake --build build --parallel $PARALLEL_JOBS

# Run tests by category
IFS=',' read -ra CATEGORIES <<< "$TEST_CATEGORIES"
for category in "${CATEGORIES[@]}"; do
    echo "🧪 Running $category tests..."
    
    case $category in
        "unit")
            cd build && ctest -L unit --output-on-failure \
              --output-junit test-results/unit-tests.xml
            ;;
        "integration")
            cd build && ctest -L integration --output-on-failure \
              --output-junit test-results/integration-tests.xml \
              --timeout 600
            ;;
        "performance")
            cd build && ctest -L performance --output-on-failure \
              --output-junit test-results/performance-tests.xml \
              --timeout 1800
            ;;
        "stress")
            cd build && timeout 3600 ctest -L stress --output-on-failure \
              --output-junit test-results/stress-tests.xml
            ;;
        "compatibility")
            cd build && ctest -L compatibility --output-on-failure \
              --output-junit test-results/compatibility-tests.xml
            ;;
    esac
    cd ..
done

# Generate comprehensive report
echo "📊 Generating test report..."
python3 scripts/generate_test_report.py build/test-results/

echo "✅ Comprehensive testing completed!"
echo "📋 Results available in build/test-results/"
```

This comprehensive testing framework provides:

1. **Multi-level testing** with unit, integration, performance, and stress tests
2. **Automated CI/CD pipeline** with matrix testing across environments
3. **Performance regression detection** with configurable thresholds
4. **Compatibility testing** across different operating systems and build configurations
5. **Comprehensive reporting** with detailed metrics and analysis
6. **Mock and fixture support** for reliable test isolation
7. **Test data management** for consistent testing datasets

The framework ensures that all aspects of the YICA/YiRage project are thoroughly tested, from individual components to full system integration, providing confidence in the stability and performance of the production-ready system.
