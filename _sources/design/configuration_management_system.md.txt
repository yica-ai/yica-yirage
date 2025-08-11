# Production-Grade Configuration Management System

## Current Configuration Issues

### 1. Scattered Configuration
- **Multiple config files**: CMakeLists.txt, setup.py, pyproject.toml, conda env files
- **Inconsistent settings**: Different default values across build systems  
- **No central management**: No single source of truth for project configuration
- **Environment-specific hardcoding**: Paths and settings hardcoded for specific environments

### 2. Runtime Configuration Problems
- **No dynamic reconfiguration**: Settings fixed at build time
- **Poor validation**: Invalid configurations cause runtime failures
- **Missing environment detection**: No automatic adaptation to available resources
- **No configuration versioning**: Configuration changes not tracked

## Comprehensive Configuration Design

### 1. Hierarchical Configuration System

```cpp
// yirage/include/yirage/core/configuration.h
#ifndef YIRAGE_CONFIGURATION_H
#define YIRAGE_CONFIGURATION_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <optional>
#include <variant>

namespace yirage {
namespace core {

    // Configuration value types
    using ConfigValue = std::variant<
        bool, int64_t, double, std::string, 
        std::vector<std::string>, std::map<std::string, std::string>
    >;
    
    // Configuration validation
    class ConfigValidator {
    public:
        virtual ~ConfigValidator() = default;
        virtual bool validate(const ConfigValue& value) = 0;
        virtual std::string get_error_message() const = 0;
    };
    
    // Range validator for numeric values
    template<typename T>
    class RangeValidator : public ConfigValidator {
    private:
        T min_value_, max_value_;
        std::string error_msg_;
        
    public:
        RangeValidator(T min_val, T max_val) : min_value_(min_val), max_value_(max_val) {
            error_msg_ = "Value must be between " + std::to_string(min_val) + 
                        " and " + std::to_string(max_val);
        }
        
        bool validate(const ConfigValue& value) override {
            if (std::holds_alternative<T>(value)) {
                T val = std::get<T>(value);
                return val >= min_value_ && val <= max_value_;
            }
            return false;
        }
        
        std::string get_error_message() const override { return error_msg_; }
    };
    
    // Choice validator for string values
    class ChoiceValidator : public ConfigValidator {
    private:
        std::vector<std::string> allowed_values_;
        std::string error_msg_;
        
    public:
        ChoiceValidator(const std::vector<std::string>& choices) : allowed_values_(choices) {
            error_msg_ = "Value must be one of: ";
            for (size_t i = 0; i < choices.size(); ++i) {
                if (i > 0) error_msg_ += ", ";
                error_msg_ += choices[i];
            }
        }
        
        bool validate(const ConfigValue& value) override;
        std::string get_error_message() const override { return error_msg_; }
    };
    
    // Configuration schema definition
    struct ConfigSchema {
        std::string key;
        ConfigValue default_value;
        std::string description;
        bool required = false;
        std::shared_ptr<ConfigValidator> validator;
        std::function<void(const ConfigValue&)> change_callback;
    };
    
    // Configuration source interface
    class ConfigSource {
    public:
        virtual ~ConfigSource() = default;
        virtual std::map<std::string, ConfigValue> load() = 0;
        virtual bool save(const std::map<std::string, ConfigValue>& config) = 0;
        virtual std::string get_source_name() const = 0;
        virtual int get_priority() const = 0; // Higher priority overrides lower
    };
    
    // File-based configuration source
    class FileConfigSource : public ConfigSource {
    private:
        std::string file_path_;
        std::string format_; // "json", "yaml", "toml", "ini"
        int priority_;
        
    public:
        FileConfigSource(const std::string& path, const std::string& format, int priority = 50);
        
        std::map<std::string, ConfigValue> load() override;
        bool save(const std::map<std::string, ConfigValue>& config) override;
        std::string get_source_name() const override { return file_path_; }
        int get_priority() const override { return priority_; }
        
    private:
        std::map<std::string, ConfigValue> load_json();
        std::map<std::string, ConfigValue> load_yaml();
        std::map<std::string, ConfigValue> load_toml();
        std::map<std::string, ConfigValue> load_ini();
        
        bool save_json(const std::map<std::string, ConfigValue>& config);
        bool save_yaml(const std::map<std::string, ConfigValue>& config);
        bool save_toml(const std::map<std::string, ConfigValue>& config);
        bool save_ini(const std::map<std::string, ConfigValue>& config);
    };
    
    // Environment variable configuration source
    class EnvironmentConfigSource : public ConfigSource {
    private:
        std::string prefix_; // e.g., "YIRAGE_"
        int priority_;
        
    public:
        EnvironmentConfigSource(const std::string& prefix = "YIRAGE_", int priority = 70);
        
        std::map<std::string, ConfigValue> load() override;
        bool save(const std::map<std::string, ConfigValue>& config) override;
        std::string get_source_name() const override { return "Environment Variables"; }
        int get_priority() const override { return priority_; }
        
    private:
        ConfigValue parse_env_value(const std::string& value);
        std::string to_env_key(const std::string& config_key);
    };
    
    // Command line argument configuration source
    class CommandLineConfigSource : public ConfigSource {
    private:
        int argc_;
        char** argv_;
        int priority_;
        
    public:
        CommandLineConfigSource(int argc, char** argv, int priority = 90);
        
        std::map<std::string, ConfigValue> load() override;
        bool save(const std::map<std::string, ConfigValue>& config) override { return false; } // Read-only
        std::string get_source_name() const override { return "Command Line"; }
        int get_priority() const override { return priority_; }
        
    private:
        ConfigValue parse_arg_value(const std::string& value);
    };
    
    // Main configuration manager
    class ConfigurationManager {
    private:
        std::map<std::string, ConfigSchema> schema_;
        std::map<std::string, ConfigValue> current_config_;
        std::vector<std::shared_ptr<ConfigSource>> sources_;
        std::mutex config_mutex_;
        std::shared_ptr<Logger> logger_;
        
        // Change tracking
        std::map<std::string, std::vector<std::function<void(const ConfigValue&)>>> change_listeners_;
        
    public:
        ConfigurationManager();
        
        // Schema management
        void register_schema(const ConfigSchema& schema);
        void register_schemas(const std::vector<ConfigSchema>& schemas);
        
        // Source management
        void add_source(std::shared_ptr<ConfigSource> source);
        void remove_source(const std::string& source_name);
        void clear_sources();
        
        // Configuration loading and saving
        bool load_configuration();
        bool save_configuration(const std::string& source_name = "");
        bool reload_configuration();
        
        // Value access
        template<typename T>
        T get(const std::string& key) const;
        
        template<typename T>
        T get(const std::string& key, const T& default_value) const;
        
        bool has(const std::string& key) const;
        
        // Value modification
        template<typename T>
        bool set(const std::string& key, const T& value);
        
        bool unset(const std::string& key);
        
        // Validation
        bool validate_all();
        std::vector<std::string> get_validation_errors();
        
        // Change notification
        void add_change_listener(const std::string& key, 
                               std::function<void(const ConfigValue&)> callback);
        void remove_change_listeners(const std::string& key);
        
        // Introspection
        std::vector<std::string> get_all_keys() const;
        std::map<std::string, ConfigValue> get_all_values() const;
        std::string get_config_summary() const;
        
        // Environment detection and auto-configuration
        void auto_detect_environment();
        void apply_environment_optimizations();
        
    private:
        void merge_configurations();
        void notify_change(const std::string& key, const ConfigValue& value);
        bool validate_value(const std::string& key, const ConfigValue& value);
    };
    
    // Specialized configuration managers
    class BuildConfiguration {
    private:
        std::shared_ptr<ConfigurationManager> config_manager_;
        
    public:
        BuildConfiguration();
        
        // Build-time configuration
        bool enable_cuda() const;
        bool enable_openmp() const;
        bool enable_z3() const;
        bool enable_triton() const;
        std::string get_build_mode() const; // "CORE", "ENHANCED", "FULL"
        std::string get_optimization_level() const; // "Debug", "Release", "RelWithDebInfo"
        
        // Paths and directories
        std::string get_install_prefix() const;
        std::string get_lib_dir() const;
        std::string get_include_dir() const;
        std::string get_bin_dir() const;
        
        // Compiler settings
        std::vector<std::string> get_cxx_flags() const;
        std::vector<std::string> get_cuda_flags() const;
        std::string get_cxx_standard() const;
        
        // Dependencies
        std::string get_z3_path() const;
        std::string get_cuda_path() const;
        std::string get_openmp_path() const;
        
        void auto_configure_build_environment();
    };
    
    class RuntimeConfiguration {
    private:
        std::shared_ptr<ConfigurationManager> config_manager_;
        
    public:
        RuntimeConfiguration();
        
        // Performance settings
        int get_num_threads() const;
        size_t get_memory_limit_mb() const;
        bool enable_performance_monitoring() const;
        double get_optimization_timeout_seconds() const;
        
        // Logging configuration
        std::string get_log_level() const;
        std::string get_log_file() const;
        bool enable_structured_logging() const;
        
        // YICA-specific settings
        int get_cim_array_count() const;
        size_t get_spm_size_mb() const;
        std::string get_optimization_strategy() const;
        bool enable_fusion_optimization() const;
        
        // Feature toggles
        bool enable_experimental_features() const;
        bool enable_compatibility_mode() const;
        bool enable_debug_output() const;
        
        void auto_configure_runtime_environment();
        void apply_performance_profile(const std::string& profile); // "development", "production", "benchmark"
    };

}} // namespace yirage::core

#endif // YIRAGE_CONFIGURATION_H
```

### 2. Configuration File Formats and Examples

#### Master Configuration Schema

```yaml
# yirage/config/yirage_schema.yaml
# Master configuration schema for YIRAGE
schema_version: "1.0"

sections:
  build:
    description: "Build-time configuration"
    properties:
      mode:
        type: string
        default: "ENHANCED"
        choices: ["CORE", "ENHANCED", "FULL"]
        description: "Build mode determining feature set"
        
      optimization_level:
        type: string
        default: "Release"
        choices: ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]
        description: "Compiler optimization level"
        
      enable_cuda:
        type: boolean
        default: false
        description: "Enable CUDA GPU acceleration"
        
      enable_openmp:
        type: boolean
        default: true
        description: "Enable OpenMP parallel processing"
        
      enable_z3:
        type: boolean
        default: true
        description: "Enable Z3 SMT solver"
        
      enable_triton:
        type: boolean
        default: false
        description: "Enable Triton kernel compilation"
        
      cxx_standard:
        type: string
        default: "17"
        choices: ["17", "20", "23"]
        description: "C++ standard version"

  runtime:
    description: "Runtime configuration"
    properties:
      num_threads:
        type: integer
        default: 0  # 0 means auto-detect
        min: 1
        max: 128
        description: "Number of worker threads (0 for auto-detect)"
        
      memory_limit_mb:
        type: integer
        default: 0  # 0 means no limit
        min: 0
        max: 1048576  # 1TB
        description: "Memory limit in MB (0 for no limit)"
        
      optimization_timeout_seconds:
        type: number
        default: 30.0
        min: 1.0
        max: 3600.0
        description: "Timeout for optimization operations"
        
      log_level:
        type: string
        default: "INFO"
        choices: ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        description: "Logging level"
        
      log_file:
        type: string
        default: ""  # Empty means console only
        description: "Log file path (empty for console only)"

  yica:
    description: "YICA-specific configuration"
    properties:
      cim_array_count:
        type: integer
        default: 512
        min: 1
        max: 2048
        description: "Number of CIM arrays to simulate"
        
      spm_size_mb:
        type: integer
        default: 128
        min: 1
        max: 1024
        description: "SPM size per CIM die in MB"
        
      optimization_strategy:
        type: string
        default: "balanced"
        choices: ["performance", "balanced", "memory", "power"]
        description: "Optimization strategy profile"
        
      enable_fusion_optimization:
        type: boolean
        default: true
        description: "Enable kernel fusion optimization"

  features:
    description: "Feature toggles"
    properties:
      enable_experimental:
        type: boolean
        default: false
        description: "Enable experimental features"
        
      enable_compatibility_mode:
        type: boolean
        default: false
        description: "Enable compatibility mode for older systems"
        
      enable_performance_monitoring:
        type: boolean
        default: true
        description: "Enable performance monitoring and profiling"
        
      enable_debug_output:
        type: boolean
        default: false
        description: "Enable verbose debug output"
```

#### Environment-Specific Configurations

```json
// yirage/config/environments/development.json
{
  "environment": "development",
  "runtime": {
    "log_level": "DEBUG",
    "enable_debug_output": true,
    "optimization_timeout_seconds": 5.0
  },
  "features": {
    "enable_experimental": true,
    "enable_performance_monitoring": true
  },
  "yica": {
    "cim_array_count": 64,
    "spm_size_mb": 32
  }
}
```

```json
// yirage/config/environments/production.json
{
  "environment": "production",
  "runtime": {
    "log_level": "WARNING",
    "enable_debug_output": false,
    "optimization_timeout_seconds": 60.0,
    "log_file": "/var/log/yirage/yirage.log"
  },
  "features": {
    "enable_experimental": false,
    "enable_performance_monitoring": true
  },
  "yica": {
    "cim_array_count": 512,
    "spm_size_mb": 128,
    "optimization_strategy": "performance"
  }
}
```

```toml
# yirage/config/environments/benchmark.toml
environment = "benchmark"

[runtime]
log_level = "ERROR"
num_threads = 0  # Use all available cores
memory_limit_mb = 0  # No memory limit
optimization_timeout_seconds = 300.0

[features]
enable_experimental = false
enable_compatibility_mode = false
enable_performance_monitoring = true
enable_debug_output = false

[yica]
cim_array_count = 1024
spm_size_mb = 256
optimization_strategy = "performance"
enable_fusion_optimization = true
```

### 3. Dynamic Configuration Management

```cpp
// yirage/include/yirage/core/dynamic_config.h
#ifndef YIRAGE_DYNAMIC_CONFIG_H
#define YIRAGE_DYNAMIC_CONFIG_H

#include "configuration.h"
#include <thread>
#include <atomic>
#include <condition_variable>

namespace yirage {
namespace core {

    // Hot-reloadable configuration
    class DynamicConfigurationManager {
    private:
        std::shared_ptr<ConfigurationManager> config_manager_;
        std::vector<std::string> watched_files_;
        std::thread watcher_thread_;
        std::atomic<bool> stop_watching_;
        std::condition_variable config_changed_cv_;
        std::mutex config_changed_mutex_;
        
        // File modification tracking
        std::map<std::string, std::time_t> file_timestamps_;
        
    public:
        DynamicConfigurationManager(std::shared_ptr<ConfigurationManager> config_mgr);
        ~DynamicConfigurationManager();
        
        // File watching
        void start_watching();
        void stop_watching();
        void add_watched_file(const std::string& file_path);
        void remove_watched_file(const std::string& file_path);
        
        // Manual reload
        bool reload_configuration();
        
        // Change notification
        void wait_for_config_change();
        
        // Configuration validation
        bool validate_hot_reload_safety(const std::string& key) const;
        std::vector<std::string> get_hot_reloadable_keys() const;
        
    private:
        void file_watcher_loop();
        bool check_file_changes();
        std::time_t get_file_timestamp(const std::string& file_path);
    };
    
    // Configuration profile manager
    class ConfigurationProfileManager {
    private:
        std::shared_ptr<ConfigurationManager> config_manager_;
        std::map<std::string, std::map<std::string, ConfigValue>> profiles_;
        std::string current_profile_;
        
    public:
        ConfigurationProfileManager(std::shared_ptr<ConfigurationManager> config_mgr);
        
        // Profile management
        bool create_profile(const std::string& name, const std::string& base_profile = "");
        bool delete_profile(const std::string& name);
        bool switch_to_profile(const std::string& name);
        
        // Profile configuration
        bool set_profile_value(const std::string& profile, const std::string& key, const ConfigValue& value);
        bool import_profile_from_file(const std::string& profile, const std::string& file_path);
        bool export_profile_to_file(const std::string& profile, const std::string& file_path);
        
        // Profile introspection
        std::vector<std::string> get_available_profiles() const;
        std::string get_current_profile() const { return current_profile_; }
        std::map<std::string, ConfigValue> get_profile_config(const std::string& profile) const;
        
        // Predefined profiles
        void create_development_profile();
        void create_production_profile();
        void create_benchmark_profile();
        void create_debugging_profile();
    };
    
    // Environment auto-detection
    class EnvironmentDetector {
    public:
        struct SystemCapabilities {
            int cpu_cores = 0;
            size_t total_memory_mb = 0;
            bool has_cuda = false;
            std::string cuda_version;
            bool has_openmp = false;
            bool has_z3 = false;
            bool has_triton = false;
            std::string os_type; // "linux", "macos", "windows"
            std::string arch; // "x86_64", "arm64", etc.
            
            std::string to_string() const;
        };
        
        static SystemCapabilities detect_system_capabilities();
        static std::map<std::string, ConfigValue> recommend_configuration(const SystemCapabilities& caps);
        static void apply_auto_configuration(ConfigurationManager& config_mgr, const SystemCapabilities& caps);
        
    private:
        static int detect_cpu_cores();
        static size_t detect_total_memory();
        static bool detect_cuda_availability();
        static std::string detect_cuda_version();
        static bool detect_openmp_availability();
        static bool detect_z3_availability();
        static bool detect_triton_availability();
        static std::string detect_os_type();
        static std::string detect_architecture();
    };

}} // namespace yirage::core

#endif // YIRAGE_DYNAMIC_CONFIG_H
```

### 4. Configuration Implementation Examples

```cpp
// Usage example: Setting up configuration in main()
int main(int argc, char** argv) {
    // Initialize logging first
    yirage::core::LoggerRegistry::setup_default_console_logging();
    auto logger = YIRAGE_GET_LOGGER("main");
    
    try {
        // Create configuration manager
        auto config_mgr = std::make_shared<yirage::core::ConfigurationManager>();
        
        // Register schemas
        auto build_config = std::make_unique<yirage::core::BuildConfiguration>();
        auto runtime_config = std::make_unique<yirage::core::RuntimeConfiguration>();
        
        // Add configuration sources (in priority order)
        config_mgr->add_source(std::make_shared<yirage::core::FileConfigSource>(
            "yirage/config/default.yaml", "yaml", 10));
        config_mgr->add_source(std::make_shared<yirage::core::FileConfigSource>(
            "yirage/config/user.json", "json", 30));
        config_mgr->add_source(std::make_shared<yirage::core::EnvironmentConfigSource>(
            "YIRAGE_", 70));
        config_mgr->add_source(std::make_shared<yirage::core::CommandLineConfigSource>(
            argc, argv, 90));
        
        // Load configuration
        if (!config_mgr->load_configuration()) {
            YIRAGE_ERROR(logger, "Failed to load configuration");
            return 1;
        }
        
        // Auto-detect environment and apply optimizations
        config_mgr->auto_detect_environment();
        
        // Validate configuration
        if (!config_mgr->validate_all()) {
            auto errors = config_mgr->get_validation_errors();
            for (const auto& error : errors) {
                YIRAGE_ERROR(logger, "Configuration error: " + error);
            }
            return 1;
        }
        
        // Set up dynamic configuration
        auto dynamic_config = std::make_unique<yirage::core::DynamicConfigurationManager>(config_mgr);
        dynamic_config->start_watching();
        
        // Configure logging based on configuration
        std::string log_level = config_mgr->get<std::string>("runtime.log_level");
        std::string log_file = config_mgr->get<std::string>("runtime.log_file");
        
        if (!log_file.empty()) {
            yirage::core::LoggerRegistry::setup_default_file_logging("./logs");
        }
        
        YIRAGE_INFO(logger, "YIRAGE started with configuration: " + config_mgr->get_config_summary());
        
        // Run main application logic
        run_yirage_application(config_mgr);
        
    } catch (const yirage::core::YirageException& e) {
        YIRAGE_CRITICAL(logger, e.to_detailed_string());
        return 1;
    }
    
    return 0;
}

// Usage example: Configuration-driven optimization
void optimize_with_config(const InputCode& input, std::shared_ptr<yirage::core::ConfigurationManager> config) {
    auto logger = YIRAGE_GET_LOGGER("optimizer");
    
    // Get configuration values
    double timeout = config->get<double>("runtime.optimization_timeout_seconds");
    std::string strategy = config->get<std::string>("yica.optimization_strategy");
    bool enable_fusion = config->get<bool>("yica.enable_fusion_optimization");
    int cim_arrays = config->get<int>("yica.cim_array_count");
    
    YIRAGE_INFO(logger, "Starting optimization with strategy: " + strategy);
    
    // Configure optimizer based on settings
    OptimizerConfig opt_config;
    opt_config.timeout_seconds = timeout;
    opt_config.strategy = strategy;
    opt_config.enable_fusion = enable_fusion;
    opt_config.cim_array_count = cim_arrays;
    
    // Add change listener for dynamic reconfiguration
    config->add_change_listener("yica.optimization_strategy", 
        [&](const yirage::core::ConfigValue& value) {
            std::string new_strategy = std::get<std::string>(value);
            YIRAGE_INFO(logger, "Optimization strategy changed to: " + new_strategy);
            opt_config.strategy = new_strategy;
        });
    
    // Run optimization
    auto result = optimize_code(input, opt_config);
    
    YIRAGE_INFO(logger, "Optimization completed successfully");
}
```

This production-grade configuration management system provides:

1. **Hierarchical configuration** with multiple sources and priorities
2. **Schema validation** with type checking and custom validators  
3. **Hot-reloading** for dynamic configuration changes
4. **Environment auto-detection** for optimal default settings
5. **Profile management** for different deployment scenarios
6. **Comprehensive validation** with detailed error reporting
7. **Change notification** for reactive configuration updates

The system ensures that YICA/YiRage can adapt to different environments, deployment scenarios, and user preferences while maintaining strict validation and error handling standards.
