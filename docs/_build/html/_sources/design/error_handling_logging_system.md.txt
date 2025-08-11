# Production-Grade Error Handling and Logging System

## Current Issues Analysis

### 1. Error Handling Problems
- **Inconsistent error reporting**: Different components use different error mechanisms
- **Poor error propagation**: Errors lost in compatibility layers
- **Insufficient context**: Error messages lack actionable information
- **No recovery mechanisms**: Failures cascade without graceful degradation

### 2. Logging Deficiencies  
- **Ad-hoc logging**: printf/cout scattered throughout codebase
- **No structured logging**: Difficult to parse and analyze
- **Missing performance logging**: No timing or resource usage tracking
- **Debug information overload**: No proper log level management

## Comprehensive Error Handling Design

### 1. Structured Exception Hierarchy

```cpp
// yirage/include/yirage/core/exceptions.h
#ifndef YIRAGE_EXCEPTIONS_H
#define YIRAGE_EXCEPTIONS_H

#include <exception>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <chrono>

namespace yirage {
namespace core {

    // Base exception with rich context
    class YirageException : public std::exception {
    private:
        std::string message_;
        std::string component_;
        std::string function_;
        std::string file_;
        int line_;
        std::chrono::system_clock::time_point timestamp_;
        std::map<std::string, std::string> context_;
        std::vector<std::string> stack_trace_;
        
    public:
        YirageException(const std::string& message,
                       const std::string& component = "",
                       const std::string& function = "",
                       const std::string& file = "",
                       int line = 0);
        
        // Context management
        YirageException& add_context(const std::string& key, const std::string& value);
        YirageException& add_context(const std::string& key, int value);
        YirageException& add_context(const std::string& key, double value);
        
        // Stack trace
        void capture_stack_trace();
        
        // Accessors
        const char* what() const noexcept override;
        const std::string& get_component() const { return component_; }
        const std::string& get_function() const { return function_; }
        const std::string& get_file() const { return file_; }
        int get_line() const { return line_; }
        const auto& get_timestamp() const { return timestamp_; }
        const auto& get_context() const { return context_; }
        const auto& get_stack_trace() const { return stack_trace_; }
        
        // Serialization for logging
        std::string to_json() const;
        std::string to_detailed_string() const;
    };
    
    // Specific exception types
    class BuildConfigurationError : public YirageException {
    public:
        BuildConfigurationError(const std::string& message,
                              const std::string& missing_component = "",
                              const std::string& suggestion = "");
    };
    
    class DependencyError : public YirageException {
    public:
        enum class Type { MISSING, VERSION_MISMATCH, INITIALIZATION_FAILED };
        
        DependencyError(const std::string& dependency_name,
                       Type error_type,
                       const std::string& expected_version = "",
                       const std::string& actual_version = "");
                       
        Type get_error_type() const { return error_type_; }
        
    private:
        Type error_type_;
    };
    
    class OptimizationError : public YirageException {
    public:
        enum class Phase { ANALYSIS, TRANSFORMATION, CODEGEN, VERIFICATION };
        
        OptimizationError(Phase phase,
                         const std::string& message,
                         const std::string& input_description = "");
                         
        Phase get_phase() const { return phase_; }
        
    private:
        Phase phase_;
    };
    
    class RuntimeError : public YirageException {
    public:
        enum class Severity { WARNING, ERROR, CRITICAL, FATAL };
        
        RuntimeError(Severity severity,
                    const std::string& message,
                    bool recoverable = true);
                    
        Severity get_severity() const { return severity_; }
        bool is_recoverable() const { return recoverable_; }
        
    private:
        Severity severity_;
        bool recoverable_;
    };
    
    class CompatibilityError : public YirageException {
    public:
        CompatibilityError(const std::string& feature,
                          const std::string& fallback_description = "");
    };
    
    class PerformanceError : public YirageException {
    public:
        PerformanceError(const std::string& operation,
                        double expected_time_ms,
                        double actual_time_ms,
                        const std::string& suggestion = "");
    };

}} // namespace yirage::core

// Convenience macros for exception throwing
#define YIRAGE_THROW(ExceptionType, message) \
    throw ExceptionType(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_THROW_WITH_CONTEXT(ExceptionType, message, ...) \
    do { \
        auto ex = ExceptionType(message, __FUNCTION__, __FILE__, __LINE__); \
        __VA_ARGS__; \
        throw ex; \
    } while(0)

#endif // YIRAGE_EXCEPTIONS_H
```

### 2. Advanced Logging Framework

```cpp
// yirage/include/yirage/core/logging.h
#ifndef YIRAGE_LOGGING_H
#define YIRAGE_LOGGING_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>

namespace yirage {
namespace core {

    enum class LogLevel {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARNING = 3,
        ERROR = 4,
        CRITICAL = 5,
        OFF = 6
    };
    
    struct LogEntry {
        LogLevel level;
        std::string message;
        std::string component;
        std::string function;
        std::string file;
        int line;
        std::thread::id thread_id;
        std::chrono::system_clock::time_point timestamp;
        std::map<std::string, std::string> metadata;
        
        std::string to_json() const;
        std::string to_formatted_string() const;
    };
    
    class LogSink {
    public:
        virtual ~LogSink() = default;
        virtual void write(const LogEntry& entry) = 0;
        virtual void flush() = 0;
    };
    
    class ConsoleSink : public LogSink {
    private:
        bool colorized_;
        std::mutex mutex_;
        
    public:
        ConsoleSink(bool colorized = true) : colorized_(colorized) {}
        void write(const LogEntry& entry) override;
        void flush() override;
        
    private:
        std::string get_color_code(LogLevel level) const;
        std::string get_level_string(LogLevel level) const;
    };
    
    class FileSink : public LogSink {
    private:
        std::string file_path_;
        std::ofstream file_stream_;
        std::mutex mutex_;
        size_t max_file_size_;
        int max_files_;
        bool json_format_;
        
    public:
        FileSink(const std::string& file_path,
                size_t max_file_size = 100 * 1024 * 1024, // 100MB
                int max_files = 5,
                bool json_format = false);
        
        ~FileSink();
        
        void write(const LogEntry& entry) override;
        void flush() override;
        
    private:
        void rotate_if_needed();
        void rotate_files();
    };
    
    class AsyncLogSink : public LogSink {
    private:
        std::unique_ptr<LogSink> underlying_sink_;
        std::vector<LogEntry> buffer_;
        std::mutex buffer_mutex_;
        std::condition_variable buffer_cv_;
        std::thread worker_thread_;
        std::atomic<bool> stop_requested_;
        size_t buffer_size_;
        
    public:
        AsyncLogSink(std::unique_ptr<LogSink> sink, size_t buffer_size = 1000);
        ~AsyncLogSink();
        
        void write(const LogEntry& entry) override;
        void flush() override;
        
    private:
        void worker_loop();
    };
    
    class Logger {
    private:
        std::string name_;
        LogLevel level_;
        std::vector<std::shared_ptr<LogSink>> sinks_;
        std::mutex sinks_mutex_;
        
    public:
        Logger(const std::string& name, LogLevel level = LogLevel::INFO);
        
        // Sink management
        void add_sink(std::shared_ptr<LogSink> sink);
        void remove_all_sinks();
        void set_level(LogLevel level) { level_ = level; }
        LogLevel get_level() const { return level_; }
        
        // Logging methods
        void log(LogLevel level, const std::string& message,
                const std::string& function = "",
                const std::string& file = "",
                int line = 0);
                
        template<typename... Args>
        void log_formatted(LogLevel level, const std::string& format,
                          const std::string& function,
                          const std::string& file,
                          int line,
                          Args&&... args);
        
        // Convenience methods
        void trace(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        void debug(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        void info(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        void warning(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        void error(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        void critical(const std::string& message, const std::string& function = "", const std::string& file = "", int line = 0);
        
        // Structured logging
        LogEntry& begin_entry(LogLevel level, const std::string& function = "", const std::string& file = "", int line = 0);
        void end_entry(LogEntry& entry);
        
        // Performance logging
        class PerformanceTimer {
        private:
            Logger& logger_;
            std::string operation_;
            std::chrono::high_resolution_clock::time_point start_time_;
            LogLevel level_;
            
        public:
            PerformanceTimer(Logger& logger, const std::string& operation, LogLevel level = LogLevel::DEBUG);
            ~PerformanceTimer();
            
            void add_metadata(const std::string& key, const std::string& value);
            double elapsed_ms() const;
        };
        
        PerformanceTimer time_operation(const std::string& operation, LogLevel level = LogLevel::DEBUG);
        
    private:
        void write_to_sinks(const LogEntry& entry);
    };
    
    class LoggerRegistry {
    private:
        static std::map<std::string, std::shared_ptr<Logger>> loggers_;
        static std::mutex registry_mutex_;
        static LogLevel global_level_;
        
    public:
        static std::shared_ptr<Logger> get_logger(const std::string& name);
        static void set_global_level(LogLevel level);
        static void shutdown_all_loggers();
        static void configure_from_file(const std::string& config_file);
        
        // Default logger setup
        static void setup_default_console_logging(LogLevel level = LogLevel::INFO);
        static void setup_default_file_logging(const std::string& log_dir, LogLevel level = LogLevel::DEBUG);
        static void setup_production_logging(const std::string& log_dir);
    };

}} // namespace yirage::core

// Convenience macros
#define YIRAGE_GET_LOGGER(name) yirage::core::LoggerRegistry::get_logger(name)

#define YIRAGE_LOG(logger, level, message) \
    logger->log(level, message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_TRACE(logger, message) \
    logger->trace(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_DEBUG(logger, message) \
    logger->debug(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_INFO(logger, message) \
    logger->info(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_WARNING(logger, message) \
    logger->warning(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_ERROR(logger, message) \
    logger->error(message, __FUNCTION__, __FILE__, __LINE__)

#define YIRAGE_CRITICAL(logger, message) \
    logger->critical(message, __FUNCTION__, __FILE__, __LINE__)

// Performance timing macro
#define YIRAGE_TIME_OPERATION(logger, operation) \
    auto timer = logger->time_operation(operation)

// Formatted logging macros
#define YIRAGE_LOG_FMT(logger, level, format, ...) \
    logger->log_formatted(level, format, __FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

#endif // YIRAGE_LOGGING_H
```

### 3. Error Recovery and Resilience Framework

```cpp
// yirage/include/yirage/core/error_recovery.h
#ifndef YIRAGE_ERROR_RECOVERY_H
#define YIRAGE_ERROR_RECOVERY_H

#include <functional>
#include <vector>
#include <memory>
#include <chrono>
#include <map>

namespace yirage {
namespace core {

    enum class RecoveryStrategy {
        IGNORE,           // Log and continue
        RETRY,            // Retry operation with backoff
        FALLBACK,         // Use alternative implementation
        GRACEFUL_DEGRADE, // Reduce functionality but continue
        FAIL_FAST        // Terminate immediately
    };
    
    struct RecoveryConfig {
        RecoveryStrategy strategy = RecoveryStrategy::FAIL_FAST;
        int max_retries = 3;
        std::chrono::milliseconds initial_delay{100};
        double backoff_multiplier = 2.0;
        std::chrono::milliseconds max_delay{5000};
        std::function<bool()> fallback_available;
        std::function<void()> fallback_action;
        std::function<void()> cleanup_action;
    };
    
    class ErrorRecoveryManager {
    private:
        std::map<std::string, RecoveryConfig> recovery_configs_;
        std::shared_ptr<Logger> logger_;
        
    public:
        ErrorRecoveryManager();
        
        // Configuration
        void set_recovery_config(const std::string& operation, const RecoveryConfig& config);
        void set_default_recovery_strategy(RecoveryStrategy strategy);
        
        // Recovery execution
        template<typename Func, typename... Args>
        auto execute_with_recovery(const std::string& operation, Func&& func, Args&&... args) 
            -> decltype(func(args...));
            
        // Monitoring
        struct RecoveryStats {
            int total_operations = 0;
            int successful_operations = 0;
            int failed_operations = 0;
            int retries_attempted = 0;
            int fallbacks_used = 0;
            std::chrono::milliseconds total_retry_time{0};
        };
        
        RecoveryStats get_stats(const std::string& operation) const;
        void reset_stats(const std::string& operation = "");
        
    private:
        std::map<std::string, RecoveryStats> stats_;
        std::mutex stats_mutex_;
        
        template<typename Func, typename... Args>
        auto retry_with_backoff(const std::string& operation, 
                              const RecoveryConfig& config,
                              Func&& func, Args&&... args) 
            -> decltype(func(args...));
    };
    
    // Circuit breaker for preventing cascading failures
    class CircuitBreaker {
    public:
        enum class State { CLOSED, OPEN, HALF_OPEN };
        
    private:
        State state_ = State::CLOSED;
        int failure_count_ = 0;
        int failure_threshold_;
        std::chrono::milliseconds timeout_;
        std::chrono::system_clock::time_point last_failure_time_;
        std::mutex mutex_;
        std::shared_ptr<Logger> logger_;
        
    public:
        CircuitBreaker(int failure_threshold = 5, 
                      std::chrono::milliseconds timeout = std::chrono::seconds(60));
        
        template<typename Func, typename... Args>
        auto call(Func&& func, Args&&... args) -> decltype(func(args...));
        
        State get_state() const;
        void reset();
        
    private:
        void record_success();
        void record_failure();
        bool should_attempt_reset() const;
    };
    
    // Global error handling utilities
    class GlobalErrorHandler {
    private:
        static std::function<void(const YirageException&)> exception_handler_;
        static std::function<void(const std::string&)> fatal_error_handler_;
        static std::shared_ptr<Logger> logger_;
        static bool initialized_;
        
    public:
        static void initialize();
        static void set_exception_handler(std::function<void(const YirageException&)> handler);
        static void set_fatal_error_handler(std::function<void(const std::string&)> handler);
        
        static void handle_exception(const YirageException& ex);
        static void handle_fatal_error(const std::string& message);
        
        // Standard signal handlers
        static void setup_signal_handlers();
        
    private:
        static void signal_handler(int signal);
        static void terminate_handler();
        static void unexpected_handler();
    };

}} // namespace yirage::core

#endif // YIRAGE_ERROR_RECOVERY_H
```

## Implementation Strategy

### 1. Exception Handling Implementation

```cpp
// yirage/src/core/exceptions.cc
#include "yirage/core/exceptions.h"
#include <sstream>
#include <iomanip>

namespace yirage {
namespace core {

YirageException::YirageException(const std::string& message,
                               const std::string& component,
                               const std::string& function,
                               const std::string& file,
                               int line)
    : message_(message), component_(component), function_(function), 
      file_(file), line_(line), timestamp_(std::chrono::system_clock::now()) {
    capture_stack_trace();
}

YirageException& YirageException::add_context(const std::string& key, const std::string& value) {
    context_[key] = value;
    return *this;
}

YirageException& YirageException::add_context(const std::string& key, int value) {
    context_[key] = std::to_string(value);
    return *this;
}

YirageException& YirageException::add_context(const std::string& key, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    context_[key] = oss.str();
    return *this;
}

const char* YirageException::what() const noexcept {
    return message_.c_str();
}

std::string YirageException::to_json() const {
    std::ostringstream json;
    json << "{\n";
    json << "  \"type\": \"YirageException\",\n";
    json << "  \"message\": \"" << message_ << "\",\n";
    json << "  \"component\": \"" << component_ << "\",\n";
    json << "  \"function\": \"" << function_ << "\",\n";
    json << "  \"file\": \"" << file_ << "\",\n";
    json << "  \"line\": " << line_ << ",\n";
    
    // Add timestamp
    auto time_t = std::chrono::system_clock::to_time_t(timestamp_);
    json << "  \"timestamp\": \"" << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ") << "\",\n";
    
    // Add context
    if (!context_.empty()) {
        json << "  \"context\": {\n";
        bool first = true;
        for (const auto& [key, value] : context_) {
            if (!first) json << ",\n";
            json << "    \"" << key << "\": \"" << value << "\"";
            first = false;
        }
        json << "\n  },\n";
    }
    
    // Add stack trace
    if (!stack_trace_.empty()) {
        json << "  \"stack_trace\": [\n";
        for (size_t i = 0; i < stack_trace_.size(); ++i) {
            if (i > 0) json << ",\n";
            json << "    \"" << stack_trace_[i] << "\"";
        }
        json << "\n  ]\n";
    }
    
    json << "}";
    return json.str();
}

std::string YirageException::to_detailed_string() const {
    std::ostringstream detail;
    detail << "YirageException: " << message_ << "\n";
    detail << "Component: " << component_ << "\n";
    detail << "Function: " << function_ << "\n";
    detail << "Location: " << file_ << ":" << line_ << "\n";
    
    auto time_t = std::chrono::system_clock::to_time_t(timestamp_);
    detail << "Time: " << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
    
    if (!context_.empty()) {
        detail << "Context:\n";
        for (const auto& [key, value] : context_) {
            detail << "  " << key << ": " << value << "\n";
        }
    }
    
    if (!stack_trace_.empty()) {
        detail << "Stack Trace:\n";
        for (const auto& frame : stack_trace_) {
            detail << "  " << frame << "\n";
        }
    }
    
    return detail.str();
}

void YirageException::capture_stack_trace() {
    // Platform-specific stack trace capture
    // Implementation would use backtrace() on Unix or CaptureStackBackTrace() on Windows
    // For brevity, simplified implementation
    stack_trace_.push_back(function_ + " (" + file_ + ":" + std::to_string(line_) + ")");
}

// Specific exception implementations
BuildConfigurationError::BuildConfigurationError(const std::string& message,
                                                const std::string& missing_component,
                                                const std::string& suggestion)
    : YirageException(message, "BuildSystem") {
    if (!missing_component.empty()) {
        add_context("missing_component", missing_component);
    }
    if (!suggestion.empty()) {
        add_context("suggestion", suggestion);
    }
}

DependencyError::DependencyError(const std::string& dependency_name,
                               Type error_type,
                               const std::string& expected_version,
                               const std::string& actual_version)
    : YirageException("Dependency error: " + dependency_name, "DependencyManager"),
      error_type_(error_type) {
    add_context("dependency", dependency_name);
    add_context("error_type", std::to_string(static_cast<int>(error_type)));
    if (!expected_version.empty()) {
        add_context("expected_version", expected_version);
    }
    if (!actual_version.empty()) {
        add_context("actual_version", actual_version);
    }
}

}} // namespace yirage::core
```

### 2. Usage Examples

```cpp
// Example: Error handling in build system
try {
    detect_cuda_installation();
} catch (const DependencyError& e) {
    if (e.get_error_type() == DependencyError::Type::MISSING) {
        logger->warning("CUDA not found, using CPU-only mode");
        enable_cpu_only_mode();
    } else {
        logger->error(e.to_detailed_string());
        throw;
    }
}

// Example: Performance monitoring with automatic error handling
auto logger = YIRAGE_GET_LOGGER("optimizer");
{
    YIRAGE_TIME_OPERATION(logger, "kernel_optimization");
    
    try {
        auto result = optimize_kernel(input);
        YIRAGE_INFO(logger, "Kernel optimization completed successfully");
        return result;
    } catch (const OptimizationError& e) {
        YIRAGE_ERROR(logger, e.to_detailed_string());
        
        if (e.get_phase() == OptimizationError::Phase::CODEGEN) {
            YIRAGE_WARNING(logger, "Falling back to reference implementation");
            return fallback_kernel_implementation(input);
        }
        throw;
    }
}

// Example: Circuit breaker usage
CircuitBreaker gpu_circuit_breaker(5, std::chrono::seconds(30));

auto result = gpu_circuit_breaker.call([&]() {
    return execute_gpu_kernel(input);
});
```

This comprehensive error handling and logging system provides:

1. **Structured exception hierarchy** with rich context
2. **Advanced logging framework** with multiple sinks and async processing  
3. **Error recovery mechanisms** with retry, fallback, and circuit breaker patterns
4. **Performance monitoring** integrated with logging
5. **Production-ready features** like log rotation, JSON formatting, and configurable levels

The system ensures that all errors are properly captured, logged, and handled according to configurable recovery strategies, making the YICA/YiRage project robust and maintainable in production environments.
