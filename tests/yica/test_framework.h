#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

// 简单的测试框架宏定义

#define BUILTIN_TEST(name) \
    class YICABasic_##name##_Test { \
    public: \
        void TestBody(); \
    }; \
    void YICABasic_##name##_Test::TestBody()

#define EXPECT_TRUE_MSG(condition, msg) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << msg << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define EXPECT_EQ_MSG(expected, actual, msg) \
    do { \
        if ((expected) != (actual)) { \
            std::ostringstream oss; \
            oss << msg << " - Expected: " << (expected) << ", Actual: " << (actual) << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define EXPECT_FALSE_MSG(condition, msg) \
    do { \
        if (condition) { \
            std::ostringstream oss; \
            oss << msg << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define EXPECT_NE_MSG(expected, actual, msg) \
    do { \
        if ((expected) == (actual)) { \
            std::ostringstream oss; \
            oss << msg << " - Both values: " << (expected) << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define EXPECT_GT_MSG(val1, val2, msg) \
    do { \
        if (!((val1) > (val2))) { \
            std::ostringstream oss; \
            oss << msg << " - " << (val1) << " should be > " << (val2) << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define EXPECT_GE_MSG(val1, val2, msg) \
    do { \
        if (!((val1) >= (val2))) { \
            std::ostringstream oss; \
            oss << msg << " - " << (val1) << " should be >= " << (val2) << " (line " << __LINE__ << ")"; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0) 