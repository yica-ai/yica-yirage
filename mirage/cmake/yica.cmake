# YICA Architecture Support for Mirage
# This cmake file configures YICA-specific optimizations

# YICA configuration options
option(ENABLE_YICA "Enable YICA architecture support" OFF)
option(YICA_SIMULATION_MODE "Enable YICA simulation mode" ON)
option(YICA_RUNTIME_PROFILING "Enable YICA runtime profiling" ON)

if(ENABLE_YICA)
    message(STATUS "Enabling YICA architecture support")
    
    # YICA specific definitions
    add_definitions(-DMIRAGE_ENABLE_YICA)
    
    if(YICA_SIMULATION_MODE)
        add_definitions(-DYICA_SIMULATION_MODE)
        message(STATUS "YICA simulation mode enabled")
    endif()
    
    if(YICA_RUNTIME_PROFILING)
        add_definitions(-DYICA_RUNTIME_PROFILING)
        message(STATUS "YICA runtime profiling enabled")
    endif()
    
    # YICA specific include directories
    set(YICA_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/runtime
    )
    
    # YICA source files
    set(YICA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/architecture_analyzer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/search_space.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/cim_simulator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/yica/spm_manager.cc
    )
    
    # YICA headers
    set(YICA_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/optimizer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/architecture_analyzer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/search_space.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/cim_simulator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/mirage/yica/spm_manager.h
    )
    
    # Create YICA library target
    if(YICA_SOURCES)
        add_library(mirage_yica STATIC ${YICA_SOURCES} ${YICA_HEADERS})
        
        target_include_directories(mirage_yica PUBLIC
            ${YICA_INCLUDE_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        # Link with Mirage core libraries
        target_link_libraries(mirage_yica PUBLIC
            mirage_search
            mirage_threadblock
            mirage_transpiler
        )
        
        # Set C++ standard
        target_compile_features(mirage_yica PUBLIC cxx_std_17)
        
        # Compiler specific options
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(mirage_yica PRIVATE
                -Wall -Wextra -O3
                $<$<CONFIG:Debug>:-g -O0>
            )
        endif()
        
        message(STATUS "YICA library target created")
    endif()
    
    # YICA Python bindings
    if(BUILD_PYTHON_BINDINGS)
        set(YICA_PYTHON_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/python/mirage/_cython/yica_core.pyx
        )
        
        # Add YICA to main Python module
        list(APPEND PYTHON_SOURCES ${YICA_PYTHON_SOURCES})
    endif()
    
    # YICA tests
    if(BUILD_TESTS)
        set(YICA_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_yica_optimizer.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_cim_simulator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/yica/test_spm_manager.cc
        )
        
        foreach(test_source ${YICA_TEST_SOURCES})
            get_filename_component(test_name ${test_source} NAME_WE)
            add_executable(${test_name} ${test_source})
            target_link_libraries(${test_name} mirage_yica gtest gtest_main)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    endif()
    
    # Installation
    install(TARGETS mirage_yica
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(FILES ${YICA_HEADERS}
        DESTINATION include/mirage/yica
    )
    
else()
    message(STATUS "YICA architecture support disabled")
endif()

# YICA utility functions
function(add_yica_optimization target)
    if(ENABLE_YICA)
        target_link_libraries(${target} mirage_yica)
        target_compile_definitions(${target} PRIVATE MIRAGE_ENABLE_YICA)
    endif()
endfunction()

# Export YICA configuration
if(ENABLE_YICA)
    set(MIRAGE_YICA_ENABLED TRUE CACHE BOOL "YICA support enabled" FORCE)
    set(MIRAGE_YICA_INCLUDE_DIRS ${YICA_INCLUDE_DIRS} CACHE STRING "YICA include directories" FORCE)
else()
    set(MIRAGE_YICA_ENABLED FALSE CACHE BOOL "YICA support disabled" FORCE)
endif() 