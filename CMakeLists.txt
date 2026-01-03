cmake_minimum_required(VERSION 3.18)
project(mini_image_pipe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources
set(LIB_SOURCES
    src/memory_manager.cu
    src/task_graph.cpp
    src/scheduler.cu
    src/pipeline.cpp
    src/operators/color_convert.cu
    src/operators/resize.cu
    src/operators/sobel.cu
    src/operators/gaussian_blur.cu
)

# Main library
add_library(mini_image_pipe STATIC ${LIB_SOURCES})
target_include_directories(mini_image_pipe PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mini_image_pipe PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA runtime
target_link_libraries(mini_image_pipe PUBLIC cudart)

# Testing
enable_testing()
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "GTest found, building tests")
    
    # Test sources
    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_memory_manager.cpp
        tests/test_color_convert.cpp
        tests/test_resize.cpp
        tests/test_sobel.cpp
        tests/test_gaussian_blur.cpp
        tests/test_task_graph.cpp
        tests/test_scheduler.cpp
        tests/test_pipeline.cpp
    )
    
    # Test executable
    add_executable(mini_image_pipe_tests ${TEST_SOURCES})
    target_link_libraries(mini_image_pipe_tests 
        mini_image_pipe 
        GTest::gtest 
        GTest::gtest_main
    )
    set_target_properties(mini_image_pipe_tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    add_test(NAME mini_image_pipe_tests COMMAND mini_image_pipe_tests)
else()
    message(STATUS "GTest not found, skipping tests")
endif()

# Example executable
add_executable(demo_pipeline examples/demo_pipeline.cpp)
target_link_libraries(demo_pipeline mini_image_pipe)
set_target_properties(demo_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS mini_image_pipe
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
