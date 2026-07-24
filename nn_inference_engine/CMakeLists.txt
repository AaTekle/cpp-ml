cmake_minimum_required(VERSION 3.16)
project(nn_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(nn_inference_engine main.cpp)

if(MSVC)
    target_compile_options(nn_inference_engine PRIVATE /O2 /W4)
else()
    target_compile_options(nn_inference_engine PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
