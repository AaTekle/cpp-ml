cmake_minimum_required(VERSION 3.16)
project(parallel_kmeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(parallel_kmeans main.cpp)
target_link_libraries(parallel_kmeans PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(parallel_kmeans PRIVATE /O2 /W4)
else()
    target_compile_options(parallel_kmeans PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
