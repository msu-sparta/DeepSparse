cmake_minimum_required(VERSION 3.3.2)
project(my_hpx_project CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wconversion")

find_package(HPX REQUIRED)
add_hpx_executable(my_hpx_program
    SOURCES main.cpp kernels.cpp utils.cpp
    COMPONENT_DEPENDENCIES iostreams)

include_directories(${MKLINCLUDE})
target_link_libraries(my_hpx_program PRIVATE 
    ${MKLPATH}
    "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm")
