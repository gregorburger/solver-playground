cmake_minimum_required(VERSION 2.6)
project(solver-playground)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "--compiler-bindir=/opt/gcc-4.4")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
cuda_add_executable(solve main.cpp solve_cusp.cu solve_cusparse.cu csr_matrix.h)
target_link_libraries(solve -lcusparse)
