cmake_minimum_required(VERSION 2.6)
project(solver-playground)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA QUIET)
if (${CUDA_FOUND})
	add_definitions(-DCUDA_FOUND)
	set(CUDA_NVCC_FLAGS "--compiler-bindir=/opt/gcc-4.4")
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
	cuda_add_library(solve_cuda solve_cusp.cu solve_cusparse.cu)
	set(LIBS "-lcusparse cuda_solve")
endif (${CUDA_FOUND})

add_library(solve_wsmp solve_wsmp.cpp solve_wsmp.h)
set(LIBS ${LIBS} solve_wsmp)

add_executable(solve main.cpp csr_matrix.h)

target_link_libraries(solve ${LIBS})
