cmake_minimum_required(VERSION 2.6)
project(solver-playground)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(solve_wsmp STATIC solve_wsmp.cpp solve_wsmp.h)
#target_link_libraries(solve_wsmp)
set(LIBS solve_wsmp ${CMAKE_CURRENT_SOURCE_DIR}/lib/libwsmp64.a)

option(WITH_CUDA "enable cuda" OFF)

find_package(CUDA QUIET)
if (CUDA_FOUND AND WITH_CUDA)
	message(STATUS "CUDA found")
	add_definitions(-DCUDA_FOUND)
	set(CUDA_NVCC_FLAGS "--compiler-bindir=/opt/gcc-4.4")
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
	cuda_add_library(solve_cuda solve_cusp.cu solve_cusparse.cu STATIC)
	set(LIBS ${LIBS} solve_cuda -lcusparse)
endif (CUDA_FOUND AND WITH_CUDA)

add_executable(solve main.cpp csr_matrix.h)
target_link_libraries(solve ${LIBS} -lpthread -lgfortran -lblas)
