cmake_minimum_required (VERSION 3.13)

project(GPU-ME759 LANGUAGES C CXX CUDA)                                                 

set(CMAKE_CUDA_HOST_COMPILER $ENV{CU_CCBIN} CACHE PATH "Cuda host compiler dir")   
                                                                                   
set(CMAKE_BUILD_TYPE, Release CACHE STRING "Build type")                           
                                                                                   
add_executable(kernelWrapper main.cpp kern.cu cudalloc.hpp kern.h)

target_compile_options(kernelWrapper PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
