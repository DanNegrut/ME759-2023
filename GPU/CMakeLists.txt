cmake_minimum_required (VERSION 3.13)

project (ME759 LANGUAGES C CXX CUDA)                                                 

set(CMAKE_CUDA_HOST_COMPILER $ENV{CU_CCBIN} CACHE PATH "Cuda host compiler dir")   
                                                                                   
set(CMAKE_BUILD_TYPE, Release CACHE STRING "Build type")                           
                                                                                   
add_executable(setArray setArray.cu)
add_executable(helloWorld helloWorld.cu)
add_executable(threadfenceExample threadfenceExample.cu)
add_executable(vector_addition vector_addition.cu)

add_executable(unifMemBasicPart1 unifMemBasicPart1.cu)
add_executable(unifMemBasicPart2 unifMemBasicPart2.cu)

add_executable(unifMemCUDAexPart1 unifMemCUDAexPart1.cu)
add_executable(unifMemCUDAexPart2 unifMemCUDAexPart2.cu)
add_executable(unifMemCUDAexPart3 unifMemCUDAexPart3.cu)

add_executable(hwCoherency hwCoherency.cu)

add_executable(stencil1D stencil1D.cu)

add_subdirectory(kernelWrapper)