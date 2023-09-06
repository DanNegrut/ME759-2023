cmake_minimum_required (VERSION 3.13)

project(CUBstudies LANGUAGES C CXX CUDA)                                                 

set(CMAKE_CUDA_HOST_COMPILER $ENV{CU_CCBIN} CACHE PATH "Cuda host compiler dir")   
                                                                                   
set(CMAKE_BUILD_TYPE, Release CACHE STRING "Build type")                           
                                                                                   
#include_directories(${YOUR_DIRECTORY})

set(CUBDIR "C:/Users/Dan/DansRepos/DanGitCAE/GPU-related/cub-1.8.0")
include_directories(${CUBDIR})

add_executable( reduceByKey reduceByKey.cu )
target_compile_options(reduceByKey PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
set_target_properties(reduceByKey PROPERTIES  CXX_STANDARD 17)

add_executable( deviceReduce deviceReduce.cu )
target_compile_options(deviceReduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
set_target_properties(deviceReduce PROPERTIES  CXX_STANDARD 17)

add_executable( sortByKey sortByKey.cu )
target_compile_options(deviceReduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)
set_target_properties(deviceReduce PROPERTIES  CXX_STANDARD 17)


