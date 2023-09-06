cmake_minimum_required(VERSION 3.10)

project(ME759-OpenMP C CXX)

find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_executable(parallelLoop parallelLoop.cpp)

add_executable(forLoopEx1 forLoopEx1.cpp)

add_executable(forLoopEx2 forLoopEx2.cpp)

add_executable(barrierExample barrierExample.cpp)
target_link_libraries(barrierExample gomp)

add_executable(affinity affinity.cpp)
target_link_libraries(affinity gomp)

add_executable(collapse collapse.cpp)
target_link_libraries(collapse gomp)
