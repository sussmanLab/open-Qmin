cmake_minimum_required(VERSION 3.0.0)
project(dDimensionalSimulation)

add_definitions(-DDIMENSION=3)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -Wall -Wno-unused-function -Wno-language-extension-token -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-language-extension-token -pedantic -std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)


find_package(CUDA REQUIRED)
message("nvcc flags are  " ${CUDA_NVCC_FLAGS})
message("cuda root is " ${CUDA_INCLUDE_DIRS})


add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} src src/utilities)
add_executable(simulationExecutable.out dDimensionalSimulation.cpp)
target_link_libraries(simulationExecutable.out utility)
