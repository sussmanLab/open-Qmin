cmake_minimum_required(VERSION 3.0.0)
project(dDimensionalSimulation)

add_definitions(-DDIMENSION=3)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")


find_package(CUDA REQUIRED)
message("nvcc flags are  " ${CUDA_NVCC_FLAGS})
message("cuda root is " ${CUDA_INCLUDE_DIRS})


add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} src src/utilities)
link_directories(${CUDA_LIBRARY_DIRS})

add_executable(simulationExecutable.out dDimensionalSimulation.cpp)
target_link_libraries(simulationExecutable.out utility model ${CUDA_LIBRARIES})
