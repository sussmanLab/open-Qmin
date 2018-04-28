cmake_minimum_required(VERSION 3.0.0)
project(dDimensionalSimulation)

add_definitions(-DDIMENSION=3)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} -Wall -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -std=c++11")


add_subdirectory(src)

include_directories(inc src src/utilities)
add_executable(simulationExecutable.out dDimensionalSimulation.cpp)
target_link_libraries(simulationExecutable.out utility)
