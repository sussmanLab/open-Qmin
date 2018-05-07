if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(simulation basicSimulation.cpp simulation.cpp)
