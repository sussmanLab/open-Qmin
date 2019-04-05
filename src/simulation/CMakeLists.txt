if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
add_library(simulation
    basicSimulation.cpp
    simulation.cpp
    multirankSimulation.cpp
    multrankSimulationBoundaries.cpp)
