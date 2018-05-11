if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(force
    baseForce.cpp
    basePairwiseForce.cpp
    harmonicRepulsion.cpp
    lennardJones6_12.cpp
    harmonicBond.cpp
    harmonicAngle.cpp
    )
cuda_add_library(forceGPU
    lennardJones6_12.cu
    harmonicRepulsion.cu
    )
