if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(force
    baseForce.cpp
    basePairwiseForce.cpp
    baseLatticeForce.cpp
    nematicInteraction.cpp
    liquidCrystalElasticity.cpp
    harmonicRepulsion.cpp
    lennardJones6_12.cpp
    harmonicBond.cpp
    harmonicAngle.cpp
    )
cuda_add_library(forceGPU
    lennardJones6_12.cu
    harmonicRepulsion.cu
    baseLatticeForce.cu
    liquidCrystalElasticity.cu
    nematicInteraction.cu
    )
