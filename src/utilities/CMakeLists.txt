if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(utility 
    hyperrectangularCellList.cpp
    kernelTuner.cpp
    noiseSource.cpp 
    neighborList.cpp
    )
cuda_add_library(utilityGPU
    noiseSource.cu 
    utilities.cu 
    hyperrectangularCellList.cu
    neighborList.cu
    )
