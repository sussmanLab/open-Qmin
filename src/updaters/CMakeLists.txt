include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS} ../simulation)
add_library(updater 
    baseUpdater.cpp 
    velocityVerlet.cpp
    energyMinimizerFIRE.cpp
    )
cuda_add_library(updaterGPU
    velocityVerlet.cu
    energyMinimizerFIRE.cu
    )
