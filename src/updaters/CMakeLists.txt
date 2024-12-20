add_library(updaters STATIC 
    baseUpdater.cpp
    velocityVerlet.cpp
    energyMinimizerAdam.cpp
    energyMinimizerFIRE.cpp
    energyMinimizerGradientDescent.cpp
    energyMinimizerLoLBFGS.cpp
    energyMinimizerNesterovAG.cpp
    )
set_target_properties(updaters PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
if(ENABLE_CUDA)
add_library(updatersGPU STATIC 
    velocityVerlet.cu
    energyMinimizerAdam.cu
    energyMinimizerFIRE.cu
    energyMinimizerNesterovAG.cu
    )
set_target_properties(updaters PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
endif()
