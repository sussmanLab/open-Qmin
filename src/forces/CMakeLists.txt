add_library(forces STATIC 
    baseForce.cpp
    baseLatticeForce.cpp
    landauDeGennesLC.cpp
    landauDeGennesLCDistortionTerms.cpp
    landauDeGennesLCOtherForces.cpp
    )
set_target_properties(forces PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
if(ENABLE_CUDA)
add_library(forcesGPU STATIC 
    baseLatticeForce.cu
    landauDeGennesLC.cu
    )
set_target_properties(forces PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
endif()
