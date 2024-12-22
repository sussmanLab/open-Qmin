add_library(forces STATIC 
    baseForce.cpp
    baseLatticeForce.cpp
    landauDeGennesLC.cpp
    landauDeGennesLCDistortionTerms.cpp
    landauDeGennesLCOtherForces.cpp
    )
set_target_properties(forces PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(forces dataTypesAndContainersLibrary)

if(CMAKE_CUDA_COMPILER)
add_library(forcesGPU STATIC 
    baseLatticeForce.cu
    landauDeGennesLC.cu
    )
set_target_properties(forcesGPU PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
endif()
