add_library(dataTypesAndContainersLibrary STATIC 
    gpuarray.cpp
    cudaDataTypes.cpp
    )
set_target_properties(dataTypesAndContainersLibrary PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

add_library(utilities STATIC 
    fileGenerator.cpp
    kernelTuner.cpp
    hyperrectangularCellList.cpp
    neighborList.cpp
    noiseSource.cpp
    utilities.cpp
    )
set_target_properties(utilities PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
if(ENABLE_CUDA)
add_library(utilitiesGPU STATIC 
    hyperrectangularCellList.cu
    neighborList.cu
    noiseSource.cu
    utilities.cu
    )
set_target_properties(utilitiesGPU PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
endif()
