add_library(models STATIC 
    simpleModel.cpp
    qTensorLatticeModel.cpp
    multirankQTensorLatticeModel.cpp
    cubicLattice.cpp
    )
target_link_libraries(models utilities)
set_target_properties(models PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_CUDA_COMPILER)
add_library(modelsGPU STATIC 
    simpleModel.cu
    qTensorLatticeModel.cu
    multirankQTensorLatticeModel.cu
    cubicLattice.cu
    )
set_target_properties(modelsGPU PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
endif()
