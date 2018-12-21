if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS} ../../inc)
add_library(model
    simpleModel.cpp
    cubicLattice.cpp
    qTensorLatticeModel.cpp
    multirankQTensorLatticeModel.cpp)
cuda_add_library(modelGPU
    simpleModel.cu
    cubicLattice.cu
    qTensorLatticeModel.cu
    multirankQTensorLatticeModel.cu)
