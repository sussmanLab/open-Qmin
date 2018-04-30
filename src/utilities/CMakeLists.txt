include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
cuda_add_library(utility noiseSource.cpp noiseSource.cu utilities.cu)
