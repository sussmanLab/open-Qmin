include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
cuda_add_library(model simple2DModel.cpp)
