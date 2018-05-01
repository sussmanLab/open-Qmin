include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(force baseForce.cpp harmonicBond.cpp)

