include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS} ../simulation)
add_library(updater baseUpdater.cpp energyMinimizerFIRE.cpp)

