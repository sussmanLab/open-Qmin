cmake_minimum_required(VERSION 3.0.0)
project(dDimensionalSimulations)

add_definitions(-DDIMENSION=3)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")


find_package(CUDA REQUIRED)
message("nvcc flags are  " ${CUDA_NVCC_FLAGS})
message("cuda root is " ${CUDA_INCLUDE_DIRS})


add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} src src/utilities src/simulation src/model src/forces src/updaters)
link_directories(${CUDA_LIBRARY_DIRS})

set(myLibs 
    force 
    updater 
    updaterGPU
    simulation 
    model 
    utility 
    utilityGPU 
    )
message("libraries = " ${myLibs})

add_executable(simulationExecutable.out dDimensionalSimulation.cpp)
target_link_libraries(simulationExecutable.out ${myLibs} ${CUDA_LIBRARIES})

#add_executable(otherExecutable.out fixedTopologyInteractions.cpp)
#target_link_libraries(otherExecutable.out ${myLibs} ${CUDA_LIBRARIES})


#file(GLOB appSrcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
#foreach(testSourceFile ${appSrcs})
#    string( REPLACE ".cpp" ".out" testname ${testSourceFile})
#    message("evaluating " ${testSourceFile} " " ${testname})
#    add_executable(${testname} ${CMAKE_CURRENT_SOURCE_DIR}${testsourcefile} )
#    target_link_libraries( ${testname} force updater simulation model utility ${CUDA_LIBRARIES} )
#endforeach( testSourceFile ${appSrcs} )
