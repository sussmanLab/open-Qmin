cmake_minimum_required(VERSION 3.0.0)
project(landauDeGUI)

find_package(MPI REQUIRED)

add_definitions(-DDIMENSION=5)

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)

# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr
                                        -arch=sm_30
                                        -gencode=arch=compute_30,code=sm_30
                                        -gencode=arch=compute_50,code=sm_50
                                        -gencode=arch=compute_52,code=sm_52
                                        ")
                                        #")#-gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets ")#-Xptxas -O3 -Xcompiler -O3")

find_package(CUDA REQUIRED)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

message("cxx flags = " "${CMAKE_CXX_FLAGS}")
message("nvcc flags = " "${CUDA_NVCC_FLAGS}")
message("lib dir = " "${CUDA_LIBRARY_DIRS}")
message("culibs = " "${CUDA_LIBRARIES}")
#message("nvcc flags are  " ${CUDA_NVCC_FLAGS})
#message("cuda root is " ${CUDA_INCLUDE_DIRS})

add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} /home/user/CGAL/CGAL-4.9/include/CGAL/ src src/utilities src/simulation src/model src/forces src/updaters ${MPI_INCLUDE_PATH})
link_directories(${CUDA_LIBRARY_DIRS})

set(myLibs
    force
    forceGPU
    updater
    updaterGPU
    simulation
    model
    modelGPU
    utility
    utilityGPU
    )
message("libraries = " "${myLibs}")

set(SOURCES)
set(HEADERS
    )

# list the names of cpp files corresponding to linked executables you'd like...
foreach(ARG minimizationTiming speedScaling hedgehogTesting cubicLatticeNematic)
add_executable("${ARG}.out" "${ARG}.cpp" ${SOURCES} ${HEADERS} ${UI_HEADERS})
target_link_libraries("${ARG}.out" ${myLibs} ${CUDA_LIBRARIES}
    ${MPI_LIBRARIES}
    )
if(MPI_COMPILE_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
endforeach()
