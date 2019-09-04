cmake_minimum_required(VERSION 3.0.0)
project(openQmin)

find_package(MPI REQUIRED)

add_definitions(-DDIMENSION=5)
add_definitions(-DDIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")
message("dir = " "${CMAKE_CURRENT_SOURCE_DIR}")

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
                                        #if you have different cuda-capable hardware, feel free to add the correct codes for optimizing performance

#if CMake is complaining about missing packages that you know you have, feel free to give CMake hints about the correct directory
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -Wno-deprecated-declarations")
    set(Qt5_DIR "/Users/dmsussma/Qt/5.11.2/clang_64/lib/cmake/Qt5")
    set(Qt5OpenGL_DIR "/Users/dmsussma/Qt/5.11.2/clang_64/lib/cmake/Qt5OpenGL")
    find_package(Qt5OpenGL REQUIRED)
    #find_package(Qt5 COMPONENTS Core Gui Widgets 3DCore 3DExtras 3DRender 3DInput OpenGL)
    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL)
    find_package(OpenGL)
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    #set(Qt5_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5")
    #set(Qt5Core_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Core")
    #set(Qt5Gui_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Gui")
    #set(Qt5Widgets_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Widgets")
    #find_package(Qt5Widgets REQUIRED)
    #find_package(Qt5OpenGL REQUIRED)
    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL)
    find_package(OpenGL)
endif()

add_definitions( ${QT_DEFINITIONS} )

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

add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS}/GL /home/user/CGAL/CGAL-4.9/include/CGAL/ src src/utilities src/simulation src/model src/forces src/updaters ${MPI_INCLUDE_PATH} /usr/include/GL)
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

qt5_wrap_ui(UI_HEADERS mainwindow.ui)
set(SOURCES mainwindow.cpp oglwidget.cpp)
set(HEADERS
       mainwindow.h
       oglwidget.h
    )

# list the names of cpp files corresponding to linked executables you'd like... This first set is
# for cpp files which DO NOT need QT
foreach(ARG openQmin customScriptFromGUI)
add_executable("${ARG}.out" "${ARG}.cpp")
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

# list the names of cpp files corresponding to linked executables you'd like... This second set is
# for GUI-related cpp files
foreach(ARG openQminGUI)
add_executable("${ARG}.out" "${ARG}.cpp" ${SOURCES} ${HEADERS} ${UI_HEADERS})
target_link_libraries("${ARG}.out" ${myLibs} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES}
    ${MPI_LIBRARIES}
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui
    )
qt5_use_modules("${ARG}.out" Widgets)
if(MPI_COMPILE_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
endforeach()
