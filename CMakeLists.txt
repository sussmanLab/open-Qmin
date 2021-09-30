cmake_minimum_required(VERSION 3.9.0)
project(openQmin LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)

set(CUDA_ARCH "30")
                #if you have different cuda-capable hardware, modify this line to get much more optimized performance. By default,
                #I have set this to work on Tesla K40s (still used at many XSEDE facilities), but add the correct codes for optimizing performance
                #on your cards

add_definitions(-DDIMENSION=5)
add_definitions(-DDIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")
message("dir = " "${CMAKE_CURRENT_SOURCE_DIR}")

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr -arch=sm_${CUDA_ARCH} -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

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

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

message("cxx flags = " "${CMAKE_CXX_FLAGS}")
message("cuda flags = " "${CMAKE_CUDA_FLAGS}")
message("cuda include dir = " "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set(FORCESLIB_DIR src/forces)
file(GLOB FORCESLIB_CPP ${FORCESLIB_DIR}/*.cpp)
file(GLOB FORCESLIB_CU ${FORCESLIB_DIR}/*.cu)
add_library(Forces ${FORCESLIB_CPP} ${FORCESLIB_CU})
set_target_properties(Forces PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(MODELLIB_DIR src/model)
file(GLOB MODELLIB_CPP ${MODELLIB_DIR}/*.cpp)
file(GLOB MODELLIB_CU ${MODELLIB_DIR}/*.cu)
add_library(Model ${MODELLIB_CPP} ${MODELLIB_CU})
set_target_properties(Model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(SIMULATIONLIB_DIR src/simulation)
file(GLOB SIMULATIONLIB_CPP ${SIMULATIONLIB_DIR}/*.cpp)
file(GLOB SIMULATIONLIB_CU ${SIMULATIONLIB_DIR}/*.cu)
add_library(Simulation ${SIMULATIONLIB_CPP} ${SIMULATIONLIB_CU})
set_target_properties(Simulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(UPDATERSLIB_DIR src/updaters)
file(GLOB UPDATERSLIB_CPP ${UPDATERSLIB_DIR}/*.cpp)
file(GLOB UPDATERSLIB_CU ${UPDATERSLIB_DIR}/*.cu)
add_library(Updaters ${UPDATERSLIB_CPP} ${UPDATERSLIB_CU})
set_target_properties(Updaters PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(UTILITIESLIB_DIR src/utilities)
file(GLOB UTILITIESLIB_CPP ${UTILITIESLIB_DIR}/*.cpp)
file(GLOB UTILITIESLIB_CU ${UTILITIESLIB_DIR}/*.cu)
add_library(Utilities ${UTILITIESLIB_CPP} ${UTILITIESLIB_CU})
set_target_properties(Utilities PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(myLibs Forces Model Simulation Updaters Utilities)
include_directories(
    inc
    ${FORCESLIB_DIR}
    ${MODELLIB_DIR}
    ${SIMULATIONLIB_DIR}
    ${UPDATERSLIB_DIR}
    ${UTILITIESLIB_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}/GL
    /usr/include/GL
    ${MPI_INCLUDE_PATH})
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

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
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out" ${myLibs} ${MPI_LIBRARIES} -lnvToolsExt)
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
target_link_libraries("${ARG}.out" ${myLibs} ${OPENGL_LIBRARIES}
    ${MPI_LIBRARIES}
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui
    -lnvToolsExt
    )
qt5_use_modules("${ARG}.out" Widgets)
if(MPI_COMPILE_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
endforeach()
