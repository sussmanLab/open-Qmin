cmake_minimum_required(VERSION 3.0.0)
project(landauDeGUI)

add_definitions(-DDIMENSION=5)

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)

# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(Qt5_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5")
set(Qt5Core_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Core")
set(Qt5Gui_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Gui")
set(Qt5Widgets_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt5Widgets")
set(Qt53DCore_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt53DCore")
set(Qt53DRender_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt53DRender")
set(Qt53DInput_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt53DInput")
set(Qt53DExtras_DIR "/home/user/Qt/5.11.2/gcc_64/lib/cmake/Qt53DExtras")
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets 3DCore 3DExtras 3DRender 3DInput)
find_package(OpenGL)
add_definitions( ${QT_DEFINITIONS} )

set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}
                                        -arch=sm_30
                                        -gencode=arch=compute_30,code=sm_30
                                        -gencode=arch=compute_50,code=sm_50
                                        -gencode=arch=compute_52,code=sm_52
                                        -gencode=arch=compute_60,code=sm_60
                                        -gencode=arch=compute_61,code=sm_61
                                        -gencode=arch=compute_61,code=compute_61
                                        ")
                                        #")#-gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets ")#-Xptxas -O3 -Xcompiler -O3")

find_package(CUDA REQUIRED)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

message("cxx flags = " "${CMAKE_CXX_FLAGS}")
message("nvcc flags = " "${CUDA_NVCC_FLAGS}")
message("lib dir = " "${CUDA_LIBRARY_DIRS}")
message("culibs = " "${CUDA_LIBRARIES}")
#message("nvcc flags are  " ${CUDA_NVCC_FLAGS})
#message("cuda root is " ${CUDA_INCLUDE_DIRS})

add_subdirectory(src)

include_directories(inc ${CUDA_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS}/GL src src/utilities src/simulation src/model src/forces src/updaters)
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

add_executable(landauDeGUI.out cubicLatticeNematic.cpp ${SOURCES} ${HEADERS} ${UI_HEADERS})
target_link_libraries(landauDeGUI.out ${myLibs} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} Qt5::Widgets Qt5::Core
    Qt5::Gui
    Qt5::3DCore
    Qt5::3DExtras
    Qt5::3DRender
    Qt5::3DInput)
qt5_use_modules(landauDeGUI.out Widgets)
