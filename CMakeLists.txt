cmake_minimum_required(VERSION 3.22) #present from, e.g., ubuntu 22.04 on

project(openQmin VERSION 0.11
                DESCRIPTION "a base project structure that allows for copmilation with or without cuda"
                LANGUAGES CXX)

#generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

add_definitions(-DDIMENSION=5)
add_definitions(-DDIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")

#set a default "Release" build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

#Set c++ standard and various compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wno-unused-function -Wno-deprecated-copy -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unknown-pragmas -Wno-class-memaccess -Wno-shadow -Wno-return-type -Wno-reorder")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    message(STATUS "Compiling with debugging symbols")
    add_definitions(-DDEBUGFLAGUP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")
endif()

#Handle initial stuff if we're building the gui
if(DEFINED ENABLE_QT)
    # list(APPEND CMAKE_PREFIX_PATH /home/daniel/Qt/6.8.1/gcc_64)
    ##Add hint here
    find_package(Qt6 REQUIRED COMPONENTS OpenGLWidgets Widgets Core Gui)
    find_package(OpenGL REQUIRED)
    qt_standard_project_setup()
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
else()
    message(STATUS "Disabling QT compilation")
endif()

# Check if the user has disabled cuda support
#If not, look for on the system and, if it is present, set a project-wide variable
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if(DEFINED DISABLE_CUDA)
    message(STATUS "Disabling cuda compilation")
else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA found, building with GPU support")
        add_definitions(-DENABLE_CUDA)
        enable_language(CUDA)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        message(STATUS "CUDA not found, building without GPU support")
    endif()
endif()

if(CMAKE_CUDA_COMPILER)
    message(STATUS "setting cuda compilation of gpuarray file")
    message("cuda directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set_source_files_properties(src/utilities/gpuarray.cpp PROPERTIES LANGUAGE CUDA)
endif()


#set the list of libraries defined and built in all of the subdirectories of src/
set(myLibraries
    dataTypesAndContainersLibrary
    utilities forces models simulations  updaters
    )
if(CMAKE_CUDA_COMPILER)
    list(APPEND myLibraries utilitiesGPU forcesGPU modelsGPU updatersGPU)
endif()


find_package(MPI REQUIRED)
include_directories(
    inc
    src/model src/forces src/simulation src/updaters src/utilities
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${MPI_INCLUDE_PATH}
)

add_subdirectory(src)




add_executable(${PROJECT_NAME}
    openQmin.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${myLibraries} MPI::MPI_CXX)


# # list the names of cpp files corresponding to linked executables you'd like... This first set is
# # for cpp files which DO NOT need QT
# foreach(ARG openQmin)
# add_executable("${ARG}.out" "${ARG}.cpp" )
# target_link_libraries("${ARG}.out" ${myLibs} ${MPI_LIBRARIES} -lnvToolsExt)
# if(MPI_COMPILE_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()
# if(MPI_LINK_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()
# endforeach()

# link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)
#
if(ENABLE_QT)
    qt_add_executable(guiOpenQMin 
        openQminGUI.cpp mainwindow.cpp oglwidget.cpp
        mainwindow.ui
        )
    target_include_directories(guiOpenQMin PRIVATE
        /usr/include/GL
        )
    target_link_libraries(guiOpenQMin PUBLIC ${myLibraries} 
        Qt6::Widgets
        Qt6::Core
        Qt6::OpenGLWidgets
        Qt6::Gui
        ${OPENGL_LIBRARY}
        MPI::MPI_CXX
        )
endif()
