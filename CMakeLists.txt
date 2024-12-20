cmake_minimum_required(VERSION 3.22) #present from, e.g., ubuntu 22.04 on

project(openQmin VERSION 0.11
                DESCRIPTION "a base project structure that allows for copmilation with or without cuda"
                LANGUAGES CXX)

#generate compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wno-unused-function -Wextra -Wno-deprecated-copy -Wno-sign-compare -Wno-unused-parameter")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif()

if(DEFINED ENABLE_QT)
    # Tell CMake to run moc when necessary:
    set(CMAKE_AUTOMOC ON)
    # As moc files are generated in the binary dir, tell CMake to always look for includes there:
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    find_package(Qt5 COMPONENTS Core Gui Widgets OpenGL)
    find_package(OpenGL)
else()
    message(STATUS "Disabling QT compilation")
endif()

# Check for CUDA, and if it is present, set a project-wide variable
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

find_package(MPI REQUIRED)

add_definitions(-DDIMENSION=5)
add_definitions(-DDIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")


set(myLibraries
    dataTypesAndContainersLibrary
    utilities forces #simulations  models updaters
    )
if(ENABLE_CUDA)
set( myLibraries "${myLibraries} utilitiesGPU forcesGPU modelGPU updatersGPU
    ")
endif()


# set(myLibs Forces Model Simulation Updaters Utilities)
include_directories(
    inc
    src/model src/forces src/simulation src/updaters src/utilities
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#     ${Qt5Widgets_INCLUDE_DIRS}
#     ${OPENGL_INCLUDE_DIRS}
#     ${OPENGL_INCLUDE_DIRS}/GL
#     /usr/include/GL
    ${MPI_INCLUDE_PATH}
)


add_subdirectory(src)

message(STATUS "${myLibraries}")

add_executable(${PROJECT_NAME}
    test.cpp
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${myLibraries})



# set(myLibs Forces Model Simulation Updaters Utilities)
# include_directories(
#     inc
#     ${FORCESLIB_DIR}
#     ${MODELLIB_DIR}
#     ${SIMULATIONLIB_DIR}
#     ${UPDATERSLIB_DIR}
#     ${UTILITIESLIB_DIR}
#     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#     ${Qt5Widgets_INCLUDE_DIRS}
#     ${OPENGL_INCLUDE_DIRS}
#     ${OPENGL_INCLUDE_DIRS}/GL
#     /usr/include/GL
#     ${MPI_INCLUDE_PATH})
# link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)
#
# message("libraries = " "${myLibs}")
#
# qt5_wrap_ui(UI_HEADERS mainwindow.ui)
# set(SOURCES mainwindow.cpp oglwidget.cpp)
# set(HEADERS
#        mainwindow.h
#        oglwidget.h
#     )
#
# # list the names of cpp files corresponding to linked executables you'd like... This first set is
# # for cpp files which DO NOT need QT
# foreach(ARG openQmin customScriptFromGUI)
# add_executable("${ARG}.out" "${ARG}.cpp" )
# target_link_libraries("${ARG}.out" ${myLibs} ${MPI_LIBRARIES} -lnvToolsExt)
# if(MPI_COMPILE_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()
# if(MPI_LINK_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()
# endforeach()
#
# # list the names of cpp files corresponding to linked executables you'd like... This second set is
# # for GUI-related cpp files
# foreach(ARG openQminGUI)
# add_executable("${ARG}.out" "${ARG}.cpp" ${SOURCES} ${HEADERS} ${UI_HEADERS})
# target_link_libraries("${ARG}.out" ${myLibs} ${OPENGL_LIBRARIES}
#     ${MPI_LIBRARIES}
#     Qt5::Widgets
#     Qt5::Core
#     Qt5::Gui
#     -lnvToolsExt
#     )
# qt5_use_modules("${ARG}.out" Widgets)
# if(MPI_COMPILE_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()
# if(MPI_LINK_FLAGS)
#     set_target_properties("${ARG}.out" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()
# endforeach()
