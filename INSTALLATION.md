# Installation {#install}

# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using CUDA-8.0.
Default compilation is via QT, so you need that, and CMake, too. Multirank simulations tested with openMPI 4.0.0

Note that there is a non-GUI version of the code (particularly intended for running on clusters), in which all 
of the QT dependencies have been stripped out. This is the "noQT" branch of the code in the git repository

# Basic compilation

* cd into build/ directory. 
* $ cmake ..
* $ make

## executables created

By default, the above steps will create two executables, "openQmin.out" and "openQminGUI.out", in the build directory.
The GUI executable will... launch the graphical user interface. The non-GUI executable is meant to be run with various 
command line parameters.

To make additional executables on compilation, copy a cpp file into the base directory, and then add the name of the 
cpp file to the base CMakeList.txt file in the "foreach()" line immediately following the comment that says
"list the names of cpp files cooresponding to linked executables you'd like..."

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads

CMAKE: https://cmake.org

QT: https://www.qt.io/
