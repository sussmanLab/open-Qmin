# INSTALLATION {#install}

CMake is used so that both the c++ and cuda code gets compiled nicely... 
The cmake file probably needs help finding the right QT modules

## Basic compilation

* cd into build/ directory. 
* $ cmake ..
* $ make

## Changing the dimension

# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using CUDA-8.0.

Default compilation is via QT, so you need that, and CMake, too.


# Sample programs

This repository comes with sample main cpp files that can be compiled into executables in both the root directory
and in examples/. Please see the [examples](@ref code) documentation for details on each.
range of parameters.

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads
CMAKE: https://cmake.org
QT: https://www.qt.io/
