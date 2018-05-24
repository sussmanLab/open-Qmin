# INSTALLATION {#install}

A general CMake scheme is included with the repository. 

## Basic compilation

* cd into build/ directory. 
* $ cmake ..
* $ make

## Changing the dimension

Go into the CMakeLists.txt file in the root directory, locate the line that says
"add_definitions(-DDIMENSION=3) "
and change the 3 (or whatever other number happens to be there) to something else.
This sets the dimensionality of the degrees of freedom, and by default also the 
dimensionality of the space those d.o.f.s are in.

# Requirements

The current iteration of the code was written using some features of C++11, and was compiled using CUDA-8.0.

Default compilation is via CMake, so you need that, too.


# Sample programs

This repository comes with sample main cpp files that can be compiled into executables in both the root directory
and in examples/. Please see the [examples](@ref code) documentation for details on each.
range of parameters.

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads
CMAKE: https://cmake.org
