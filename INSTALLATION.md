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

Note: by default the code will compile gpu code targeted at the (old, but still used in some XSEDE facilities) Tesla K40 cards... if you have newer GPUs, it is highly recommended to go to 
line 6 of the CMakeLists.txt file and set the CUDA_ARCH variable appropriately

## executables created

By default, the above steps will create two executables, "openQmin.out" and "openQminGUI.out", in the build directory.
The GUI executable will... launch the graphical user interface. The non-GUI executable is meant to be run with various 
command line parameters.

Are you wondering about the mysterious third executable, "customScriptFromGUI.out", that also gets made? We currently have
an experimental (but functional) feature by which you can record sequences of actions in the GUI and save them to a new,
compilable "customScriptFromGUI.cpp" file. This can be used, for instance, to specify specific initial conditions more easily
than by fussing with the command line or writing your own cpp codes. Note that the "customScriptFromGUI.out" executable 
itself has command line options (such as changing the lattice size), and is suitable to be run as an MPI executable.

To make additional executables on compilation, copy a cpp file into the base directory, and then add the name of the 
cpp file to the base CMakeList.txt file in the "foreach()" line immediately following the comment that says
"list the names of cpp files cooresponding to linked executables you'd like..."

# Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads

CMAKE: https://cmake.org

QT: https://www.qt.io/
