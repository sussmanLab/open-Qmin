# Installation {#install}

## Requirements

The current iteration of the code was written using some features of C++17, and was compiled using CUDA-10.0 and CUDA 12, with a relatively modern version of cmake

Multirank simulations tested with openMPI 4.0.0.

The GUI needs Qt6 and also openGL.

A flag during the cmake phase can be used to not compile the GUI (particularly useful when you just want to run the command-line code on a cluster), and another flag can be used to build the entire project *without* any of the GPU / CUDA stuff (particularly useful, e.g., on a mac now that Apple has stopped supporting cuda). See below for those options.

## Basic compilation

Basic compilation (with CUDA support, and with the graphical user interface also being compiled):

    $ cd build/ 
    $ cmake -DENABLE_QT=ON ..
    $ make

### Compilation without the GUI

    $ cd build/ 
    $ cmake ..
    $ make

### Compilation without cuda


    $ cd build/ 
    $ cmake -DDISABLE_CUDA=ON ..
    $ make

(And of course, the cuda and QT flags are independent, so `cmake -DDISABLE_CUDA=ON -DENABLE_QT=ON ..` will work as expected)

### Compilation 

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

## Helpful websites
The requirements can be obtained by looking at the info on the following:

CUDA: https://developer.nvidia.com/cuda-downloads

CMAKE: https://cmake.org

QT: https://www.qt.io/
