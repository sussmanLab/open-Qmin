# Installation

## Requirements

You will need 
* a C++ compiler compatible with the C++11 standard
* openMPI (development is using version 4.0.0)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (development is using version 8.0)
* [CMake](https://cmake.org)
* [Qt](https://www.qt.io/) (this dependency may be removed in future versions)

Note that there is a non-GUI version of the code (particularly intended for running on clusters), in which all 
of the QT dependencies have been stripped out. This is the "noQT" branch of the code in the git repository.

## Basic compilation

In the terminal, cd into the `build/` directory and type
        
    cmake ..
    make
            
:::{note}
By default the code will compile GPU code targeted at the (old, but still used in some XSEDE facilities) Tesla K40 cards. If you have newer GPUs, it is *highly* recommended to go to line 6 of the CMakeLists.txt file and set the `CUDA_ARCH` variable appropriately. 

You should set `CUDA_ARCH` to 10 times your GPU model's "compute capability". One place you can look this up is [this Wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported), in the leftmost column of the row where your GPU's model is listed. For example, the A100 GPU is listed there as having CUDA compute ability 8.0, so users with this model should set `CUDA_ARCH` to "80".
:::

### Compilation troubleshooting

If all required dependencies have been installed, the modern CMake build system should keep compilation difficulties to a minimum. If you installed any dependencies by hand you may have to give CMake hints about the location of various directories (see, e.g., lines 35-40 of the CMakeLists.txt file, as well as the "include_directories" list farther down in that file).

If using a GPU, be sure to check what generation your card is and set the `CUDA_ARCH` variable (line 6 of the CMakeLists.txt file) appropriately for maximum performance; see the Note just above.

If you used conda to install some packages, you may need to `conda deactivate` in order for the compilation to succeed.

If installing on a Linux cluster with software "modules", you may need to do something like

    module load cuda qt 
    
before the "basic compilation" steps above. You may also need to request an interactive session with a GPU resource; how to do so will vary from system to system.


## Executables created

By default, the above steps will create two executables, "openQmin.out" and "openQminGUI.out", in the build directory.
The GUI executable will launch the graphical user interface. Users can run "openQmin.out" from the command-line with various [command-line flags](Command-Line-Options); this is also the executable accessed by Python wrappers under development. 

Are you wondering about the mysterious third executable, "customScriptFromGUI.out", that also gets made? We currently have
an experimental (but functional) feature by which you can record sequences of actions in the GUI and save them to a new,
compilable "customScriptFromGUI.cpp" file. This can be used, for instance, to specify specific initial conditions more easily
than by fussing with the command line or writing your own cpp codes. Note that the "customScriptFromGUI.out" executable 
itself has command line options (such as changing the lattice size), and is suitable to be run as an MPI executable.

```{admonition} For the pros
:class: tip
To make additional executables on compilation, copy a cpp file into the base directory, and then add the name of the 
cpp file to the base CMakeList.txt file in the "foreach()" line immediately following the comment that says
"list the names of cpp files cooresponding to linked executables you'd like."
```

## GUI troubleshooting

Depending on your terminal and X11 set-up, there may be a variety of GL errors you might encounter
(mismatched client/server versions of openGL, libGL errors related to fbConfigs and swrast, etc.).
The developers have had luck using the "terminal" interface within the [x2go](https://wiki.x2go.org/doku.php)
client, or alternatively using [VcXserv](https://sourceforge.net/projects/vcxsrv/) with the no-WGL option.