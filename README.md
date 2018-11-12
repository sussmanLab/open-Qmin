# landauDeGUI

This repository contains code that (a) calculates liquid crystal stuff (in a phenomenological,
landau deGennes Q-tensor framework), and also (b) provides an (optional, but on by default)
graphical user interface for running and setting up the simulations. It is a GPU-accelerated
package, although in can also be run in CPU-only mode (indeed, one does not even need a GPU
on the machine running it, although CUDA is still required for compilation).

Some doxygen documentation has bee written, but lots of stuff is still to be written. In any
event, executing the command
"doxygen doc/landauDeGUI_documentation"
from the root directory will produce a set of html documentation files that may (or may not!)
be helpful.

# Running in non-visual mode

By default, once compiled landauDeGUI will launch a GUI that will let you initialize and interact
with a simulation of Q-tensors living on a 3D cubic lattice. It can also be launched from the command
line with the additional flag "-v" (no arguments)... a long sequence of other command line flags and
arguments can then be appended to control the program behavior. In this way landauDeGUI can be used
on a cluster. If default compilation has been done, then from the build directory executing:
./landauDeGUI.out --help
will output a list of possible command line options.

# Basic compilation

see "INSTALLATION.md" for a few more details.

## Project information
Here are some convenient links to a variety of general information about the landau deGUI project; all
of the below can also be accessed from the @ref projectInfo tab (links work on the gitlab.io
documenation website)

[Basic class overview](@ref basicinfo)

[Installation guide](@ref install)

[Contributing to cellGPU](@ref contrib)

[Citations](@ref cite)

[Open-source information](@ref license)

[cellGPU version information](@ref changelog)

[Contributors](@ref contributorList)
