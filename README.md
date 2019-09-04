# open Qmin

This repository contains code for performing lattice-discretized Landau deGennes liquid crystal modeling,
using single or multiple CPU or GPU resources for efficient, large-scale simulations. Included is an optional
graphical user interface for running and setting up the simulations. Usefully, the GUI has the functionality
to generate a compilable file, allowing the GUI to be used for prototyping and then directly running larger-
scale simulations.

Some doxygen documentation has been completed, but lots remains to be written. In any
event, executing the command
"doxygen doc/openQmin_documentation"
from the root directory will produce a set of html documentation files.

# Basic compilation

See "INSTALLATION.md" for a few more details. Note that the "noQT" branch has stripped away all code
related to the GUI for easy installation in environments without QT installed or where it is not necessary (e.g., on clusters)

# Basic use

## To use the GUI

build/openQminGUI.out

## Running on the command line

build/openQmin.out options...

For example, to run 100 FIRE minimization steps on a cubic lattice of side length 250:
build/openQmin.out -i 100 -l 250 

To do the same thing but using a GPU in slot 0:
build/openQmin.out -i 100 -l 250 -g 0

To run 100 FIRE minimization steps on a cubic lattice of side length 500, split across 8 processors, each handling an
eighth of the simulation volume:
mpirun -n 8 build/openQmin.out -i 100 -l 250 

To load a file, e.g. "asests/boundaryInput.txt"  with custom boundaries prepared for a cubic lattice of side length 80
build/openQmin.out -i 100 -l 80 --boundaryFile assets/boundaryInput.txt

To do the above, but also save the post-minimization state:
./build/openQmin.out -i 80 --boundaryFile assets/boundaryInput.txt -l 80 --saveFile data/saveTesting

(for the above two lines, note that the file path is relative to where you currently are.)

## adding various colloids and boundaries to the command-line executable

A separate header file exists in the main directory of the repository, "addObjectsToOpenQmin.h", which exists just to
give the user a sense of how to add some of the pre-defined objects to the simulation. See the comments in that file for
more details.

# Project information
Here are some convenient links to a variety of general information about the landau deGUI project; all
of the below can also be accessed from the @ref projectInfo tab (links work on the gitlab.io
documenation website). Alternately, you can look at the markdown (.md) files in the base directory and the doc/markdown/
directories

[Basic information about the code base and operation](@ref basicinfo)

[Installation guide](@ref install)

[Sample code snippets](@ref code)

[Contributing to landau-deGUI](@ref contrib)

[Citations](@ref cite)

[Open-source information](@ref license)

[Changelog](@ref changelog)

[Contributors](@ref contributorList)
