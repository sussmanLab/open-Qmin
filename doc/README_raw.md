# open-Qmin

open-Qmin is an open-source code base for performing lattice-discretized Landau-deGennes modeling of
liquid crystal systems. It has been developed as a collaboration between the research groups of 
[Daniel Sussman](https://www.dmsussman.org/) and [Daniel Beller](https://bellerphysics.gitlab.io/).
If you have any comments, want to contribute to this project, or simply want to share how you're using
our code to do cool stuff feel free to send an email to either of us!

This repository contains code for performing lattice-discretized Landau-de Gennes liquid crystal modeling,
using single or multiple CPU or GPU resources for efficient, large-scale simulations. Included is an optional
graphical user interface for running and setting up the simulations. Usefully, the GUI has the functionality
to generate a compilable file, allowing the GUI to be used for prototyping and then directly running larger-
scale simulations.

A user guide to open-Qmin can be found at [https://sussmanlab.github.io/open-Qmin/](https://sussmanlab.github.io/open-Qmin/).

Some doxygen documentation has been completed, but lots remains to be written. In any
event, executing the command
`doxygen doc/openQmin_documentation`
from the root directory will produce a set of html documentation files.

# Basic compilation

See "INSTALLATION.md" for a few more details. Note that the "noQT" branch has stripped away all code
related to the GUI for easy installation in environments without QT installed or where it is not necessary (e.g., on clusters)

As noted in the INSTALLATION file, for maximum performance set CUDA_ARCH in line 6 of the CMakeLists.txt file to correspond to the card you want to use

# Troubleshooting

If you are having troubles compiling or using the open-Qmin Graphical User Interface, before contacting the developers (who are happy to help!)
we recommend trying the following "standard" troubleshooting steps:

## Compilation troubleshooting

If all required dependencies have been installed, the modern CMake build system should keep compilation difficulties to a minimum. If you installed any dependencies by hand you may have to give CMake hints about the location of various directories (see, e.g., lines 35-40 of the CMakeLists.txt file).

If using a GPU, be sure to check what generation your card is and set the "CUDA_ARCH'' variable (line 6 of the CMakeLists.txt file) appropriately for maximum performance. [This Website](https://en.wikipedia.org/wiki/CUDA) may help if you are unsure what to use.

If you used conda to install some packages, you may need to `conda deactivate` in order for the compilation to succeed.

## GUI troubleshooting

Depending on your terminal and X11 set-up, there may be a variety of GL errors you might encounter
(mismatched client/server versions of openGL, libGL errors related to fbConfigs and swrast, etc.).
The developers have had luck using the "terminal" interface within the [x2go](https://wiki.x2go.org/doku.php)
client, or alternatively using [VcXserv](https://sourceforge.net/projects/vcxsrv/) with the no-WGL option.

# Basic use

## To use the GUI

`build/openQminGUI.out`

## Running on the command line

build/openQmin.out options...

For example, to run 100 FIRE minimization steps on a cubic lattice of side length 250:  
`build/openQmin.out -i 100 -l 250`

To do the same thing but using a GPU in slot 0:  
`build/openQmin.out -i 100 -l 250 -g 0`


To load a file, e.g. "asests/boundaryInput.txt"  with custom boundaries prepared for a cubic lattice of side length 80  
`build/openQmin.out -i 100 -l 80 --boundaryFile assets/boundaryInput.txt`


To do the above, but also save the post-minimization state:  
`./build/openQmin.out -i 80 --boundaryFile assets/boundaryInput.txt -l 80 --saveFile data/saveTesting`

(for the above two lines, note that the file path is relative to where you currently are.)

## using the command line to specify how the RNG will be used

First, by default the executable compiled from openQmin.cpp will use a reproducible random number generator with a fixed initial seed. To use a random number as the seed to the random number generator, use the -r flag, eg:
`build/openQmin.out -i 100 -l 250 -r`

To specify a specific seed to use (so that, eg., you can reproducibly study an ensemble of different random conditions), use the --randomSeed command line option, eg:
`build/openQmin.out -i 100 -l 250 --randomSeed 123456234`


## Using the command line to specify MPI jobs

As noted above, the "-l" command line flag can be used to specify the side length of a cubic simulation
domain (other flags exist if you want the domain to be a rectangular prism). When using MPI to run a larger
simulation domain, the default behavior is to take the command line size to specify the domain size ON EVERY
MPI RANK. (One could imagine an alternate default behavior in which the command line specified the TOTAL size of the
simulation domain, which was then divided among the different processors).

So, for example, the command:  
`mpirun -n 1 build/openQmin.out -l 100`  
will run execute a simulation domain of total size (100x100x100) lattice sites, on a single rank.


Moving to two processors, the command:  
`mpirun -n 2 build/openQmin.out -l 100`  
will run execute a simulation domain of total size (200x100x100) lattice sites, on two ranks (that
is, each rank will control a 100x100x100 domain). 


Similarly the command:  
`mpirun -n 8 build/openQmin.out -l 100`  
will run execute a simulation domain of total size (200x200x200) lattice sites, where each of the
eight ranks continues to control a block of 100x100x100 lattice sites.

## saving states and reading the output

Both the command-line and gui executables can save the current configuration of the simulation, and simple visualization
tools are available in the /visualizationTools directory to see the output of these systems. The output format is a simple txt file, where each line is:
`x y z Qxx Qxy Qxz Qyy Qyz t d`,
where `x` `y` `z` are the lattice site (in the case of MPI simulations, each rank will produce a separate 
file with `x` `y` `z` in absolute coordinates), followed by the current values of the Q tensor. The integer
`t` specifies the "type" of the lattice site (bulk, boundary, part of an object), and `d` is a defect
measure that is computed for all lattice sites not part of an object. By default this will be the largest
eigenvalue of the Q tensor at that site; for lattice sites that are part of an object this will always be zero.

## adding various colloids and boundaries to the command-line executable

A separate header file exists in the main directory of the repository, "addObjectsToOpenQmin.h", which exists just to
give the user a sense of how to add some of the pre-defined objects to the simulation. See the comments in that file for more details.

### Preparing a custom boundary file

Both the command-line and gui versions of the executable have the ability to read in a user-prepared
file that can specify objects composed of arbitrary collections of lattice sites and anchoring conditions
(rather than just being limited to the predefined walls and colloids described in the addObjectsToOpenQmin.h file).


The input txt file specifying such objects must be very precisely formatted, and in the /tools/ directory
we have included a simple utility to assist with this (although any other method can be used to generate
the desired txt file, e.g. via python).  A few examples of these custom boundary files are in the /assets/
directory; these were all created using the included mathematica notebook.

The formatting requirements are the following (copying from identical information provided in the comments of src/simulation/multirankSimulationBoundaries.cpp):

The first line must be a single integer specifying the number of objects to be read in.
This is meant in the colloquial English sense, not in the zero-indexed counting sense. So, if you want your file to specify one object, make sure the first line is the number 1.

Each subsequent block must be formatted as follows (with no additional lines between the blocks):
The first line MUST be formatted as
`a b c d`
where `a=0` means oriented anchoring (such as homeotropic or oriented planar), `a=1` means degenerate Planar
`b` is a scalar setting the anchoring strength
`c` is the preferred value of $S_0$
`d` is an integer specifying the number of sites, $N_B$.

Subsequently, there MUST be $N_b$ lines of the form 
`x y z C1 C2 C3 C4 C5`,
where `x`, `y`, and `z` are the integer lattice sites, and `C1`, `C2`, `C3`, `C4`, `C5` are real numbers
corresponding to the desired anchoring conditions:
For oriented anchoring, `C1`, `C2`, `C3`, `C4`, `C5` correspond directly to the surface-preferred Q-tensor:
`C1` = Qxx, `C2` = Qxy, `C3` = Qxz, `C4` = Qyy, `C5` = Qyz,
where one often will set the Q-tensor by choosing a locally preferred director, $\nu^s$, and setting
$Q^B = 3 S_0/2 * (\nu^s \nu^s - \delta_{ab}/3)$.

For degenerate planar anchoring the five constants should be specified as,
`C1` = $\hat{\nu}^s_x$
`C2` = $\hat{\nu}^s_y$
`C3` = $\hat{\nu}^s_z$
`C4` = $0.0$
`C5` = $0.0$,
where $\nu^s = \{\cos\phi \sin\theta, \sin\phi sin\theta, cos\theta\}$
is the direction to which the LC should try to be orthogonal.

# Project information
Here are some convenient links to markdown files containing a variety of general information about the open-Qmin project.

[Basic information about the code base and operation](doc/markdown/BasicInformation.md)

[Installation guide](INSTALLATION.md)

[Sample code snippets](doc/markdown/EXAMPLES.md)

[Contributing to open-Qmin](doc/markdown/CONTRIBUTING.md)

[Citations](CITATIONS.md)

[Open-source information](LICENSE.md)

[Changelog](ChangeLog.md)

[Contributors](doc/markdown/ContributorList.md)
