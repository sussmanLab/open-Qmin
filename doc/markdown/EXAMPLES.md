# Code snippets {#code}

These mimimal examples show how to run custom cpp files. As mentioned, just add the cpp file to the main directory,
add the name of the cpp file to the base CMakeLists.txt in the "foreach()" line, then compile from the build directory.

# examples/minimizationTiming.cpp

This file has been used to make timing information about finding minima in the presence of various objects and
boundary conditions (i.e., to create figures 2 and 3 of the "Fast, scalable, and interactive software for
Landau-de Gennes numerical modeling of nematic topological defects" paper. A slight modification of it was used
on the Comet XSEDE cluster to make Fig. 4.


# examples/speedScalaing.cpp

A file that quickly evaluate how the code performance scales with system size.

# examples/hedgehogTesting.cpp

This file was used both in single-GPU and multi-CPU mode to test the relative stability of 
dipolar vs quadrupolar defects around a spherical colloid. This was used to make Fig. 5A and 5B
of the "Fast, scalable,...." paper.
