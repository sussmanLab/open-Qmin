# Basic overview of the project {#basicinfo}

landau deGUI is an extension of a standard d-Dimensional molecular dynamics simulation package
written by DMS, and much of the class structure reflects this...



## Directory structure of the project

The "/inc" directory contains a bunch of classes and functions used throughout the project,
including array-like data structures for shuttling data back and forth betwen the GPU and CPU,
common Q-tensor operations, etc. The "/src" directory has subdirectories "forces", "models",
"simulation", "updaters", and "utilities". Below is a very brief description of the kinds of
things in each, more detailed descriptions of key classes are further down in this file.

### root directory

In the rood directory there are a bunch of markdown files. The "mainwindow" and "oglwidget" files are used
to construct the GUI, so if you'd like to tinker with any of the things available from the graphical side of
things that's where to go... The "cubicLatticeNematic.cpp" file actually has the main function of the executable,
and is what should be modified if you want to tinker with the non-visual operation of the code.

### models directory

"simpleModel" is a basic collection of d-dimensional degrees of freedom... "cubicLattice" specifies
neighbor relations so that these d-dimensional d.o.f.'s live on a 3D cubic lattice. "qTensorLatticeModel"
has some additional specializations of this relevant for Q-tensor simulations.

### forces directory

Forces objects are connected with a model object -- and baseForce.h defines what the pointers to force
classes can typically report -- and can compute forces and energies. baseLatticeForces know that the
model can return neighbor information. The landaDeGennes derived class implements several approximations
to a phenomenological model. The landauDeGennesLCBoundary.h header is just for convenience, because it's
so messy.

### updaters directory

Contains the classes that can update degrees of freedom in a model object. Most typically, these are equations
of motion or various energy minimization methods.

### simulation directory

Simulation objects help tie together a model with any number of forces and any number of updaters. They
provide functions of convenience for things like computing the forces, progressing an equation of motion by
a timestep, etc.

### utilities directory

Not surprisingly, contains assorted utilites. This includes things like random number generation classes,
tools to auto-tune GPU kernel parameters, etc. The utilities.cu[h] files also provide common GPU functions,
such as zeroing out arrays, performing parallel reductions, etc.
