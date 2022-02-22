# Change log

### open-Qmin version 0.9 -- in progress

* Switch to GUI initialization according to Frank constants rather than landau-deGennes ones
* Corrections to the metric used to compute forces in the non-orthogonal basis of Qxx, Qxy, Qxz, Qyy, Qyz
* Update to CMAKE files in response to community feedback
* Substantial command-line improvements and user-friendliness

### open-Qmin version 0.8

* Extensive minor improvements in advance of paper submission
* Creation of noQT branch to maintain a version without the QT dependency
* Slight change to the logic of distortion energy when more than L1 is used

### landauDeGUI version 0.7

* multi-rank simulations possible via openMPI functionality

### landauDeGUI version 0.6
* various boundary conditions implemented
* 1- 2- and 3- constant approximation to the distortion term implemented
* semi-reasonable visualizations added


### landauDeGUI version 0.5

* functional Q-tensor on a cubic lattice solver
* QT-based GUI

### dDimensionalSimulation version 0.1

* hypercubic celllist and neighbor list implemented
* d-dimensional simulations of simple pairwise potentials
* fixed-topology interactions permitted
* nve, nvt, and energy minimization enabled
* compile-time setting of dimensionality via root CMakeLists.txt file
