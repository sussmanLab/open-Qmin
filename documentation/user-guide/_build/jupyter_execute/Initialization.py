#!/usr/bin/env python
# coding: utf-8

# # Initialization
# 
# ## Command-line or C++-code control 
# 
# Two command-line flags control initialization: `-z` and `--initialConfigurationFile`. The `-z` flag allows you to set the variable `initializationSwitch` in "setInitialConditions.h"; users can make changes in that file and recompile openQmin.cc as needed. The 
# * `-z 0 ` the default behavior: for each lattice site pick a random director uniformly over the unit sphere and make a Q tensor corresponding to it (using magnitude of the order parameter S0)
# * `-z -1` use the loadState function to load an initial configuration from a saved file. In order to correctly load mpi-based jobs, the file names must be in the specific format of “fileName_xAyBzC.txt”, where A,B,and C are integers corresponding to how openQmin splits up multirank jobs (so, even if you are not using MPI, your file must be named “myFileName_x0y0z0.txt”, and you would load the file with the command “sim->loadState(“myFileName”);”. Furthermore, the format of the file must be precisely the text format that openQmin saves files as.
# :::{note}
# Using EITHER `-z -1` or `--initialConfigurationFile myfilename` will trigger this initialization path. With `-z -1` the program looks to "setIninitialConditions.h" for the initial state.
# :::
# * `-z 1` pick a single random director and make a uniform nematic texture in that direction
# * `-z 2` choose a specific uniform texture to initialize (modify the function below to hard-code it)
# * `-z 3` define a function, $f(x,y,z)$, that contols both the initial director and the initial $S_0$ parameter to which to set the system. (Modify the function "localDirectors" in "setInitialConditions.h" to the desired functions of position.)
