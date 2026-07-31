# Command-Line Options

## System setup

For a cubic simulation box, you can use `--boxL <int>` or `-l <int>`.

For simulation boxes with unequal dimensions, you will have to separately specify `--Lx <int> --Ly <int> --Lz <int>`.

### Boundaries 

By default, the simulation box includes no boundaries, and periodic boundary conditions are applied at the box walls. To add boundaries such as walls and colloidal particles, one option is to uncomment lines from "addObjectsToOpenQmin.h", edit as necessary, and recompile open-Qmin. 

Another option, which does not require recompilation, is to define the boundaries' geometries and anchoring conditions in a separate file and import this using `--boundaryFile <filename>`. See [here](Boundary-conditions.html#preparing-a-custom-boundary-file) for tips on how to prepare this file. 

## Initialization

Two command-line flags control initialization: `-z` and `--initialConfigurationFile`. The `-z` flag allows you to set the variable `initializationSwitch` in "setInitialConditions.h"; users can make changes in that file and recompile open-Qmin as needed. The `-z` flag has the following options:
* `-z 0`: The default behavior: for each lattice site pick a random director uniformly over the unit sphere and make a Q-tensor corresponding to it (using nematic degree of order `S0`).
* `-z -1`: Use the loadState function to load an initial configuration from a saved file. 
<a class="anchor" id="MPI-file-naming"></a>
:::{note}
In order to correctly load MPI-based jobs, the file names must be in the specific format of “fileName_x$X$y$Y$z$Z$.txt”, where $X$, $Y$, and $Z$ are integers corresponding to how open-Qmin splits up multirank jobs. (Even if you are not using MPI, your file must be named “myFileName_x0y0z0.txt”, and you would load the file with the code `sim->loadState(“myFileName”);`.) Furthermore, the format of the file must be precisely the [text format used by open-Qmin to save the system's state](Saved-output). 
:::
:::{note}
Using EITHER `-z -1` or `--initialConfigurationFile myfilename` will trigger this initialization path. With `-z -1` the program looks to "setIninitialConditions.h" for the initial state.
:::
* `-z 1`: Initialize with a uniform nematic director along a randomly chosen direction. 
* `-z 2`: Initialize with a uniform nematic director along a specified direction. (Modify the components of the variable `targetDirector` in "setInitialConditions.h", then recompile.)
* `-z 3`: Define a function, $f(x,y,z)$, that controls both the initial director and the initial $S_0$ parameter at each site. (Modify the function `localDirectors` in "setInitialConditions.h" to the desired functions of position, then recompile.)

## Minimization

### Choosing CPU or GPU 

If you have one GPU and want to use it, add the flag `--GPU 0` or `-g 0`. 

If you have multiple GPUs, specify which GPU to use via `--GPU <int>` or `-g <int>`.

If you want to use CPU resources, leave the default value `--GPU -1` or `-g -1`. 


### Minimizer step size

`--deltaT <float>` or `-e <float>` sets the minimizer's step size. The default value is 0.005.

### Stopping conditions 

The minimizer seeks the equilibrium configuration where the net force at each site is zero. We set a small value for the summed norm of residual forces, at which we are near enough to equilibrium to quit successfully, using `--fTarget <float>`, with default value  1e-12.

However, the minimizer will stop before reaching this force threshold if the number of minimization steps reaches the value set by `--iterations <int>` or `-i <int>`, with defalt value 100. 

## Material constants

The non-dimensionalized coefficients $A$, $B$, $C$ of the Landau-de Gennes [bulk free energy density](Landau-de-Gennes.html#bulk-free-energy) are respectively set via 

* `--phaseConstantA <float>` or `-a <float>`
* `--phaseConstantB <float>` or `-b <float>`
* `--phaseConstantC <float>` or `-c <float>`

:::{note}
Generally, $A$ and (typically) $B$ will be negative. open-Qmin expects you to provide the **signed** value. For example, the default value $A=-0.172$ is equivalent to `-a -0.172`.
:::


The non-dimensionalized distortion coefficients $L_i$, $i=1,2,3,4,6$ of the Landau-de Gennes [distortion free energy density](Landau-de-Gennes.html#distortion-free-energy) are respectively set via 

* `--L1 <float>`
* `--L2 <float>`
* `--L3 <float>`
* `--L4 <float>`
* `--L6 <float>`

## Output

Choose a unique name (including relative filepath but *excluding* the filetype suffix such as ".txt") to save the results of a given run, using 
`--saveFile <string>`. For example, `--saveFile data/my_example_run` will cause results to be saved in "data/my_example_run_x0y0z0.txt", [etc. for MPI](#MPI-file-naming). See the [Saved Output](Saved-output) page for the format of the saved files. 


### Saving time sequences

By default, open-Qmin saves the system state only from the last step of the minimization. To keep a record of the minimization process, you can use `--linearSpacedSaving <N (int)>` to save to unique files every $N$ steps. For example, `--saveFile data/my_example_run --linearSpacedSaving 10` will create files "data/my_example_run_t0_x0y0z0.txt", "data/my_example_run_t10_x0y0z0.txt", "data/my_example_run_t20_x0y0z0.txt", etc.

Alternatively, you may wish to save the state at smaller time intervals early in the minimization but at longer time intervals later on (where the "dynamics" generally slows down). For this purpose, you can use `--logSpacedSaving <F (float)>` which will save a unique file whenever the timestep $t$ reaches or exceeds $F^j$ for $j=0,1,2,3,\dots$

### Data stride

For larger systems, saving the state of every site can be expensive in terms of computing time, storage space, and analysis later on. To save the state of only every $n$ sites along *each* direction, use `--stride <n (int)>`. 

## Random number generation

By default, the executable compiled from openQmin.cpp will use a reproducible random number generator with a fixed initial seed. To use a random number as the seed to the random number generator, use the -r flag, e.g.:
`build/openQmin.out -i 100 -r`.

To specify a specific seed to use (so that you can reproducibly study an ensemble of different random conditions, for example), use the `--randomSeed` command line option, e.g.:
`build/openQmin.out -i 100 --randomSeed 123456234`.

## Glossary of command-line options and their default values 

```
##    VARIABLE_NAME = DEFAULT_VALUE, # <VARIABLE_TYPE> DESCRIPTION
initializationSwitch = 0, # <int> an integer controlling program branch
GPU = -1, # <int> which gpu to use
phaseConstantA = -0.172, # <float> value of phase constant A
phaseConstantB = -2.12, # <float> value of phase constant B
phaseConstantC = 1.73, # <float> value of phase constant C
deltaT = 0.0005, # <float> step size for minimizer
fTarget = 1e-12, # <float> target minimization threshold for norm of residual forces
iterations = 100, # <int> maximum number of minimization steps
randomSeed = -1, # <int> seed for reproducible random number generation
L1 = 4.64, # <float> value of L1 term
L2 = 4.64, # <float> value of L2 term
L3 = 4.64, # <float> value of L3 term
L4 = 4.64, # <float> value of L4 term
L6 = 4.64, # <float> value of L6 term
boxL = 50, # <int> number of lattice sites for cubic box
Lx = 50, # <int> number of lattice sites in x direction
Ly = 50, # <int> number of lattice sites in y direction
Lz = 50, # <int> number of lattice sites in z direction
initialConfigurationFile = "", # <string> carefully prepared file of the initial state of all lattice sites
spatiallyVaryingFieldFile = "", # <string> carefully prepared file containing information on a spatially varying external H field
boundaryFile = "", # <string> carefully prepared file of boundary sites
saveFile = "", # <string> the base name to save the post-minimization configuration
linearSpacedSaving = -1, # <int> save a file every x minimization steps
logSpacedSaving = -1, # <float> save a file every x^j for integer j
stride = 1, # <int> stride of the saved lattice sites
hFieldX = 0, # <float> x component of external H field
hFieldY = 0, # <float> y component of external H field
hFieldZ = 0, # <float> z component of external H field
hFieldMu0 = 1, # <float> mu0 for external magenetic field
hFieldChi = 1, # <float> Chi for external magenetic field
hFieldDeltaChi = 0.5, # <float> mu0 for external magenetic field
eFieldX = 0, # <float> x component of external E field
eFieldY = 0, # <float> y component of external E field
eFieldZ = 0, # <float> z component of external E field
eFieldEpsilon0 = 1, # <float> epsilon0 for external electric field
eFieldEpsilon = 1, # <float> Epsilon for external electric field
eFieldDeltaEpsilon = 0.5, # <float> DeltaEpsilon for external electric field
```