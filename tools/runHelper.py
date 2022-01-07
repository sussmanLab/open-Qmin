
"""
This Python script simply calls openQmin.out in command-line mode, 
with all available command-line arguments presented as a Python dictionary.
Variables with empty-string values are ignored, as are commented-out lines. 
"""

import os
from initHelper import partition_processors

mpi_num_processes = 1

params = dict(
	initializationSwitch = 0, # <int> an integer controlling program branch
	GPU = -1, # <int> which gpu to use
	phaseConstantA = 0.172, # <float> value of phase constant A
	phaseConstantB = 2.12, # <float> value of phase constant B
	phaseConstantC = 1.73, # <float> value of phase constant C
	deltaT = 0.0005, # <float> step size for minimizer
	fTarget = 1e-12, # <float> target minimization threshold for norm of residual forces
	iterations = 100, # <int> maximum number of minimization steps
	nConstants = 1, # <int> approximation for distortion term
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

	### The following override Lx, Ly, Lz if and only if do_partition==True in get_runcmd
	whole_Lx = 50, # <int> number of lattice sites in x direction *before* MPI subdivision
	whole_Ly = 50, # <int> number of lattice sites in y direction *before* MPI subdivision
	whole_Lz = 50, # <int> number of lattice sites in z direction *before* MPI subdivision
)

  
def get_runcmd(do_partition=True):
    if do_partition: 
        # calculate partitioned simulation box size automatically 
        # from entire simulation box size and number of MPI processes
        Rx, Ry, Rz = partition_processors(mpi_num_processes)
        params['Lx'] = int(params['whole_Lx'] // Rx)
        params['Ly'] = int(params['whole_Ly'] // Ry)
        params['Lz'] = int(params['whole_Lz'] // Rz)         
    if mpi_num_processes > 1:
        runcmd = f'mpirun -n {mpi_num_processes} '       
    else:
        runcmd = ''
    runcmd += '../build/openQmin.out ' + ' '.join([
        f'--{key} {val}' for key, val in zip(
            params.keys(), params.values()
        )
        if len(str(val)) > 0
        and not key in [
            'whole_Lx', 'whole_Ly', 'whole_Lz'
        ]
    ])
    return runcmd
