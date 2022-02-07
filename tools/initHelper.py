""" Create initialConfigurationFile based on input from initStateHelper.py """

import numpy as np
from numpy import * # just so users don't have to type 'np.'

def director_component_array_from_function_string(
    function_string, Lx, Ly, Lz
):
    """ Translate user-supplied function strings into arrays for director components. """
    # convert polar coords to Cartesian
    function_string = function_string.replace('theta', 'np.arctan2(rho, z-(Lz-1)/2)')
    function_string = function_string.replace('r_sph', 'np.sqrt((z-(Lz-1)/2)**2 + rho**2)')
    function_string = function_string.replace('rho', 'np.sqrt((x-(Lx-1)/2)**2 + (y-(Ly-1)/2)**2)')
    function_string = function_string.replace('phi', 'np.arctan2(y-(Lx-1)/2, x-(Ly-1)/2)')
    function_string = function_string.replace('Lx', str(Lx))
    function_string = function_string.replace('Ly', str(Ly))
    function_string = function_string.replace('Lz', str(Lz))
    def director_component_function(x,y,z):
        return 0*x + eval(function_string) # 0*x ensures correct shape when function is a constant
    return director_component_function

def Q_array_from_director_array(director_array, S):
    """ director to Q-tensor conversion """
    return 3/2 * S * np.array([
        director_array[...,0]**2 - 1/3,
        director_array[...,0]*director_array[...,1],
        director_array[...,0]*director_array[...,2],
        director_array[...,1]**2 - 1/3,
        director_array[...,1]*director_array[...,2]
    ]).T

def partition_processors(mpi_num_processes):
    """ determine xyz layout of MPI divisions """
    z = int(np.floor(mpi_num_processes**(1/3)))
    nLeft = int(np.floor(mpi_num_processes/z))
    y = int(np.floor(np.sqrt(nLeft)))
    x = int(np.floor(nLeft/y))
    return (x,y,z)

def create_init_state(
    Lx, Ly, Lz, S, mpi_num_processes, filename,
    nx_function_string, ny_function_string, nz_function_string
):
    state_array = np.empty((Lx*Ly*Lz, 10)) # will hold saved data

    Z, Y, X = np.meshgrid(np.arange(Lz), np.arange(Ly), np.arange(Lx))
    # coords in first three columns
    state_array[...,0:3] = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T 

    # create director array
    director_array = np.empty((Lx*Ly*Lz, 3))
    for i, function_string in enumerate(
        (nx_function_string, ny_function_string, nz_function_string)
    ):
        director_component_function = director_component_array_from_function_string(
            function_string, Lx, Ly, Lz
        )
        director_array[...,i] = director_component_function(X, Y, Z).ravel()
    
    # ensure director field is normalized
    director_norms = np.sqrt(np.sum(director_array**2, axis=-1))
    for i in range(3):
        director_array[...,i] /= director_norms
    
    # turn director array into Q-tensor array
    state_array[...,3:8] = Q_array_from_director_array(director_array, S)

    state_array[...,8] = 0 # no info about boundaries at this stage
    state_array[...,9] = S # all sites have same initial uniaxial order
    
    state_array = state_array.reshape((Lz, Ly, Lx, 10))
    
    ranks_x, ranks_y, ranks_z = partition_processors(mpi_num_processes)    
    for Rz in range(ranks_z):
        for Ry in range(ranks_y):
            for Rx in range(ranks_x):                
                full_filename = filename + f'_x{Rx}y{Ry}z{Rz}.txt'
                state_array_chunk = state_array[(Rx-1)*Lx:Rx*Lx, (Ry-1)*Ly:Ry*Ly, (Rz-1)*Lz:Rz*Lz].reshape(-1, 10)                                                
                np.savetxt(full_filename, state_array_chunk, fmt='%i\t%i\t%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%i\t%.6f') # output
                print(f'Configuration for process ({Rx},{Ry},{Rz}) has been saved to {full_filename}.')
    
