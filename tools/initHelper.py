""" Create initialConfigurationFile based on input from initStateHelper.py """

import numpy as np
from numpy import pi, sin, cos, tan, arcsin, asin, arccos, acos, arctan, atan, hypot, arctan2, atan2, degrees, radians, deg2rad, rad2deg, sinh, cosh, tanh, arcsinh, asinh, arccosh, acosh, arctanh, atanh, round, around, rint, fix, floor, ceil, trunc, nanprod, nansum, cumulative_sum, cumulative_prod, cumprod, cumsum, nancumprod, nancumsum, diff, ediff1d, gradient, cross, trapezoid, exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2, i0, sinc, signbit, copysign, frexp, ldexp, nextafter, spacing, lcm, gcd, add, reciprocal, positive, negative, multiply, divide, power, subtract, true_divide, floor_divide, fmod, mod, modf, remainder, divmod, angle, real, imag, conj, conjugate, maximum, amax, fmax, nanmax, minimum, amin, fmin, nanmin, convolve, clip, sqrt, cbrt, square, absolute, fabs, sign, heaviside, nan_to_num, real_if_close, interp, bitwise_count  # Almost all of NumPy's mathematical functions from https://numpy.org/doc/stable/reference/routines.math.html, just so users don't have to type 'np.' Exclude `min`, `max,` sum`, `pow`, to avoid collision with Python built-in functions.

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
    dims, S, filename,
    nx_function_string, ny_function_string, nz_function_string,
    mpi_num_processes=1
):
    whole_Lx, whole_Ly, whole_Lz = dims
    state_array = np.empty((whole_Lx * whole_Ly * whole_Lz, 10)) # will hold saved data

    Z, Y, X = np.meshgrid(np.arange(whole_Lz), np.arange(whole_Ly), np.arange(whole_Lx), indexing='ij')
    # coords in first three columns
    state_array[...,0:3] = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T 

    # create director array
    director_array = np.empty((whole_Lx * whole_Ly * whole_Lz, 3))
    for i, function_string in enumerate(
        (nx_function_string, ny_function_string, nz_function_string)
    ):
        director_component_function = director_component_array_from_function_string(
            function_string, whole_Lx, whole_Ly, whole_Lz
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
    
    state_array = state_array.reshape((whole_Lz, whole_Ly, whole_Lx, 10))
    
    ranks_x, ranks_y, ranks_z = partition_processors(mpi_num_processes) 
    Lx = whole_Lx // ranks_x
    Ly = whole_Ly // ranks_y 
    Lz = whole_Lz // ranks_z
    initstate_filenames = [] 
    for Rz in range(ranks_z):
        for Ry in range(ranks_y):
            for Rx in range(ranks_x):                
                full_filename = filename + f'_x{Rx}y{Ry}z{Rz}.txt'
                state_array_chunk = state_array[
                    Rz*Lz:min((Rz+1)*Lz, whole_Lz), 
                    Ry*Ly:min((Ry+1)*Ly, whole_Ly), 
                    Rx*Lx:min((Rx+1)*Lx, whole_Lx)
                ].reshape(-1, 10)                                                
                np.savetxt(full_filename, state_array_chunk, fmt='%i\t%i\t%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%i\t%.6f') # output
                print(f'Configuration for process ({Rx},{Ry},{Rz}) has been saved to {full_filename}.')
                initstate_filenames.append(full_filename)
    return initstate_filenames
