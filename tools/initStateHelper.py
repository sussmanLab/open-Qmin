""" This script allows you to supply functions of position 
    as initial conditions for the nematic director field. 
    
    Step 1. Supply three strings for the functions 
    giving the x,y,z director components, along with 
    system size and nematic degree of order.
    
    Step 2. Run the script from the command line as 
        python3 initStateHelper.py
    The script will create an initial configuration file 
    where the Q-tensor has this director field and uniform, 
    uniaxial nematic order.
    
    Step 3. You can then load the configuration file into 
    open-Qmin with the command-line flag 
        --initialConfigurationFile /path/to/my_init_file,
    or in the GUI by clicking File -> load Configuration. 
"""

from ConversionUtilityInitState import create_init_state

Lx = 50 # system size
Ly = 50
Lz = 50
S = 0.53 # uniaxial order for all sites
filename="../my_init_file" # file prefix for where to save output

"""
    Below, replace the example functions with your 
    functions for each director component,
    inside quotes, using Python and/or NumPy functions.
    The allowed variables are:
    - Cartesian: x, y, z.
        These are relative to a simulation box corner
        and will range over the values [0,Lx-1], [0,Ly-1],
        and [0,Lz-1] respectively. 
    - Cylindrical polar: rho, phi, z.
        These are relative to the center of the bottom face of 
        the box.
    - Spherical polar: r_sph, theta, phi.
        These are relative to the box center, with r_cyl
        in units of lattice spacings.        
"""
nx_function_string = "cos(rho*2*pi/50)*cos(phi)" 
ny_function_string = "cos(rho*2*pi/50)*sin(phi)" 
nz_function_string = "sin(rho*2*pi/50)"

# don't change this part 
create_init_state(
    Lx, Ly, Lz, S, filename, 
    nx_function_string, ny_function_string, nz_function_string
)

# TODO: automate mpi division

