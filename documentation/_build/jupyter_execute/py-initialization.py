#!/usr/bin/env python
# coding: utf-8

# # Initialization (Python interface)

# ## Demo for usage of *tools/initHelper.py*
# 
# One option for initialization of openQmin is to specify the nematic director at each nematic site, storing this information in an initialConfigurationFile that we will import using either the command-line flag
# 
#     --initialConfigurationFile /path/to/my_init_file
#     
# or by clicking *File -> load Configuration* in the GUI. 
# 
# *tools/initHelper.py* is a Python script to automate creation of the initialConfigurationFile, using NumPy to define functions of position 
# as initial conditions for the nematic director field. Supply three strings for the functions 
# giving the *x, y, z* director components, along with 
# system size and nematic degree of order. The script will create an initialConfigurationFile
# where the Q-tensor has this director field and uniform, 
# uniaxial nematic order.

# In[1]:


from sys import path
path.append("../tools/")  # <-- replace with your path to initHelper.py

from initHelper import create_init_state


# First let's define the setup and a name (with path) for our initConfigurationFiles. We're using 3 MPI processes for this example, but you can also run it with `mpi_num_processes = 1`. 

# In[2]:


# system size *before* MPI subdivision
whole_Lx = 50
whole_Ly = 50
whole_Lz = 50
S = 0.53  # uniaxial order for all sites
mpi_num_processes = 3  # set to 1 for non-MPI run
# file path, without ".txt", for initialConfigurationFile(s)
initfilename_prefix = "../my_init_file"


# Now we define functions for our director components, formatted as **strings** that will be evaluated as expressions by NumPy. This example creates a cholesteric helix, with pitch equal to the system's $z$-height and pitch axis along the $z$ direction.

# In[3]:


# change these to your own functions:
nx_function_string = f"cos(2 * pi * z / {whole_Lz})"
ny_function_string = f"sin(2 * pi * z / {whole_Lz})"
nz_function_string = "0"


# The above example strings illustrate a few important points:
# 
# * The strings will be evaluated after `from numpy import *`, so you can use any NumPy function (`cos`, `sin`, etc.) and constants such as `pi`.
# * Using Python f-strings, we can include values of variables (such as `whole_Lz`) in the string.
# * Coordinates (in this case, `z`) will be transformed to arrays automatically.
# * Constant values (such as `0` in this case for $n_z$) are allowed.

# The allowed coordinate variables are:
# - **Cartesian**: `x`, `y`, `z`.
#     These are relative to a simulation box corner
#     and will range over the values [0,Lx-1], [0,Ly-1],
#     and [0,Lz-1] respectively. 
# - **Cylindrical polar**: `rho`, `phi`, `z`.
#     These are relative to the center of the bottom face of 
#     the box.
# - **Spherical polar**: `r_sph`, `theta`, `phi`.
#     These are relative to the box center, with `r_sph`
#     in units of lattice spacings.        
# 

# Now we call *initHelper.py*'s function `create_init_state` to generate the initialConfigurationFiles. If using MPI, you'll see that one file is created for each MPI rank.

# In[4]:


initfilenames = create_init_state(
    (whole_Lx, whole_Ly, whole_Lz),
    S,
    initfilename_prefix,
    nx_function_string,
    ny_function_string,
    nz_function_string,
    mpi_num_processes=mpi_num_processes,
)
print(initfilenames)


# Let's take a look at our initial configuration. If you're running this Jupyter notebook locally, you can set `off_screen=False` for more interactivity.

# In[ ]:


# Install openViewMin dependencies as needed
get_ipython().system('{sys.executable} -m pip install pandas "PyQt5<5.14" pyvistaqt tqdm imageio-ffmpeg >> /dev/null')


# In[ ]:


# replace with your path to openViewMin
path.append("../../openviewmin/")

import openViewMin

nplot = openViewMin.NematicPlot(initfilenames, off_screen=True, window_size=(800, 800))
display(nplot.to_pythreejs())
nplot.close()

