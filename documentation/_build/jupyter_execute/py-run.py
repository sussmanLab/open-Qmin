#!/usr/bin/env python
# coding: utf-8

# # Running openQmin (Python interface)

# Using *tools/runHelper.py*, you can define command-line parameters through a Python dictionary `runHelper.params`. These, along with any default parameters you didn't change, are converted by `runHelper.get_runcmd()` into a command-line string that calls *build/openQmin.out*.
# 
# The dictionary keys of `runHelper.params` are the same as the long forms (appearing after the `--`s) of the command-line flags seen when you run `build/openQmin.out --help`, with the following exceptions:
# * `help` itself is not a key in `runHelper.params`
# * Parameters `'whole_Lx'`, `'whole_Ly'`, and `'whole_Lz'`, which define the system size **before** subdivision over MPI ranks, override `'Lx'`, `'Ly'`, and `'Lz'` by default. If you want to use `'Lx'`, `'Ly'`, `'Lz'` instead (which give the system size on each rank), you can pass `do_partition=False` to `runHelper.get_runcmd()`.
# 
# In the example below, we'll make use of an example boundaryFile that we created in the page on [Boundary conditions (Python interface)](py-boundaries) and the example initialConfigurationFiles that we created in the page on [Initialization (Python interface)](py-initialization). 
# 
# Notice that the main openQmin directory path, assigned to `runHelper.directory`, is automatically prepended to the filepaths for imported and exported data. This directory path should be either an absolute path or relative to where you'll be running the command.

# In[1]:


from sys import path
path.append("../tools/")  # path to runHelper.py

import runHelper

runHelper.directory = "../" # path to openQmin main directory
runHelper.mpi_num_processes = 3  # set to 1 for non-MPI run

runHelper.params["boundaryFile"] = "ceiling_and_wavy_floor.txt"
runHelper.params["initialConfigurationFile"] = "my_init_file"

# choose a location and filename-prefix for this run's results
runHelper.params["saveFile"] = "data/my_example_run"
runHelper.params["iterations"] = 500  # max number of minimization timesteps

# system size BEFORE subdivision across MPI ranks:
runHelper.params["whole_Lx"] = 50
runHelper.params["whole_Ly"] = 50
runHelper.params["whole_Lz"] = 50

runcmd = runHelper.get_runcmd()  # generate command-line string
print(runcmd)


# We can run openQmin with these options by any of the following routes:
# 
# * Call `runHelper.run()`, which executes the result of `runHelper.get_runcmd()`
# * Copy and paste the string into a terminal
# * Use the `runcmd` string in a Python script via `import os; os.system(runcmd)`
# * Run as shell command in a Jupyter notebook with `!{runcmd}`

# In[2]:


runHelper.run()


# Let's take a look at the result using openViewMin.

# In[3]:


# Install openViewMin dependencies as needed
get_ipython().system('{sys.executable} -m pip install pandas "PyQt5<5.14" pyvistaqt tqdm imageio-ffmpeg >> /dev/null')


# In[4]:


# replace with your path to openViewMin:
sys.path.append("../../openviewmin/")

import openViewMin
import glob

# collect all files from this run
savedFiles = glob.glob("../data/my_example_run*") 
# generate plot off-screen
nplot = openViewMin.NematicPlot(savedFiles, off_screen=True, window_size=(800, 800))
# rotate plane of director glyphs to y-normal
nplot.update_filter("director_plane", {"normal":[0,1,0]}, update_self_actor=True)
# reduce lighting intensity a bit
nplot.set_lights_intensity(0.6)
# display in notebook
display(nplot.to_pythreejs())
nplot.close()

