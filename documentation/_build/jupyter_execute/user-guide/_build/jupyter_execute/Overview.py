#!/usr/bin/env python
# coding: utf-8

# # open-Qmin
# *Fast, scalable, customizable software for Landau-de Gennes numerical modeling of nematic liquid crystals and their topological defects*

# ## Overview 
# 
# [Open-Qmin](https://github.com/sussmanLab/open-Qmin.git) is an open-source code base for numerical modeling of nematic liquid crystals and their topological defects at equilibrium, by relaxation of a finite-difference lattice discretization of the Landau-de Gennes free energy. This project aims to be:
# 
# * **Fast:** Allows utilization of GPU or CPU resources, and uses efficient minimization routines like the Fast Inertial Relaxation Engine (FIRE).
# 
# * **Scalable:** Automates MPI parallelization, allowing users to exploit as many CPU cores as they have available in order to access supra-micron scales often closer to experimental dimensions.
# 
# * **Customizable:** Users can specify boundary conditions, external fields, and material properties, usually without requiring recompilation. Common boundaries like planar walls or colloidal inclusions can be added with provided recipes; or users can define custom boundary geometries site-by-site.
# 
# A [peer-reviewed article in *Frontiers in Physics*](https://www.frontiersin.org/articles/10.3389/fphy.2019.00204/full) describes open-Qmin's theoretical approach, numerical implementation, and basic usage. Some information is more up-to-date in the documentation you're reading here.

# ## Visualization
# 
# A Python-based 3D visualization environment called "openViewMin" is under development as a visualization companion to open-Qmin. It will supersede the visualization routines described in the *Frontiers in Physics* article. This documentation will make use of open-ViewMin to examine example results from open-Qmin simulations. 

# ## Citation
# 
# If you use open-Qmin for a publication or project, please cite:
# 
# > Daniel M. Sussman and Daniel A. Beller. "Fast, scalable, and interactive software for Landau-de Gennes numerical modeling of nematic topological defects." *Frontiers in Physics* **7** 204 (2019). [DOI: 10.3389/fphy.2019.00204](https://doi.org/10.3389/fphy.2019.00204)

# ## Authors
# 
# open-Qmin is developed as a collaboration between the research groups of [Daniel Sussman](https://www.dmsussman.org/) (Emory University) and [Daniel Beller](https://bellerphysics.gitlab.io/) (Johns Hopkins University). We very much welcome users to contribute to the project and to tell us how you're using open-Qmin and how it might be improved.

# ## Acknowledgements
# 
# The first release of open-Qmin was supported by
# 
# * National Science Foundation Grant POLS-1607416
# * Simons Foundation Grant 454947
# * NVIDIA Corporation
# * XSEDE Grant NSF-TG-PHY190027
# * The Multi-Envrionment Computer for Exploration and Discovery (MERCED) cluster at UC Merced, funded by NSF grant ACI-1429783
# 
# Ongoing development is supported by
# 
# * National Science Foundation Grant DMR-2046063
# 
