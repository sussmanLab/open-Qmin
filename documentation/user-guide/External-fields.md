# External fields

External electric fields $\mathbf{E}$ and magnetic fields $\mathbf{H}$ contribute the following free energy densities, respectively, calculated at each site in the nematic:

$$ f_E = -\tfrac{1}{2} \varepsilon_0 E_i \varepsilon_{ij} E_j, \qquad \varepsilon_{ij} = \varepsilon \delta_{ij} + \Delta \varepsilon Q_{ij} $$

$$ f_H = - \tfrac{1}{2} \mu_0 H_i \chi_{ij} H_j, \qquad \chi_{ij} = \chi \delta_{ij} + \Delta \chi Q_{ij} $$ 

where

* $\varepsilon_0$ is the non-dimensionalized vacuum permittivity
* $\varepsilon_{ij}$ is the material's dielectric tensor
* $\varepsilon$ is the material's dielectric constant (relative permittivity)
* $\Delta \varepsilon$ is the material's dielectric anisotropy 
* $\mu_0$ is the magnetic permeability of free space
* $\chi_{ij}$ is the material's magnetic susceptibility tensor
* $\chi$ is the isotropic part of the material's magnetic susceptibility 
* $\Delta \chi$ is the anisotropic part of the material's magnetic susceptibility

and $\delta_{ij}$ is the Kronecker delta.

## Uniform fields

For a uniform external electric field $\mathbf{E}$ or magnetic field $\textbf{H}$, you can use the following command-line flags:

* For electric field:
    * `--eFieldDeltaEpsilon <scalar>`  $\rightarrow \Delta \varepsilon$ 
    * `--eFieldEpsilon <scalar>` $\rightarrow \varepsilon$ 
    * `--eFieldEpsilon0 <scalar>` $\rightarrow \varepsilon_0$ 
    * `--eFieldZ <scalar>` $\rightarrow E_z$
    * `--eFieldY <scalar>` $\rightarrow E_y$ 
    * `--eFieldX <scalar>` $\rightarrow E_x$ 
 * For magnetic field:
     * `--hFieldDeltaChi <scalar>` $\rightarrow \Delta \chi$
    * `--hFieldChi <scalar>` $\rightarrow \chi$ 
    * `--hFieldMu0 <scalar>` $\rightarrow \mu_0$
    * `--hFieldZ <scalar>` $\rightarrow H_z$
    * `--hFieldY <scalar>` $\rightarrow H_y$
    * `--hFieldX <scalar>` $\rightarrow H_x$
    
    
## Non-uniform, static fields

You can specify an external magnetic field that has arbitrary spatial dependence (but is still fixed in time) by reading in an external file or files with the command-line flag

    --spatiallyVaryingFieldFile my_field_file_title

where the files "my_field_file_title_x0y0z0.txt", [etc. for MPI](Command-Line_Usage.html#specifying-mpi-jobs), contain the magnetic field vector's components at each site, one site per line, with each line formatted as 

    x y z Hx Hy Hz    