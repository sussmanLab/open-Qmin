# External fields

The couplings to external electric and magnetic fields modeled by open-Qmin are described in [here](Landau-de-Gennes.html#external-fields-free-energy).

## Uniform fields

For a uniform external electric field $\mathbf{E}$ or magnetic field $\textbf{H}$, you can use the following command-line flags:

* For electric field:
    * `--eFieldDeltaEpsilon <float>`  $\rightarrow \Delta \varepsilon$ 
    * `--eFieldEpsilon <float>` $\rightarrow \varepsilon$ 
    * `--eFieldEpsilon0 <float>` $\rightarrow \varepsilon_0$ 
    * `--eFieldZ <float>` $\rightarrow E_z$
    * `--eFieldY <float>` $\rightarrow E_y$ 
    * `--eFieldX <float>` $\rightarrow E_x$ 
 * For magnetic field:
     * `--hFieldDeltaChi <float>` $\rightarrow \Delta \chi$
    * `--hFieldChi <float>` $\rightarrow \chi$ 
    * `--hFieldMu0 <float>` $\rightarrow \mu_0$
    * `--hFieldZ <float>` $\rightarrow H_z$
    * `--hFieldY <float>` $\rightarrow H_y$
    * `--hFieldX <float>` $\rightarrow H_x$
    
    
## Non-uniform, static fields

You can specify an external magnetic field that has arbitrary spatial dependence (but is still fixed in time) by reading in an external file or files with the command-line flag `--spatiallyVaryingFieldFile my_field_file_title` where the file(s) "my_field_file_title_x0y0z0.txt", [etc. for MPI](Command-Line-Options.html#MPI-file-naming), contain the magnetic field vector's components at each site, one site per line. Each line must be formatted as 

    x y z Hx Hy Hz    