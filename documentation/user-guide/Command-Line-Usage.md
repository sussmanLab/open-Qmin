# Command-line usage

## To use the GUI
`build/openQminGUI.out`
    
## Running on the command line

Run `build/openQmin.out` with any of the available command-line options, which are described [here](Command-Line-Options) in detail; for a brief summary you can type `build/openQmin.out --help`.
    
Some examples:
* To run 100 FIRE minimization steps on a cubic lattice of side length 250:  
`build/openQmin.out -i 100 -l 250`
* To do the same thing but using a GPU in slot 0:  
`build/openQmin.out -i 100 -l 250 -g 0`
* To load a file, e.g. "assets/boundaryInput.txt"  with custom boundaries prepared for a cubic lattice of side length 80  
`build/openQmin.out -i 100 -l 80 --boundaryFile assets/boundaryInput.txt`
* To do the above, but also save the post-minimization state:  
`./build/openQmin.out -i 80 --boundaryFile assets/boundaryInput.txt -l 80 --saveFile data/saveTesting`

:::{note}
For `--boundaryFile` and `--saveFile`, the file path is relative to your present working directory.
:::

### Specifying MPI jobs

As noted above, the `-l` command line flag can be used to specify the side length of a cubic simulation
domain (other flags exist if you want the domain to be a rectangular prism). **When using MPI, the default behavior is to take the command line size to specify the domain size ON EVERY MPI RANK.** (One could imagine an alternate default behavior in which the command line specified the TOTAL size of the
simulation domain, which was then divided among the different processors.)

So, for example, the command:  
`mpirun -n 1 build/openQmin.out -l 100`  
will run a simulation domain of total size (100x100x100) lattice sites, on a single rank.

Moving to two processors, the command:  
`mpirun -n 2 build/openQmin.out -l 100`  
will run a simulation domain of total size (200x100x100) lattice sites, on two ranks (that
is, each rank will control a 100x100x100 domain). 

Similarly the command:  
`mpirun -n 8 build/openQmin.out -l 100`  
will run a simulation domain of total size (200x200x200) lattice sites, where each of the
eight ranks continues to control a block of 100x100x100 lattice sites.