# Saved output

Both the command-line and GUI executables can save the current configuration of the simulation, and simple visualization tools are available in the /visualizationTools directory to see the output of these systems. 

The output format is a simple txt file, where each line is:

    x y z Qxx Qxy Qxz Qyy Qyz t d
    
where 
* `x y z` are the lattice site (in the case of MPI simulations, each rank will produce a separate 
file with `x y z` in absolute coordinates), 
* The third to eighth columns are the current independent values of the Q-tensor. 
* The integer `t` specifies the "type" of the lattice site (bulk, boundary, part of an object),
* and `d` is the nematic degree of order (a defect measure) that is computed for all lattice sites not part of an object. By default this will be the largest eigenvalue of the Q-tensor at that site; for lattice sites that are part of an object this will always be zero.