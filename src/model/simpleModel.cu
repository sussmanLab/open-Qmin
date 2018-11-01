#include "simpleModel.cuh"
/*! \file simpleModel.cu */

/*!
    \addtogroup modelKernels
    @{
*/

__global__ void gpu_move_particles_kernel(dVec *d_pos,
                      dVec *d_disp,
                      periodicBoundaryConditions Box,
                      scalar scale,
                      int N
                      )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_pos[idx] += scale*d_disp[idx];
    Box.putInBoxReal(d_pos[idx]);
    };


bool gpu_move_particles(dVec *d_pos,
                      dVec *d_disp,
                      periodicBoundaryConditions &Box,
                      scalar scale,
                      int N
                      )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_move_particles_kernel<<<nblocks,block_size>>>(d_pos,d_disp,Box,scale,N);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
