#include "energyMinimizerFIRE.cuh"
/*! \file energyMinimizerFIRE.cu 

\addtogroup updaterKernels
@{
*/

/*!
update the velocity according to a FIRE step
*/
__global__ void gpu_update_velocity_FIRE_kernel(dVec *d_velocity, dVec *d_force, scalar alpha, scalar scaling, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx] = (1-alpha)*d_velocity[idx] + alpha*scaling*d_force[idx];
    };

/*!
\param d_velocity array of velocity
\param d_force array of force
\param alpha the FIRE parameter
\param scaling the square root of (v.v / f.f)
\param N      the length of the arrays
\post v = (1-alpha)v + alpha*scalaing*force
*/
bool gpu_update_velocity_FIRE(dVec *d_velocity, dVec *d_force, scalar alpha, scalar scaling, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_FIRE_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                alpha,
                                                scaling,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
