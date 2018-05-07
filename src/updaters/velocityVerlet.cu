#include "velocityVerlet.cuh"
/*! \file velocityVerlet.cu 

\addtogroup updaterKernels
@{
*/

/*!
update the velocity in a velocity Verlet step
*/
__global__ void gpu_update_velocity_kernel(dVec *d_velocity, dVec *d_force, scalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_velocity[idx] += 0.5*deltaT*d_force[idx];
    };

/*!
calculate the displacement in a velocity verlet step according to the force and velocity
*/
__global__ void gpu_displacement_vv_kernel(dVec *d_displacement, dVec *d_velocity,
                                           dVec *d_force, scalar deltaT, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_displacement[idx] = deltaT*d_velocity[idx]+0.5*deltaT*deltaT*d_force[idx];
    };

/*!
\param d_velocity dVec array of velocity
\param d_force dVec array of force
\param deltaT time step
\param N      the length of the arrays
\post v = v + 0.5*deltaT*force
*/
bool gpu_update_velocity(dVec *d_velocity, dVec *d_force, scalar deltaT, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_update_velocity_kernel<<<nblocks,block_size>>>(
                                                d_velocity,
                                                d_force,
                                                deltaT,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_displacement dVec array of displacements
\param d_velocity dVec array of velocities
\param d_force dVec array of forces
\param Dscalar deltaT the current time step
\param N      the length of the arrays
\post displacement = dt*velocity + 0.5 *dt^2*force
*/
bool gpu_displacement_velocity_verlet(dVec *d_displacement,
                      dVec *d_velocity,
                      dVec *d_force,
                      scalar deltaT,
                      int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_displacement_vv_kernel<<<nblocks,block_size>>>(
                                                d_displacement,d_velocity,d_force,deltaT,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
