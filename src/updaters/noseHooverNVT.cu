#include "noseHooverNVT.cuh"
/*! \file noseHooverNVT.cu
\addtogroup updaterKernels
@{
*/

__global__ void gpu_propagate_noseHoover_chain_kernel(scalar *kes,
                                    scalar4 *bath,
                                    scalar deltaT,
                                    scalar temperature,
                                    int Nchain,
                                    int Ndof)
    {
    scalar dt8 = 0.125*deltaT;
    scalar dt4 = 0.25*deltaT;
    scalar dt2 = 0.5*deltaT;
    //first quarter time step
    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    for (int ii = Nchain-1; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        bath[ii].z = (bath[ii-1].w*bath[ii-1].y*bath[ii-1].y-temperature)/bath[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        scalar ef = exp(-dt8*bath[ii+1].y);
        bath[ii].y *= ef;
        bath[ii].y += bath[ii].z*dt4;
        bath[ii].y *= ef;
        };

    bath[0].z = (2.0*kes[0] - DIMENSION*(Ndof-DIMENSION)*temperature)/bath[0].w;
    scalar ef = exp(-dt8*bath[1].y);
    bath[0].y *= ef;
    bath[0].y += bath[0].z*dt4;
    bath[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        bath[ii].x += dt2*bath[ii].y;

    //get the factor that will scale particle velocities...
    kes[1] = exp(-dt2*bath[0].y);
    //...and pre-emptively update the kinetic energy
    kes[0] = kes[1]*kes[1]*kes[0];

    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    bath[0].z = (2.0*kes[0] - DIMENSION*(Ndof-DIMENSION)*temperature)/bath[0].w;
    ef = exp(-dt8*bath[1].y);
    bath[0].y *= ef;
    bath[0].y += bath[0].z*dt4;
    bath[0].y *= ef;
    for (int ii = 1; ii < Nchain; ++ii)
        {
        bath[ii].z = (bath[ii-1].w*bath[ii-1].y*bath[ii-1].y-temperature)/bath[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        scalar ef = exp(-dt8*bath[ii+1].y);
        bath[ii].y *= ef;
        bath[ii].y += bath[ii].z*dt4;
        bath[ii].y *= ef;
        };
    };


bool gpu_propagate_noseHoover_chain(scalar *d_kes,
                                    scalar4 *d_bath,
                                    scalar deltaT,
                                    scalar temperature,
                                    int Nchain,
                                    int Ndof)
    {
        gpu_propagate_noseHoover_chain_kernel<<<1,1>>>(d_kes,d_bath,deltaT,temperature,Nchain,Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/** @} */ //end of group declaration
