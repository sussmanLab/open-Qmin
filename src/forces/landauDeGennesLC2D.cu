#include "landauDeGennesLC2D.cuh"
#include "qTensorFunctions2D.h"
#include "lcForces.h"

/*! \file landauDeGennesLC2D.cu */
/** @addtogroup forceKernels force Kernels
 * @{
 */


__global__ void gpu_2DqTensor_oneConstantForce_kernel(dVec *d_force, dVec *d_spins, int *d_types, int *d_latticeNeighbors,
                                Index2D neighborIndex, scalar a, scalar c,scalar L1, int N, bool zeroOutForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    dVec qCurrent, xDown, xUp, yDown, yUp, xDownyDown, xUpyDown, xDownyUp, xUpyUp;
    int ixd, ixu, iyd, iyu, ixdyd, ixdyu, ixuyd, ixuyu;

    dVec force(0.0);
    
    if(d_types[idx] <= 0) //only compute forces on sites that aren't boundaries
        {
        //phase part
        qCurrent = d_spins[idx];
        force -= a*derivativeTr2DQ2(qCurrent);
        force -= c*derivativeTr2DQ2Squared(qCurrent);
        

        //get neighbor indices and data for nearest neighbors
        ixd = d_latticeNeighbors[neighborIndex(0,idx)];
        ixu = d_latticeNeighbors[neighborIndex(1,idx)];
        iyd = d_latticeNeighbors[neighborIndex(2,idx)];
        iyu = d_latticeNeighbors[neighborIndex(3,idx)];
        xDown = d_spins[ixd]; xUp = d_spins[ixu];
        yDown = d_spins[iyd]; yUp = d_spins[iyu];
        
        //get next nearest-neighbors indices and data
        ixdyd = d_latticeNeighbors[neighborIndex(4,idx)];
        ixdyu = d_latticeNeighbors[neighborIndex(5,idx)];
        ixuyd = d_latticeNeighbors[neighborIndex(6,idx)];
        ixuyu = d_latticeNeighbors[neighborIndex(7,idx)];
        xDownyDown = d_spins[ixdyd]; xDownyUp = d_spins[ixdyu];
        xUpyDown = d_spins[ixuyd]; xUpyUp = d_spins[ixuyu];

        dVec spatialTerm(0.0);
        
        if(d_types[idx] == 0 || d_types[idx] == -2) // bulk part
            {
            spatialTerm = 1.0*laplacianStencil(L1, qCurrent, xDown, xUp, yDown, yUp, xDownyDown, xUpyDown, xDownyUp, xUpyUp);
            }
        //else //near a boundary is less easy... 
            //{
            //This part of the code is yet to be written
            //};
        force += spatialTerm;
        };
    if(zeroOutForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }



bool gpu_2DqTensor_oneConstantForce(dVec *d_force, dVec *d_spins, int *d_types, int *d_latticeNeighbors, Index2D neighborIndex,
                                    scalar A,scalar C,scalar L, int N, bool zeroOutForce, int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size + 1;
    scalar a = 0.5*A;
    scalar c = 0.25*C;
    scalar l = L;
    gpu_2DqTensor_oneConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,d_latticeNeighbors,neighborIndex,
                                                                    a, c, l, N, zeroOutForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }


/** @} */ //end of group declaration
