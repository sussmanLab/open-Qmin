#include "energyMinimizerNesterovAG.cuh"
/*! \file energyMinimizerNesterovAG.cu

\addtogroup updaterKernels
@{
*/

__global__ void gpu_nesterovAG_step_kernel(dVec *force,
                   dVec *position,
                   dVec *alternatePosition,
                   scalar deltaT,
                   scalar mu,
                   int N)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pidx = idx/DIMENSION;
    if(pidx>=N) return;
    int didx = idx%DIMENSION;

    scalar f = force[pidx][didx];
    scalar oldAltPos = alternatePosition[pidx][didx];

    alternatePosition[pidx][didx] = position[pidx][didx] + deltaT*force[pidx][didx];
    position[pidx][didx] = alternatePosition[pidx][didx] + mu*(alternatePosition[pidx][didx] - oldAltPos);
    }

/*!
A memory-efficiency optimization has each thread acting on one dimension of one degree of freedom...
  */
bool gpu_nesterovAG_step(dVec *force,
                   dVec *position,
                   dVec *alternatePosition,
                   scalar deltaT,
                   scalar mu,
                   int N,
                   int blockSize)
    {
    int block_size=blockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = DIMENSION*N/block_size + 1;
    gpu_nesterovAG_step_kernel<<<nblocks,block_size>>>(force,position,alternatePosition,
            deltaT,mu,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
