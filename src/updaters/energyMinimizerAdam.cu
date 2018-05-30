#include "energyMinimizerAdam.cuh"
/*! \file energyMinimizerAdam.cu 

\addtogroup updaterKernels
@{
*/

__global__ void gpu_adam_step_kernel(dVec *force,
                   dVec *biasedMomentum,
                   dVec *biasedMomentum2,
                   dVec *correctedMomentum,
                   dVec *correctedMomentum2,
                   dVec *displacement,
                   scalar deltaT,
                   scalar beta1,
                   scalar beta2,
                   scalar beta1t,
                   scalar beta2t,
                   int N)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    biasedMomentum[idx] = beta1*biasedMomentum[idx]+(beta1-1)*force[idx];
    biasedMomentum2[idx] = beta2*biasedMomentum2[idx]+(1-beta2)*force[idx]*force[idx];
    correctedMomentum[idx] = biasedMomentum[idx]*(1.0/(1.0-beta1t));
    correctedMomentum2[idx] = biasedMomentum2[idx]*(1.0/(1.0-beta2t));
    scalar rootvc;
    for(int dd = 0; dd < DIMENSION; ++dd)
        {
        rootvc = sqrt(correctedMomentum2[idx][dd]);
        if(rootvc ==0) rootvc = 1e-10;
        displacement[idx][dd] = -deltaT*correctedMomentum[idx][dd]/(rootvc);
        }
    }

bool gpu_adam_step(dVec *force,
                   dVec *biasedMomentum,
                   dVec *biasedMomentum2,
                   dVec *correctedMomentum,
                   dVec *correctedMomentum2,
                   dVec *displacement,
                   scalar deltaT,
                   scalar beta1,
                   scalar beta2,
                   scalar beta1t,
                   scalar beta2t,
                   int N,
                   int blockSize)
    {
    int block_size=blockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_adam_step_kernel<<<nblocks,block_size>>>(force,biasedMomentum,biasedMomentum2,
            correctedMomentum,correctedMomentum2, displacement,
            deltaT,beta1,beta2,beta1t,beta2t,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
