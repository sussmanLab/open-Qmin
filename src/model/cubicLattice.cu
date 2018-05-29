#include "cubicLattice.cuh"
#include "functions.h"
/*! \file cubicLattice.cu */

/*!
    \addtogroup utilityKernels
    @{
*/
__global__ void gpu_set_random_spins_kernel(dVec *pos, curandState *rngs,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    curandState randState;
    randState = rngs[blockIdx.x];
//    skipahead(threadIdx.x,&randState);
    for (int j =0 ; j < threadIdx.x; ++j)
        curand(&randState);
    for (int dd = 0; dd < DIMENSION; ++dd)
        pos[idx][dd] = curand_normal(&randState);
    scalar lambda = sqrt(dot(pos[idx],pos[idx]));
    pos[idx] = (1/lambda)*pos[idx];
    rngs[blockIdx.x] = randState;
    return;
    };

bool gpu_set_random_spins(dVec *d_pos,
                          curandState *rngs,
                          int blockSize,
                          int nBlocks,
                          int N
                          )
    {
    cout << "calling gpu spin setting" << endl;
    gpu_set_random_spins_kernel<<<nBlocks,blockSize>>>(d_pos,rngs,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
