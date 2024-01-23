#include "qTensorLatticeModel2D.h"
#include "../../inc/qTensorFunctions2D.h"

#include "qTensorLatticeModel2D.cuh"

/*! \file qTensorLatticeModel2D.cu */

/*!
    \addtogroup modelKernels
    @{
*/

__global__ void gpu_update_2DqTensor_simple_kernel(dVec *d_disp, dVec *d_pos, int N)
    { 
    //sums displacement to position on the gpu
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pidx = idx/DIMENSION;  //row index
    if(pidx>=N) return;
    int didx = idx%DIMENSION;  //column index

    d_pos[pidx][didx] += d_disp[pidx][didx];   
    return;
    };


__global__ void gpu_update_2DqTensor_simple_kernel(dVec *d_disp, dVec *d_pos, int N, scalar scale)
    {
    //sums scaled displacemment to position on the gpu
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pidx = idx/DIMENSION; 
    if(pidx>=N) return;
    int didx = idx%DIMENSION;

    d_pos[pidx][didx] += scale*d_disp[pidx][didx];
    return;
    };


__global__ void gpu_2DqTensor_largestEigenvalue_kernel(dVec *Q, scalar *defects, int *t, int N)
    {
    //calls the function that finds the largest eigenvalue for a qtensor for gpu
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;
    if(t[idx] > 0)
        return;

    scalar a, b;
    eigenvaluesOfQ2D(Q[idx], a, b);
    defects[idx] = max(a,b);
    return;
    };


__global__ void gpu_2DqTensor_computeDeterminant_kernel(dVec *Q, scalar *defects, int *t, int N)
    {
    //calls the function that finds the determinant of a qtensor for gpu
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;
    if(t[idx] > 0)
        return;
    defects[idx] = determinantOf2DQ(Q[idx]);
    return;
    };


__global__ void gpu_set_random_nematic_2DqTensors_kernel(dVec *pos, int *type, curandState *rngs, scalar amplitude,
                                                            bool globallyAligned, scalar globalPhi, int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;
    curandState randState;
    randState = rngs[idx];

    scalar phi = 2.0*PI*curand_uniform(&randState);
    if(globallyAligned)
        {
        phi = globalPhi;    
        }

    scalar2 n;
    n.x = cos(phi);
    n.y = sin(phi);

    if(type[idx] <= 0)
        qTensorFromDirector2D(n, amplitude, pos[idx]);
    rngs[idx] = randState;
    return;
    };



bool gpu_update_2DqTensor(dVec *d_disp, dVec *Q, scalar scale, int N, int blockSize)
    {
    if (N < 128) blockSize = 16;
    unsigned int nBlocks  = DIMENSION*N/blockSize + 1;
    gpu_update_2DqTensor_simple_kernel<<<nBlocks,blockSize>>>(d_disp, Q,scale,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


bool gpu_update_2DqTensor(dVec *d_disp, dVec *Q, int N,int blockSize)
    {
    if (N < 128) blockSize = 16;
    unsigned int nBlocks  = DIMENSION*N/blockSize + 1;
    gpu_update_2DqTensor_simple_kernel<<<nBlocks,blockSize>>>(d_disp, Q,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };


bool gpu_set_random_nematic_2DqTensors(dVec *d_pos, int *d_types, curandState *rngs, scalar amplitude, int blockSize,
                                            int nBlocks, bool globallyAligned, scalar phi, int N)
    {
    if(DIMENSION < 2)
        {
        printf("\nAttempting to initialize Q- tensors with incorrectly set dimension...change the root CMakeLists.txt file to have dimension 2 and recompile\n");
        throw std::exception();
        }
    
    gpu_set_random_nematic_2DqTensors_kernel<<<nBlocks, blockSize>>>(d_pos, d_types, rngs, amplitude, globallyAligned, phi, N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };



bool gpu_get_2DqTensor_DefectMeasures(dVec *Q, scalar *defects, int *t, int defectType, int N)
    {
    unsigned int block_size = 128;
    if(N < 128) block_size = 16;
    unsigned int nblocks = N/block_size + 1;

    if(defectType == 0) gpu_2DqTensor_largestEigenvalue_kernel<<<nblocks, block_size>>>(Q, defects, t, N);
    if(defectType == 1) gpu_2DqTensor_computeDeterminant_kernel<<<nblocks, block_size>>>(Q, defects, t, N);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
