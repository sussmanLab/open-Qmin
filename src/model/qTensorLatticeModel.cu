#include "qTensorLatticeModel.cuh"
#include "../../inc/qTensorFunctions.h"
//#incldue "qTensorfunctions.h"
/*! \file qTensorLatticeModel.cu */

/*!
    \addtogroup modelKernels
    @{
*/

__global__ void gpu_largestEigenvalue_kernel(dVec *Q,scalar *defects,int *t, int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if(t[idx]>0)
        return;
    scalar a,b,c;
    eigenvaluesOfQ(Q[idx],a,b,c);
    defects[idx] = max(max(a,b),c);
    return;
    }


__global__ void gpu_computeDeterminant_kernel(dVec *Q,scalar *defects,int *t, int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if(t[idx]>0)
        return;
    defects[idx] = determinantOfQ(Q[idx]);
    return;
    }

__global__ void gpu_degenerateEigenvalue_kernel(dVec *Q,scalar *defects,int *t, int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if(t[idx]>0)
        return;
    scalar trQ2 = TrQ2(Q[idx]);
    scalar det = determinantOfQ(Q[idx]);
    defects[idx] = trQ2*trQ2*trQ2 - 54.0*det*det;
    return;
    }

__global__ void gpu_set_random_nematic_qTensors_kernel(dVec *pos, int *type, curandState *rngs,scalar amplitude, bool globallyAligned, scalar globalTheta, scalar globalPhi,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    curandState randState;
    randState = rngs[idx];

    scalar theta = acos(2.0*curand_uniform(&randState)-1);
    scalar phi = 2.0*PI*curand_uniform(&randState);
    if(globallyAligned)
        {
        theta = globalTheta;
        phi = globalPhi;
        }
    scalar3 n;
    n.x = cos(phi)*sin(theta);
    n.y = sin(phi)*sin(theta);
    n.z = cos(theta);

    if(type[idx] <=0)
        qTensorFromDirector(n, amplitude, pos[idx]);
    rngs[idx] = randState;
    return;
    };

__global__ void gpu_update_qTensor_kernel(dVec *d_disp,
                            dVec *d_pos,
                            int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_pos[idx] += d_disp[idx];
    return;
    };

bool gpu_update_qTensor(dVec *d_disp,
                            dVec *Q,
                            int N)
    {
    unsigned int blockSize = 1024;
    if (N < 128) blockSize = 16;
    unsigned int nBlocks  = N/blockSize + 1;
    if(DIMENSION <5)
        {
        printf("\nAttempting to initialize Q-tensors with incorrectly set dimension...change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    gpu_update_qTensor_kernel<<<nBlocks,blockSize>>>(d_disp, Q,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_set_random_nematic_qTensors(dVec *d_pos,
                          int *d_types,
                          curandState *rngs,
                          scalar amplitude,
                          int blockSize,
                          int nBlocks,
                          bool globallyAligned,
                          scalar theta,
                          scalar phi,
                          int N
                          )
    {
    if(DIMENSION <5)
        {
        printf("\nAttempting to initialize Q-tensors with incorrectly set dimension...change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    gpu_set_random_nematic_qTensors_kernel<<<nBlocks,blockSize>>>(d_pos,d_types, rngs,amplitude, globallyAligned, theta, phi,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }


bool gpu_get_qtensor_DefectMeasures(dVec *Q,
                                    scalar *defects,
                                    int *t,
                                    int defectType,
                                    int N)
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    if(defectType ==0)
        gpu_largestEigenvalue_kernel<<<nblocks,block_size>>>(Q,defects,t,N);
    if(defectType ==1)
        gpu_computeDeterminant_kernel<<<nblocks,block_size>>>(Q,defects,t,N);
    if(defectType ==2)
        gpu_degenerateEigenvalue_kernel<<<nblocks,block_size>>>(Q,defects,t,N);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
