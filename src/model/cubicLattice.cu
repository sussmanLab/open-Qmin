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
    if(type[idx] <=0)
        {
        pos[idx][0] = amplitude*(sin(theta)*sin(theta)*cos(phi)*cos(phi)-1.0/3.0);
        pos[idx][1] = amplitude*sin(theta)*sin(theta)*cos(phi)*sin(phi);
        pos[idx][2] = amplitude*sin(theta)*cos(theta)*cos(phi);
        pos[idx][3] = amplitude*(sin(theta)*sin(theta)*sin(phi)*sin(phi)-1.0/3.0);
        pos[idx][4] = amplitude*sin(theta)*cos(theta)*sin(phi);
        }
    rngs[idx] = randState;
    return;
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

__global__ void gpu_update_spins_kernel(dVec *d_disp,
                      dVec *d_pos,
                      scalar scale,
                      int N,
                      bool normalize)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    d_pos[idx] += scale*d_disp[idx];
    if(normalize)
        {
        scalar nrm =norm(d_pos[idx]);
        d_pos[idx] = (1.0/nrm)*d_pos[idx];
        }
    }

__global__ void gpu_update_spins_simple_kernel(dVec *d_disp,
                      dVec *d_pos,
                      int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pidx = idx/DIMENSION;
    if(pidx>=N) return;
    int didx = idx%DIMENSION;

    d_pos[pidx][didx] += d_disp[pidx][didx];
    }

bool gpu_update_spins(dVec *d_disp,
                      dVec *d_pos,
                      scalar scale,
                      int N,
                      bool normalize)
    {
    unsigned int block_size = 1024;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    if(!normalize && scale == 1.)
        {
        nblocks = DIMENSION*N/block_size + 1;
        gpu_update_spins_simple_kernel<<<nblocks,block_size>>>(d_disp,d_pos,N);
        }
    else
        gpu_update_spins_kernel<<<nblocks,block_size>>>(d_disp,d_pos,scale,N,normalize);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
