#include "cubicLattice.cuh"
#include "functions.h"
/*! \file cubicLattice.cu */

/*!
    \addtogroup modelKernels
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

__global__ void gpu_copy_boundary_object_kernel(dVec *pos,
                              int *sites,
                              int *neighbors,
                              pair<int,dVec> *assistStructure,
                              int *types,
                              Index2D neighborIndex,
                              int motionDirection,
                              bool resetLattice,
                              int Nsites)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx>=Nsites) return;
    int site = sites[idx];
    int motionSite = neighbors[neighborIndex(motionDirection,site)];
    assistStructure[idx].first = motionSite;
    assistStructure[idx].second = pos[site];
    if(resetLattice)
        types[site] = 0;
    return;
    }

bool gpu_copy_boundary_object(dVec *pos,int *sites,int *neighbors,pair<int,dVec> *assistStructure,
                              int *types,Index2D neighborIndex,int motionDirection,bool resetLattice,int Nsites)
    {
    unsigned int block_size = 512;
    if (Nsites < 512) block_size = 16;
    unsigned int nblocks  = Nsites/block_size + 1;
    gpu_copy_boundary_object_kernel<<<nblocks,block_size>>>(pos,sites,neighbors,assistStructure,types,
                                                            neighborIndex,motionDirection,resetLattice,Nsites);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

__global__ void gpu_move_boundary_object_kernel(dVec *pos,int *sites,pair<int,dVec> *assistStructure,
                              int *types,int newTypeValue,int Nsites)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx>=Nsites) return;

    int site = assistStructure[idx].first;
    sites[idx] =site;
    
    pos[site] =assistStructure[idx].second;
    types[site] = newTypeValue;
    return;
    }

bool gpu_move_boundary_object(dVec *pos,int *sites,pair<int,dVec> *assistStructure,
                              int *types,int newTypeValue,int Nsites)
    {
    unsigned int block_size = 512;
    if (Nsites < 512) block_size = 16;
    unsigned int nblocks  = Nsites/block_size + 1;
    gpu_move_boundary_object_kernel<<<nblocks,block_size>>>(pos,sites,assistStructure,types,
                                                        newTypeValue,Nsites);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
