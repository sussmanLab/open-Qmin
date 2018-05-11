#include "lennardJones6_12.cuh"
/*! \file lennardJones6_12.cu
\addtogroup forceKernels
@{
*/

__global__ void gpu_lennardJones6_12_energy_kernel(scalar *d_energy,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rc,
                                   int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int neighs = d_neighborsPerParticle[idx];
    scalar energy = 0;
    for (int nn = 0; nn < neighs; ++nn)
        {
        int nIdx = neighborIndexer(nn,idx);
        int p2 = d_neighbors[nIdx];
        dVec relativeDistance = d_neighborVectors[nIdx];
        //get parameters
        int particlePairType = particleTypeIndexer(particleType[p2],particleType[idx]);
        scalar epsilon = d_epsilon[particlePairType];
        scalar sigma = d_sigma[particlePairType];
        scalar dnorm = norm(relativeDistance);
        if (dnorm <= rc)
            {
            scalar rinv = sigma/dnorm;
            scalar rinv3 = rinv*rinv*rinv;
            scalar rinv6 = rinv3*rinv3;
            scalar rinv12=rinv6*rinv6;
            //avoid double-counting...each particle gets half of the energy of the interaction
            energy += 0.5*4.0*epsilon*(rinv12 - rinv6);
            };
        }
    d_energy[idx] = energy;
    }

__global__ void gpu_lennardJones6_12_calculation_kernel(dVec *d_force,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rc,
                                   int N,
                                   bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int neighs = d_neighborsPerParticle[idx];
    if(zeroForce)
        d_force[idx] = make_dVec(0.0);
    for (int nn = 0; nn < neighs; ++nn)
        {
        int nIdx = neighborIndexer(nn,idx);
        int p2 = d_neighbors[nIdx];
        dVec relativeDistance = d_neighborVectors[nIdx];
        //get parameters
        int particlePairType = particleTypeIndexer(particleType[p2],particleType[idx]);
        scalar epsilon = d_epsilon[particlePairType];
        scalar sigma = d_sigma[particlePairType];
        scalar dnorm = norm(relativeDistance);
        if (dnorm <= rc)
            {
            scalar rinv = sigma/dnorm;
            scalar rinv3 = rinv*rinv*rinv;
            scalar rinv6 = rinv3*rinv3;
            scalar rinv12=rinv6*rinv6;
            scalar negativedUdr = 4.0*epsilon*(12.0*rinv12/dnorm - 6.0*rinv6/dnorm);
            d_force[idx] += (negativedUdr/dnorm)*relativeDistance;
            };
        }
    }

bool gpu_lennardJones6_12_energy(scalar *d_energy,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rCut,
                                   int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_lennardJones6_12_energy_kernel<<<nblocks,block_size>>>(
                                   d_energy,
                                   d_neighborsPerParticle,
                                   d_neighbors,
                                   d_neighborVectors,
                                   particleType,
                                   d_epsilon,
                                   d_sigma,
                                   neighborIndexer,
                                   particleTypeIndexer,
                                   rCut,
                                   N);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
bool gpu_lennardJones6_12_calculation(dVec *d_force,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rCut,
                                   int N,
                                   bool zeroForce)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;

    gpu_lennardJones6_12_calculation_kernel<<<nblocks,block_size>>>(
                                   d_force,
                                   d_neighborsPerParticle,
                                   d_neighbors,
                                   d_neighborVectors,
                                   particleType,
                                   d_epsilon,
                                   d_sigma,
                                   neighborIndexer,
                                   particleTypeIndexer,
                                   rCut,
                                   N,
                                   zeroForce);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
