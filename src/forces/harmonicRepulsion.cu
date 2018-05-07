#include "harmonicRepulsion.cuh"
/*! \file energyMinimizerFIRE.cu 

\addtogroup forceKernels
@{
*/

/*!
calculate the force per particle given harmonic repulsions
*/
__global__ void gpu_harmonic_repulsion_kernel(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,int *particleType,scalar *d_radii,scalar *d_params,Index2D neighborIndexer,Index2D particleTypeIndexer,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int neighs = d_neighborsPerParticle[idx];
    for (int nn = 0; nn < neighs; ++nn)
        {
        int nIdx = neighborIndexer(nn,idx);
        int p2 = d_neighbors[nIdx];
        dVec relativeDistance = d_neighborVectors[nIdx];
        //get parameters
        scalar K = d_params[particleTypeIndexer(particleType[p2],particleType[idx])];
        scalar sigma0 = d_radii[idx]+d_radii[p2];
        //compute force
        scalar dnorm = norm(relativeDistance);
        if (dnorm <= sigma0)
            d_force[idx] += K*(1.0/sigma0)*(1.0-dnorm/sigma0)*(1.0/dnorm)*relativeDistance;
        };
    };

/*!
Calculate harmonic repulsion forces, launching one thread per particle
*/
bool gpu_harmonic_repulsion_calculation(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,int *particleType,scalar *d_radii,scalar *d_params,Index2D neighborIndexer,Index2D particleTypeIndexer,int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_harmonic_repulsion_kernel<<<nblocks,block_size>>>(d_force,
           d_neighborsPerParticle,
           d_neighbors,
           d_neighborVectors,
           particleType,
           d_radii,
           d_params,
           neighborIndexer,
           particleTypeIndexer,
           N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
