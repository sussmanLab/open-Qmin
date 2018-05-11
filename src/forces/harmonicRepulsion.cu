#include "harmonicRepulsion.cuh"
/*! \file harmonicRepulsion.cu 

\addtogroup forceKernels
@{
*/

/*!
calculate the force per particle given harmonic repulsions
*/
__global__ void gpu_harmonic_repulsion_kernel(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,int *particleType,scalar *d_radii,scalar *d_params,Index2D neighborIndexer,Index2D particleTypeIndexer,int N,bool zeroForce)
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
        scalar K = d_params[particleTypeIndexer(particleType[p2],particleType[idx])];
        scalar sigma0 = d_radii[idx]+d_radii[p2];
        //compute force
        scalar dnorm = norm(relativeDistance);
        if (dnorm <= sigma0)
            d_force[idx] += K*(1.0/sigma0)*(1.0-dnorm/sigma0)*(1.0/dnorm)*relativeDistance;
        };
    };

__global__ void gpu_harmonic_repulsion_monodisperse_kernel(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,Index2D neighborIndexer,int N,bool zeroForce)
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
        dVec relativeDistance = d_neighborVectors[nIdx];
        //compute force
        scalar dnorm = norm(relativeDistance);
        if (dnorm <= 1.0)
            d_force[idx] += 1.0*(1.0-dnorm)*(1.0/dnorm)*relativeDistance;
        };
    };


__global__ void gpu_harmonic_repulsion_allPairs_kernel(dVec *d_force,dVec *d_pos,int *particleType, scalar *d_radii,scalar *d_params,Index2D particleTypeIndexer,periodicBoundaryConditions Box, int N,bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    dVec disp;
    scalar dnorm;
    scalar r1 = d_radii[idx];
    if(zeroForce)
        d_force[idx] = make_dVec(0.0);
    for (int nn = 0; nn < N; ++nn)
        {
        if(nn == idx) continue;
        Box.minDist(d_pos[idx],d_pos[nn],disp);
        dnorm = norm(disp);
        scalar sigma0 = r1 + d_radii[nn];
        if(dnorm < sigma0)
            {
            scalar K = d_params[particleTypeIndexer(particleType[nn],particleType[idx])];
                d_force[idx] += K*(1.0/sigma0)*(1.0-dnorm/sigma0)*(1.0/dnorm)*disp;
            };
        };
    };


/*!
Calculate harmonic repulsion forces, launching one thread per particle
*/
bool gpu_harmonic_repulsion_monodisperse_calculation(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,Index2D neighborIndexer,int N,bool zeroForce)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_harmonic_repulsion_monodisperse_kernel<<<nblocks,block_size>>>(d_force,
           d_neighborsPerParticle,
           d_neighbors,
           d_neighborVectors,
           neighborIndexer,
           N,
           zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/*!
Calculate harmonic repulsion forces, launching one thread per particle
*/
bool gpu_harmonic_repulsion_calculation(dVec *d_force,unsigned int *d_neighborsPerParticle,int *d_neighbors,dVec *d_neighborVectors,int *particleType,scalar *d_radii,scalar *d_params,Index2D neighborIndexer,Index2D particleTypeIndexer,int N,bool zeroForce)
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
           N,
           zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/*!
Calculate harmonic repulsion forces, launching one thread per particle, by brute force
*/
bool gpu_harmonic_repulsion_allPairs(dVec *d_force,dVec *d_pos,int *particleType, scalar *d_radii,scalar *d_params,Index2D particleTypeIndexer,periodicBoundaryConditions &Box, int N, bool zeroForce)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_harmonic_repulsion_allPairs_kernel<<<nblocks,block_size>>>(d_force,
       d_pos, 
       particleType,
       d_radii,
       d_params,
       particleTypeIndexer,
       Box,
       N,
       zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
