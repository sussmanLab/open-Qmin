#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
/*! \file landauDeGennesLC.cu */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

__global__ void gpu_qTensor_oneConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                Index3D latticeIndex,
                                scalar a,scalar b,scalar c,scalar l,
                                int N,
                                bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int3 target = latticeIndex.inverseIndex(idx);
    int3 latticeSizes = latticeIndex.getSizes();
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    dVec force(0.0);
    qCurrent = d_spins[idx];
    force -= a*derivativeTrQ2(qCurrent);
    force -= b*derivativeTrQ3(qCurrent);
    force -= c*derivativeTrQ2Squared(qCurrent);

    xDown = d_spins[latticeIndex(wrap(target.x-1,latticeSizes.x),target.y,target.z)];
    xUp = d_spins[latticeIndex(wrap(target.x+1,latticeSizes.x),target.y,target.z)];
    yDown = d_spins[latticeIndex(target.x,wrap(target.y-1,latticeSizes.y),target.z)];
    yUp = d_spins[latticeIndex(target.x,wrap(target.y+1,latticeSizes.y),target.z)];
    zDown = d_spins[latticeIndex(target.x,target.y,wrap(target.z-1,latticeSizes.z))];
    zUp = d_spins[latticeIndex(target.x,target.y,wrap(target.z+1,latticeSizes.z))];
    dVec spatialTerm = l*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
    scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
    spatialTerm[0] += AxxAyy;
    spatialTerm[1] *= 2.0;
    spatialTerm[2] *= 2.0;
    spatialTerm[3] += AxxAyy;
    spatialTerm[4] *= 2.0;
    force -= spatialTerm;

    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

bool gpu_qTensor_oneConstantForce(dVec *d_force,
                                dVec *d_spins,
                                Index3D latticeIndex,
                                scalar A,scalar B,scalar C,scalar L,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    scalar l = 2.0*L;
    gpu_qTensor_oneConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,latticeIndex,
                                                             a,b,c,l,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
