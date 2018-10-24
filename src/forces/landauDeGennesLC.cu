#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
/*! \file landauDeGennesLC.cu */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

__device__ void gpu_get_six_neighbors(int3 &target, int &ixd, int &ixu,int &iyd, int &iyu,int &izd, int &izu,
                                      Index3D &latticeIndex, int3 &latticeSizes)
    {
    ixd = latticeIndex(wrap(target.x-1,latticeSizes.x),target.y,target.z);
    ixu = latticeIndex(wrap(target.x+1,latticeSizes.x),target.y,target.z);
    iyd = latticeIndex(target.x,wrap(target.y-1,latticeSizes.y),target.z);
    iyu = latticeIndex(target.x,wrap(target.y+1,latticeSizes.y),target.z);
    izd = latticeIndex(target.x,target.y,wrap(target.z-1,latticeSizes.z));
    izu = latticeIndex(target.x,target.y,wrap(target.z+1,latticeSizes.z));
    };

__global__ void gpu_qTensor_computeBoundaryForcesGPU_kernel(dVec *d_force,
                                 dVec *d_spins,
                                 int *d_types,
                                 boundaryObject *d_bounds,
                                 Index3D latticeIndex,
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
    dVec tempForce(0.0);
    if(d_types[idx] < 0) //compute only for sites adjacent to boundaries
        {
        qCurrent = d_spins[idx];
        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);

        if(d_types[ixd] > 0)
            {
            xDown = d_spins[ixd];
            computeBoundaryForce(qCurrent, xDown, d_bounds[d_types[ixd]-1],tempForce);
            force = force + tempForce;
            }
        if(d_types[ixu] > 0)
            {
            xUp = d_spins[ixu];
            computeBoundaryForce(qCurrent, xUp, d_bounds[d_types[ixu]-1],tempForce);
            force = force +tempForce;
            };
        if(d_types[iyd] > 0)
            {
            yDown = d_spins[iyd];
            computeBoundaryForce(qCurrent, yDown, d_bounds[d_types[iyd]-1],tempForce);
            force = force +tempForce;
            };
        if(d_types[iyu] > 0)
            {
            yUp = d_spins[iyu];
            computeBoundaryForce(qCurrent, yUp, d_bounds[d_types[iyu]-1],tempForce);
            force = force +tempForce;
            };
        if(d_types[izd] > 0)
            {
            zDown = d_spins[izd];
            computeBoundaryForce(qCurrent, zDown, d_bounds[d_types[izd]-1],tempForce);
            force = force +tempForce;
            };
        if(d_types[izu] > 0)
            {
            zUp = d_spins[izu];
            computeBoundaryForce(qCurrent, zUp, d_bounds[d_types[izu]-1],tempForce);
            force = force +tempForce;
            };
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

__global__ void gpu_qTensor_firstDerivatives_kernel(cubicLatticeDerivativeVector *d_derivatives,
                                dVec *d_spins,
                                int *d_types,
                                Index3D latticeIndex,
                                int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int3 target = latticeIndex.inverseIndex(idx);
    int3 latticeSizes = latticeIndex.getSizes();
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    cubicLatticeDerivativeVector zero(0.0);
    d_derivatives[idx] = zero;

    if(d_types[idx] <= 0) //no force on sites that are part of boundaries
        {
        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);
        xDown = d_spins[ixd];
        xUp = d_spins[ixu];
        yDown = d_spins[iyd];
        yUp = d_spins[iyu];
        zDown = d_spins[izd];
        zUp = d_spins[izu];
        if(d_types[idx] == 0) // bulk is easy
            {
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                };
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                };
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                };
            }
        else //near a boundary is less easy
            {
            if(d_types[ixd] <=0 ||d_types[ixu] <= 0) //x bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                    };
                }
            else if (d_types[ixd] <=0 ||d_types[ixu] > 0) //right is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][qq] = (qCurrent[qq]-xDown[qq]);
                    };
                }
            else//left is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][qq] = (qCurrent[qq]-xUp[qq]);
                    };
                };
            if(d_types[iyd] <=0 ||d_types[iyu] <= 0) //y bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                    };
                }
            else if (d_types[iyd] <=0 ||d_types[iyu] > 0) //up is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][DIMENSION+qq] = (qCurrent[qq]-yDown[qq]);
                    };
                }
            else//down is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][DIMENSION+qq] = (qCurrent[qq]-yUp[qq]);
                    };
                };
            if(d_types[izd] <=0 ||d_types[izu] <= 0) //z bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                    };
                }
            else if (d_types[izd] <=0 ||d_types[izu] > 0) //up is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][2*DIMENSION+qq] = (qCurrent[qq]-zDown[qq]);
                    };
                }
            else//down is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][2*DIMENSION+qq] = (qCurrent[qq]-zUp[qq]);
                    };
                };
            };
        };
    }

__global__ void gpu_qTensor_oneConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
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

    if(d_types[idx] <= 0) //no force on sites that are part of boundaries
        {
        //phase part is simple
        qCurrent = d_spins[idx];
        force -= a*derivativeTrQ2(qCurrent);
        force -= b*derivativeTrQ3(qCurrent);
        force -= c*derivativeTrQ2Squared(qCurrent);

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);
        xDown = d_spins[ixd];
        xUp = d_spins[ixu];
        yDown = d_spins[iyd];
        yUp = d_spins[iyu];
        zDown = d_spins[izd];
        zUp = d_spins[izu];
        dVec spatialTerm(0.0);
        if(d_types[idx] == 0) // bulk is easy
            {
            spatialTerm = l*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
            scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += AxxAyy;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += AxxAyy;
            spatialTerm[4] *= 2.0;
            }
        else //near a boundary is less easy
            {
            if(d_types[ixd] <=0)
                spatialTerm += qCurrent - xDown;
            if(d_types[ixu] <=0)
                spatialTerm += qCurrent - xUp;
            if(d_types[iyd] <=0)
                spatialTerm += qCurrent - yDown;
            if(d_types[iyu] <=0)
                spatialTerm += qCurrent - yUp;
            if(d_types[izd] <=0)
                spatialTerm += qCurrent - zDown;
            if(d_types[izu] <=0)
                spatialTerm += qCurrent - zUp;
            scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += AxxAyy;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += AxxAyy;
            spatialTerm[4] *= 2.0;
            spatialTerm = l*spatialTerm;
            };
        force -= spatialTerm;
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

bool gpu_qTensor_computeBoundaryForcesGPU(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                boundaryObject *d_bounds,
                                Index3D latticeIndex,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_computeBoundaryForcesGPU_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,d_bounds,latticeIndex,
                                                             N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_qTensor_firstDerivatives(cubicLatticeDerivativeVector *d_derivatives,
                          dVec *d_spins,
                          int *d_types,
                          Index3D latticeIndex,
                          int N,
                          int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_firstDerivatives_kernel<<<nblocks,block_size>>>(d_derivatives,d_spins,d_types,latticeIndex,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_oneConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
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
    gpu_qTensor_oneConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,latticeIndex,
                                                             a,b,c,l,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
