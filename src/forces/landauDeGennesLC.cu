#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
#include "lcForces.h"
/*! \file landauDeGennesLC.cu */
/** @addtogroup forceKernels force Kernels
 * @{
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

__device__ void gpu_phase_force(dVec &qCurrent, scalar &a, scalar &b, scalar &c, dVec &force)
    {
    force -= a*derivativeTrQ2(qCurrent);
    force -= b*derivativeTrQ3(qCurrent);
    force -= c*derivativeTrQ2Squared(qCurrent);
    //force += allPhaseComponentForces(qCurrent,a,b,c);
    }

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
                                int *latticeNeighbors,
                                Index2D neighborIndex,
                                int N)
    {
    unsigned int currentIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (currentIndex >= N)
        return;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    cubicLatticeDerivativeVector zero(0.0);
    d_derivatives[currentIndex] = zero;

    if(d_types[currentIndex] <= 0) //no force on sites that are part of boundaries
        {
        //get neighbor indices and data
        int ixd =latticeNeighbors[neighborIndex(0,currentIndex)];
        int ixu =latticeNeighbors[neighborIndex(1,currentIndex)];
        int iyd =latticeNeighbors[neighborIndex(2,currentIndex)];
        int iyu =latticeNeighbors[neighborIndex(3,currentIndex)];
        int izd =latticeNeighbors[neighborIndex(4,currentIndex)];
        int izu =latticeNeighbors[neighborIndex(5,currentIndex)];
        xDown = d_spins[ixd];
        xUp = d_spins[ixu];
        yDown = d_spins[iyd];
        yUp = d_spins[iyu];
        zDown = d_spins[izd];
        zUp = d_spins[izu];
        qCurrent = d_spins[currentIndex];
        if(d_types[currentIndex] == 0) // bulk is easy
            {
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[currentIndex][qq] = 0.5*(xUp[qq]-xDown[qq]);
                };
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[currentIndex][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                };
            for (int qq = 0; qq < DIMENSION; ++qq)
                {
                d_derivatives[currentIndex][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                };
            }
        else //near a boundary is less easy
            {
            if(d_types[ixd] <=0 &&d_types[ixu] <= 0) //x bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][qq] = 0.5*(xUp[qq]-xDown[qq]);
                    };
                }
            else if (d_types[ixu] > 0) //right is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][qq] = (qCurrent[qq]-xDown[qq]);
                    };
                }
            else//left is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][qq] = (xUp[qq]-qCurrent[qq]);
                    };
                };
            if(d_types[iyd] <=0 && d_types[iyu] <= 0) //y bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                    };
                }
            else if (d_types[iyu] > 0) //up is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][DIMENSION+qq] = (qCurrent[qq]-yDown[qq]);
                    };
                }
            else//down is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][DIMENSION+qq] = (yUp[qq]-qCurrent[qq]);
                    };
                };
            if(d_types[izd] <=0 && d_types[izu] <= 0) //z bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                    };
                }
            else if (d_types[izu] > 0) //up is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][2*DIMENSION+qq] = (qCurrent[qq]-zDown[qq]);
                    };
                }
            else//down is boundary
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[currentIndex][2*DIMENSION+qq] = (zUp[qq]-qCurrent[qq]);
                    };
                };
            };
        };
    }

__global__ void gpu_qTensor_computeObjectForceFromStresses_kernel(int *sites,
                                        int *latticeTypes,
                                        int *latticeNeighbors,
                                        Matrix3x3 *stress,
                                        scalar3 *objectForces,
                                        Index2D neighborIndex,
                                        int nSites)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nSites)
        return;
    int currentIndex = sites[idx];
    int ixd =latticeNeighbors[neighborIndex(0,currentIndex)];
    int ixu =latticeNeighbors[neighborIndex(1,currentIndex)];
    int iyd =latticeNeighbors[neighborIndex(2,currentIndex)];
    int iyu =latticeNeighbors[neighborIndex(3,currentIndex)];
    int izd =latticeNeighbors[neighborIndex(4,currentIndex)];
    int izu =latticeNeighbors[neighborIndex(5,currentIndex)];
    scalar3 surfaceArea = make_scalar3(0,0,0);
    if(latticeTypes[ixd] >0)
        surfaceArea.x = -1.0;
    if(latticeTypes[ixu] >0)
        surfaceArea.x = 1.0;
    if(latticeTypes[iyd] >0)
        surfaceArea.y = -1.0;
    if(latticeTypes[iyu] >0)
        surfaceArea.y = 1.0;
    if(latticeTypes[izd] >0)
        surfaceArea.z = -1.0;
    if(latticeTypes[izu] >0)
        surfaceArea.z = 1.0;
    objectForces[idx] = surfaceArea*stress[idx];
    }

__global__ void gpu_qTensor_oneConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                int *d_latticeNeighbors,
                                Index2D neighborIndex,
                                scalar a,scalar b,scalar c,scalar L1,
                                int N,
                                bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    dVec force(0.0);

    if(d_types[idx] <= 0) //only compute forces on sites that aren't boundaries
        {
        //phase part is simple
        qCurrent = d_spins[idx];
        gpu_phase_force(qCurrent, a, b, c, force);

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        ixd =d_latticeNeighbors[neighborIndex(0,idx)];
        ixu =d_latticeNeighbors[neighborIndex(1,idx)];
        iyd =d_latticeNeighbors[neighborIndex(2,idx)];
        iyu =d_latticeNeighbors[neighborIndex(3,idx)];
        izd =d_latticeNeighbors[neighborIndex(4,idx)];
        izu =d_latticeNeighbors[neighborIndex(5,idx)];

        xDown = d_spins[ixd]; xUp = d_spins[ixu];
        yDown = d_spins[iyd]; yUp = d_spins[iyu];
        zDown = d_spins[izd]; zUp = d_spins[izu];
        dVec spatialTerm(0.0);
        if(d_types[idx] == 0) // bulk is easy
            {
            spatialTerm = L1*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
            scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += AxxAyy;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += AxxAyy;
            spatialTerm[4] *= 2.0;
            }
        else //near a boundary is less easy... ternary operators are slightly better than many ifs (particularly if boundaries are typically jagged)
            {
            if(d_types[ixd] >0)//xDown is a boundary
                spatialTerm -= (xUp-qCurrent);
            if(d_types[ixu] >0)//xUp is a boundary
                spatialTerm -= (xDown-qCurrent);//negative derivative and negative nu_x cancel
            if(d_types[iyd] >0)//ydown
                spatialTerm -= (yUp-qCurrent);
            if(d_types[iyu] >0)
                spatialTerm -= (yDown-qCurrent);//negative derivative and negative nu_y cancel
            if(d_types[izd] >0)//zDown is boundary
                spatialTerm -= (zUp-qCurrent);
            if(d_types[izu] >0)
                spatialTerm -= (zDown-qCurrent);//negative derivative and negative nu_z cancel
            spatialTerm = spatialTerm*L1;
            scalar crossTerm = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += crossTerm;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += crossTerm;
            spatialTerm[4] *= 2.0;
            };
        force -= spatialTerm;
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

__global__ void gpu_qTensor_multiConstantForce_kernel(dVec *d_force,
                                    dVec *d_spins,
                                    int *d_types,
                                    cubicLatticeDerivativeVector *d_derivatives,
                                    int *d_latticeNeighbors,
                                    Index2D neighborIndex,
                                    scalar a,scalar b,scalar c,
                                    scalar L1,scalar L2, scalar L3,
                                    scalar L4, scalar L6,
                                    int N,
                                    bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    dVec force(0.0);

    if(d_types[idx] <= 0) //only compute forces on sites that aren't boundaries
        {
        //phase part is simple
        qCurrent = d_spins[idx];
        gpu_phase_force(qCurrent, a, b, c, force);

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        ixd =d_latticeNeighbors[neighborIndex(0,idx)];
        ixu =d_latticeNeighbors[neighborIndex(1,idx)];
        iyd =d_latticeNeighbors[neighborIndex(2,idx)];
        iyu =d_latticeNeighbors[neighborIndex(3,idx)];
        izd =d_latticeNeighbors[neighborIndex(4,idx)];
        izu =d_latticeNeighbors[neighborIndex(5,idx)];

        xDown = d_spins[ixd]; xUp = d_spins[ixu];
        yDown = d_spins[iyd]; yUp = d_spins[iyu];
        zDown = d_spins[izd]; zUp = d_spins[izu];
        cubicLatticeDerivativeVector xDownDerivative = d_derivatives[ixd];
        cubicLatticeDerivativeVector xUpDerivative = d_derivatives[ixu];
        cubicLatticeDerivativeVector yDownDerivative = d_derivatives[iyd];
        cubicLatticeDerivativeVector yUpDerivative = d_derivatives[iyu];
        cubicLatticeDerivativeVector zDownDerivative = d_derivatives[izd];
        cubicLatticeDerivativeVector zUpDerivative = d_derivatives[izu];

        dVec spatialTerm(0.0);
        dVec individualTerm(0.0);
        if(d_types[idx] == 0 || d_types[idx] == -2) // bulk is easy
            {
            lcForce::bulkL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,individualTerm);
            spatialTerm += individualTerm;
            if(L2 != 0)
                {
                lcForce::bulkL2Force(L2,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L3 != 0)
                {
                lcForce::bulkL3Force(L3,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L4 != 0)
                {
                lcForce::bulkL4Force(L4,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L6 != 0)
                {
                lcForce::bulkL6Force(L6,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            }
        else //near a boundary is less easy... ternary operators are slightly better than many ifs (particularly if boundaries are typically jagged)
            {
            lcForce::boundaryL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        d_types[ixd],d_types[ixu],d_types[iyd],
                        d_types[iyu],d_types[izd],d_types[izu],individualTerm);
            spatialTerm += individualTerm;
            int boundaryCase = lcForce::getBoundaryCase(d_types[ixd],d_types[ixu],d_types[iyd],
                                                        d_types[iyu],d_types[izd],d_types[izu]);
            if(L2 != 0)
                {
                lcForce::boundaryL2Force(L2,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L3 != 0)
                {
                lcForce::boundaryL3Force(L3,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L4 != 0)
                {
                lcForce::boundaryL4Force(L4,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            if(L6 != 0)
                {
                lcForce::boundaryL6Force(L6,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerm);
                spatialTerm += individualTerm;
                }
            };
        force -= spatialTerm;
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

__global__ void gpu_qTensor_uniformFieldForcekernel(dVec *d_force,
                                                    int *d_types,
                                                    int N,
                                                    scalar3 field,
                                                    scalar anisotropicSusceptibility,
                                                    scalar vacuumPermeability,
                                                    bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if(d_types[idx]>0)
        return;
    scalar fieldProduct = anisotropicSusceptibility*vacuumPermeability;
    dVec fieldForce(0.);
    fieldForce[0] = -0.5*fieldProduct*(field.x*field.x-field.z*field.z);
    fieldForce[1] = -fieldProduct*field.x*field.y;
    fieldForce[2] = -fieldProduct*field.x*field.z;
    fieldForce[3] = -0.5*fieldProduct*(field.y*field.y-field.z*field.z);
    fieldForce[4] = -fieldProduct*field.y*field.z;
    if(zeroForce)
        d_force[idx] = fieldForce;
    else
        d_force[idx] -= fieldForce;
    }

bool gpu_qTensor_computeObjectForceFromStresses(int *sites,
                                        int *latticeTypes,
                                        int *latticeNeighbors,
                                        Matrix3x3 *stress,
                                        scalar3 *objectForces,
                                        Index2D neighborIndex,
                                        int nSites,
                                        int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = nSites/block_size+1;
    gpu_qTensor_computeObjectForceFromStresses_kernel<<<nblocks,block_size>>>(sites,latticeTypes,latticeNeighbors,stress,objectForces,neighborIndex,nSites);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

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
                          int *latticeNeighbors,
                          Index2D neighborIndex,
                          int N,
                          int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_firstDerivatives_kernel<<<nblocks,block_size>>>(d_derivatives,d_spins,d_types,latticeNeighbors,neighborIndex,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_oneConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                int *d_latticeNeighbors,
                                Index2D neighborIndex,
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
    scalar l = L;
    gpu_qTensor_oneConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,d_latticeNeighbors,
                                                                neighborIndex,
                                                                a,b,c,l,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_multiConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                int *d_latticeNeighbors,
                                Index2D neighborIndex,
                                scalar A,scalar B,scalar C,
                                scalar L1,scalar L2,scalar L3,
                                scalar L4,scalar L6,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    gpu_qTensor_multiConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,
                                                                d_derivatives,d_latticeNeighbors,
                                                                neighborIndex,
                                                                a,b,c,
                                                                L1,L2,L3,L4,L6,
                                                                N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_computeUniformFieldForcesGPU(dVec * d_force,
                                       int *d_types,
                                       int N,
                                       scalar3 field,
                                       scalar anisotropicSusceptibility,
                                       scalar vacuumPermeability,
                                       bool zeroOutForce,
                                       int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_uniformFieldForcekernel<<<nblocks,block_size>>>(d_force,d_types,N,field,anisotropicSusceptibility,
                                                                vacuumPermeability, zeroOutForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };;

/** @} */ //end of group declaration
