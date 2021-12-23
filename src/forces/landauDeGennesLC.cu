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

__global__ void gpu_energyPerSite_kernel(scalar *energyPerSite,
                               dVec *Qtensors,
                               int *latticeTypes,
                               boundaryObject *bounds,
                               int *d_latticeNeighbors,
                               Index2D neighborIndex,
                               scalar a, scalar b, scalar c,
                               scalar L1, scalar L2, scalar L3, scalar L4, scalar L6,
                               bool computeEfieldContribution,
                               bool computeHfieldContribution,
                               scalar epsilon, scalar epsilon0, scalar deltaEpsilon, scalar3 Efield,
                               scalar Chi, scalar mu0, scalar deltaChi, scalar3 Hfield,
                               int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= N)
        return;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    energyPerSite[idx] = 0.0;
    if(latticeTypes[idx] <= 0) //only compute forces on sites that aren't boundaries
        {
        //phase part is simple
        qCurrent = Qtensors[idx];
        energyPerSite[idx] += a*TrQ2(qCurrent) + b*TrQ3(qCurrent) + c* TrQ2Squared(qCurrent);

        //external fields are also simple
        if(computeEfieldContribution)
            {
            scalar eFieldAtSite = epsilon0*(-0.5*Efield.x*Efield.x*(epsilon + deltaEpsilon*qCurrent[0]) -
                          deltaEpsilon*Efield.x*Efield.y*qCurrent[1] - deltaEpsilon*Efield.x*Efield.z*qCurrent[2] -
                          0.5*Efield.z*Efield.z*(epsilon - deltaEpsilon*qCurrent[0] - deltaEpsilon*qCurrent[3]) -
                          0.5*Efield.y*Efield.y*(epsilon + deltaEpsilon*qCurrent[3]) - deltaEpsilon*Efield.y*Efield.z*qCurrent[4]);
            energyPerSite[idx] += eFieldAtSite;
            }
        if(computeHfieldContribution)
            {
            scalar hFieldAtSite=mu0*(-0.5*Hfield.x*Hfield.x*(Chi + deltaChi*qCurrent[0]) -
                      deltaChi*Hfield.x*Hfield.y*qCurrent[1] - deltaChi*Hfield.x*Hfield.z*qCurrent[2] -
                      0.5*Hfield.z*Hfield.z*(Chi - deltaChi*qCurrent[0] - deltaChi*qCurrent[3]) -
                      0.5*Hfield.y*Hfield.y*(Chi + deltaChi*qCurrent[3]) - deltaChi*Hfield.y*Hfield.z*qCurrent[4]);
            energyPerSite[idx] +=hFieldAtSite;
            }

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        ixd =d_latticeNeighbors[neighborIndex(0,idx)];
        ixu =d_latticeNeighbors[neighborIndex(1,idx)];
        iyd =d_latticeNeighbors[neighborIndex(2,idx)];
        iyu =d_latticeNeighbors[neighborIndex(3,idx)];
        izd =d_latticeNeighbors[neighborIndex(4,idx)];
        izu =d_latticeNeighbors[neighborIndex(5,idx)];
        xDown = Qtensors[ixd]; xUp = Qtensors[ixu];
        yDown = Qtensors[iyd]; yUp = Qtensors[iyu];
        zDown = Qtensors[izd]; zUp = Qtensors[izu];

        dVec firstDerivativeX = 0.5*(xUp - xDown);
        dVec firstDerivativeY = 0.5*(yUp - yDown);
        dVec firstDerivativeZ = 0.5*(zUp - zDown);
        scalar anchoringEnergyAtSite = 0.0;
        if(latticeTypes[idx] <0)
            {
            if(latticeTypes[ixd]>0)
                {
                anchoringEnergyAtSite+= computeBoundaryEnergy(qCurrent, xDown, bounds[latticeTypes[ixd]-1]);
                firstDerivativeX = xUp - qCurrent;
                }
            if(latticeTypes[ixu]>0)
                {
                anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, xUp, bounds[latticeTypes[ixu]-1]);
                firstDerivativeX = qCurrent - xDown;
                }
            if(latticeTypes[iyd]>0)
                {
                anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, yDown, bounds[latticeTypes[iyd]-1]);
                firstDerivativeY = yUp - qCurrent;
                }
            if(latticeTypes[iyu]>0)
                {
                anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, yUp, bounds[latticeTypes[iyu]-1]);
                firstDerivativeY = qCurrent - yDown;
                }
            if(latticeTypes[izd]>0)
                {
                anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, zDown, bounds[latticeTypes[izd]-1]);
                firstDerivativeZ = zUp - qCurrent;
                }
            if(latticeTypes[izu]>0)
                {
                anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, zUp, bounds[latticeTypes[izu]-1]);
                firstDerivativeZ = qCurrent - zDown;
                }
            energyPerSite[idx] += anchoringEnergyAtSite;
            }
        scalar distortionEnergyAtSite=0.0;
        if(L1 !=0 )
            {
            distortionEnergyAtSite+=L1*(firstDerivativeX[0]*firstDerivativeX[3] + firstDerivativeY[0]*firstDerivativeY[3] + firstDerivativeZ[0]*firstDerivativeZ[3] + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeX[3]*firstDerivativeX[3] + firstDerivativeX[4]*firstDerivativeX[4] + firstDerivativeY[0]*firstDerivativeY[0]
                                    + firstDerivativeY[1]*firstDerivativeY[1] + firstDerivativeY[2]*firstDerivativeY[2] + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4] + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[1]*firstDerivativeZ[1] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]);
            };
        if(L2 !=0 )
            {
            distortionEnergyAtSite+=(L2*(2*firstDerivativeX[2]*firstDerivativeY[4] - 2*firstDerivativeX[2]*firstDerivativeZ[0] - 2*firstDerivativeY[4]*firstDerivativeZ[0] + 2*firstDerivativeY[1]*firstDerivativeZ[2] + 2*firstDerivativeX[0]*(firstDerivativeY[1] + firstDerivativeZ[2]) - 2*firstDerivativeX[2]*firstDerivativeZ[3] - 2*firstDerivativeY[4]*firstDerivativeZ[3] + 2*firstDerivativeZ[0]*firstDerivativeZ[3]
                                    + 2*firstDerivativeY[3]*firstDerivativeZ[4] + 2*firstDerivativeX[1]*(firstDerivativeY[3] + firstDerivativeZ[4]) + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeY[1]*firstDerivativeY[1] + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4]
                                    + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]))/2.;
            };
        if(L3 !=0 )
            {
            distortionEnergyAtSite+=(L3*(2*firstDerivativeX[1]*firstDerivativeY[0] + 2*firstDerivativeX[3]*firstDerivativeY[1] + 2*firstDerivativeX[4]*firstDerivativeY[2] + 2*firstDerivativeX[2]*firstDerivativeZ[0] + 2*firstDerivativeX[4]*firstDerivativeZ[1] + 2*firstDerivativeY[2]*firstDerivativeZ[1] - 2*firstDerivativeX[0]*firstDerivativeZ[2] - 2*firstDerivativeX[3]*firstDerivativeZ[2]
                                    + 2*firstDerivativeY[4]*firstDerivativeZ[3] + 2*firstDerivativeZ[0]*firstDerivativeZ[3] - 2*firstDerivativeY[0]*firstDerivativeZ[4] - 2*firstDerivativeY[3]*firstDerivativeZ[4] + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeY[1]*firstDerivativeY[1]
                                    + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4] + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]))/2.;
            };
        if(L4 !=0 )
            {
            distortionEnergyAtSite+=(L4*(-(firstDerivativeY[4]*qCurrent[0]) + firstDerivativeZ[4]*qCurrent[0] + firstDerivativeX[2]*qCurrent[1] - firstDerivativeY[4]*qCurrent[1] - firstDerivativeZ[2]*qCurrent[1] + firstDerivativeZ[4]*qCurrent[1] - firstDerivativeY[4]*qCurrent[2] + firstDerivativeZ[4]*qCurrent[2] + firstDerivativeX[2]*qCurrent[3] - firstDerivativeZ[2]*qCurrent[3]
                                    + firstDerivativeX[1]*(qCurrent[0] - qCurrent[2] + qCurrent[3] - qCurrent[4]) + firstDerivativeX[2]*qCurrent[4] - firstDerivativeZ[2]*qCurrent[4] + firstDerivativeY[1]*(-qCurrent[0] + qCurrent[2] - qCurrent[3] + qCurrent[4])))/2.;
            };
        if(L6 !=0 )
            {
            distortionEnergyAtSite+=L6*(-(firstDerivativeZ[0]*firstDerivativeZ[3]*qCurrent[0]) + firstDerivativeX[0]*firstDerivativeX[0]*qCurrent[0] + firstDerivativeX[1]*firstDerivativeX[1]*qCurrent[0] + firstDerivativeX[2]*firstDerivativeX[2]*qCurrent[0] + firstDerivativeX[3]*firstDerivativeX[3]*qCurrent[0] + firstDerivativeX[4]*firstDerivativeX[4]*qCurrent[0]
                                    - firstDerivativeZ[0]*firstDerivativeZ[0]*qCurrent[0] - firstDerivativeZ[1]*firstDerivativeZ[1]*qCurrent[0] - firstDerivativeZ[2]*firstDerivativeZ[2]*qCurrent[0] - firstDerivativeZ[3]*firstDerivativeZ[3]*qCurrent[0] - firstDerivativeZ[4]*firstDerivativeZ[4]*qCurrent[0] + firstDerivativeX[3]*firstDerivativeY[0]*qCurrent[1]
                                    + 2*firstDerivativeX[2]*firstDerivativeY[2]*qCurrent[1] + 2*firstDerivativeX[3]*firstDerivativeY[3]*qCurrent[1] + 2*firstDerivativeX[4]*firstDerivativeY[4]*qCurrent[1] + firstDerivativeX[3]*firstDerivativeZ[0]*qCurrent[2] + 2*firstDerivativeX[2]*firstDerivativeZ[2]*qCurrent[2] + 2*firstDerivativeX[3]*firstDerivativeZ[3]*qCurrent[2]
                                    + 2*firstDerivativeX[4]*firstDerivativeZ[4]*qCurrent[2] + 2*firstDerivativeX[1]*(firstDerivativeY[1]*qCurrent[1] + firstDerivativeZ[1]*qCurrent[2]) + firstDerivativeX[0]*(firstDerivativeX[3]*qCurrent[0] + 2*firstDerivativeY[0]*qCurrent[1] + firstDerivativeY[3]*qCurrent[1] + 2*firstDerivativeZ[0]*qCurrent[2] + firstDerivativeZ[3]*qCurrent[2])
                                    + firstDerivativeY[0]*firstDerivativeY[3]*qCurrent[3] - firstDerivativeZ[0]*firstDerivativeZ[3]*qCurrent[3] + firstDerivativeY[0]*firstDerivativeY[0]*qCurrent[3] + firstDerivativeY[1]*firstDerivativeY[1]*qCurrent[3] + firstDerivativeY[2]*firstDerivativeY[2]*qCurrent[3] + firstDerivativeY[3]*firstDerivativeY[3]*qCurrent[3]
                                    + firstDerivativeY[4]*firstDerivativeY[4]*qCurrent[3] - firstDerivativeZ[0]*firstDerivativeZ[0]*qCurrent[3] - firstDerivativeZ[1]*firstDerivativeZ[1]*qCurrent[3] - firstDerivativeZ[2]*firstDerivativeZ[2]*qCurrent[3] - firstDerivativeZ[3]*firstDerivativeZ[3]*qCurrent[3] - firstDerivativeZ[4]*firstDerivativeZ[4]*qCurrent[3]
                                    + 2*firstDerivativeY[0]*firstDerivativeZ[0]*qCurrent[4] + firstDerivativeY[3]*firstDerivativeZ[0]*qCurrent[4] + 2*firstDerivativeY[1]*firstDerivativeZ[1]*qCurrent[4] + 2*firstDerivativeY[2]*firstDerivativeZ[2]*qCurrent[4] + firstDerivativeY[0]*firstDerivativeZ[3]*qCurrent[4] + 2*firstDerivativeY[3]*firstDerivativeZ[3]*qCurrent[4] + 2*firstDerivativeY[4]*firstDerivativeZ[4]*qCurrent[4]);
            };
        energyPerSite[idx] +=distortionEnergyAtSite;
        };
    };

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
        if(d_types[idx] == 0 || d_types[idx] == -2) // bulk is easy
            {
            lcForce::bulkL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
            }
        else //near a boundary is less easy... ternary operators are slightly better than many ifs (particularly if boundaries are typically jagged)
            {
            lcForce::boundaryL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        d_types[ixd],d_types[ixu],d_types[iyd],
                        d_types[iyu],d_types[izd],d_types[izu],spatialTerm);
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

__global__ void gpu_qTensor_spatiallyVaryingFieldForcekernel(dVec *d_force,
                                                    int *d_types,
                                                    int N,
                                                    scalar3 *d_field,
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
    scalar3 field = d_field[idx];
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

__global__ void gpuCorrectForceFromMetric_kernel(dVec *d_force, int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    scalar QxxOld, QyyOld;
    QxxOld = d_force[idx][0];
    QyyOld = d_force[idx][3];
    d_force[idx][0] = 2.*(2.*QxxOld-QyyOld)/3.;
    d_force[idx][3] = 2.*(2.*QyyOld-QxxOld)/3.;
    }

bool gpuCorrectForceFromMetric(dVec *d_force,
                                int N,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpuCorrectForceFromMetric_kernel<<<nblocks,block_size>>>(d_force,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
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
    };

bool gpu_qTensor_computeSpatiallyVaryingFieldForcesGPU(dVec * d_force,
                                       int *d_types,
                                       int N,
                                       scalar3 *d_field,
                                       scalar anisotropicSusceptibility,
                                       scalar vacuumPermeability,
                                       bool zeroOutForce,
                                       int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_spatiallyVaryingFieldForcekernel<<<nblocks,block_size>>>(d_force,d_types,N,d_field,anisotropicSusceptibility,
                                                                vacuumPermeability, zeroOutForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

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
    };

bool gpu_computeAllEnergyTerms(scalar *energyPerSite,
                               dVec *Qtensors,
                               int *latticeTypes,
                               boundaryObject *bounds,
                               int *d_latticeNeighbors,
                               Index2D neighborIndex,
                               scalar a, scalar b, scalar c,
                               scalar L1, scalar L2, scalar L3, scalar L4, scalar L6,
                               bool computeEfieldContribution,
                               bool computeHfieldContribution,
                               scalar epsilon, scalar epsilon0, scalar deltaEpsilon, scalar3 Efield,
                               scalar Chi, scalar mu0, scalar deltaChi, scalar3 Hfield,
                               int N)
    {
    int maxBlockSize=256;
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_energyPerSite_kernel<<<nblocks,block_size>>>(energyPerSite,Qtensors,latticeTypes,bounds,
                                                             d_latticeNeighbors,neighborIndex,a,b,c,
                                                             L1,L2,L3,L4,L6,
                                                             computeEfieldContribution,computeHfieldContribution,
                                                             epsilon,epsilon0,deltaEpsilon,Efield,
                                                             Chi,mu0,deltaChi,Hfield,
                                                             N);
    cout << "NOTE: gpu_computeAllEnergyTerms DOES NOT correctly account for spatially varying external fields... if you have applied such a field, recompute the energy on the CPU" << endl;
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
