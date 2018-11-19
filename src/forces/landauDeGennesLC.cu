#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
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
            if(d_types[ixd] <=0 &&d_types[ixu] <= 0) //x bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                    };
                }
            else if (d_types[ixu] > 0) //right is boundary
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
                    d_derivatives[idx][qq] = (xUp[qq]-qCurrent[qq]);
                    };
                };
            if(d_types[iyd] <=0 && d_types[iyu] <= 0) //y bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                    };
                }
            else if (d_types[iyu] > 0) //up is boundary
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
                    d_derivatives[idx][DIMENSION+qq] = (yUp[qq]-qCurrent[qq]);
                    };
                };
            if(d_types[izd] <=0 && d_types[izu] <= 0) //z bulk
                {
                for (int qq = 0; qq < DIMENSION; ++qq)
                    {
                    d_derivatives[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                    };
                }
            else if (d_types[izu] > 0) //up is boundary
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
                    d_derivatives[idx][2*DIMENSION+qq] = (zUp[qq]-qCurrent[qq]);
                    };
                };
            };
        };
    }

__global__ void gpu_qTensor_oneConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                int *d_latticeNeighbors,
                                Index3D latticeIndex,
                                Index2D neighborIndex,
                                scalar a,scalar b,scalar c,scalar L1,
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
            if(d_types[ixd]>0)//xDown is a boundary
                {
                spatialTerm -= (xUp-qCurrent);
                }
            if(d_types[ixu] >0)//xUp is a boundary
                {
                spatialTerm -= (xDown-qCurrent);//negative derivative and negative nu_x cancel
                }
            if(d_types[iyd]>1)//ydown
                {
                spatialTerm -= (yUp-qCurrent);
                }
            if(d_types[iyu] >0)
                {
                spatialTerm -= (yDown-qCurrent);//negative derivative and negative nu_y cancel
                }
            if(d_types[izd] >0)//zDown is boundary
                {
                spatialTerm -= (zUp-qCurrent);
                }
            if(d_types[izu] >0)
                {
                spatialTerm -= (zDown-qCurrent);//negative derivative and negative nu_z cancel
                }
            scalar crossTerm = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += crossTerm;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += crossTerm;
            spatialTerm[4] *= 2.0;
            };
        force -= L1*spatialTerm;
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

__global__ void gpu_qTensor_twoConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                scalar a,scalar b,scalar c,scalar L1,scalar L2, scalar q0,
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
        gpu_phase_force(qCurrent, a, b, c, force);

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);
        xDown = d_spins[ixd]; xUp = d_spins[ixu];
        yDown = d_spins[iyd]; yUp = d_spins[iyu];
        zDown = d_spins[izd]; zUp = d_spins[izu];
        cubicLatticeDerivativeVector qCurrentDerivative = d_derivatives[idx];
        cubicLatticeDerivativeVector xDownDerivative = d_derivatives[ixd];
        cubicLatticeDerivativeVector xUpDerivative = d_derivatives[ixu];
        cubicLatticeDerivativeVector yDownDerivative = d_derivatives[iyd];
        cubicLatticeDerivativeVector yUpDerivative = d_derivatives[iyu];
        cubicLatticeDerivativeVector zDownDerivative = d_derivatives[izd];
        cubicLatticeDerivativeVector zUpDerivative = d_derivatives[izu];

        dVec xMinusTerm(0.0);
        dVec xPlusTerm(0.0);
        dVec yMinusTerm(0.0);
        dVec yPlusTerm(0.0);
        dVec zMinusTerm(0.0);
        dVec zPlusTerm(0.0);
        if(d_types[ixd] <= 0) //xMinus
            {
            xMinusTerm[0]=-(L1*(2*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) + qCurrentDerivative[12] - 2*xDown[0] - 2*xDown[3] - 18*q0*(qCurrentDerivative[11] - 2*(qCurrentDerivative[7] + xDown[4])) + xDownDerivative[12]))/2. - (L2*(2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] - 2*xDown[0] + xDownDerivative[6] + xDownDerivative[12]))/2.;

            xMinusTerm[1]=(L1*(qCurrentDerivative[5] - 2*(qCurrent[1] + 324*(q0*q0)*qCurrent[1] - xDown[1] + 9*q0*(qCurrentDerivative[9] + qCurrentDerivative[10] - qCurrentDerivative[13] + 2*xDown[2])) + xDownDerivative[5]))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xDown[1] + xDownDerivative[8] + xDownDerivative[14]))/2.;

            xMinusTerm[2]=-(L1*(1296*(q0*q0)*qCurrent[2] - 36*q0*(2*qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xDown[1]) - 2*(-2*qCurrent[2] + qCurrentDerivative[10] + 2*xDown[2] + xDownDerivative[10])))/4. + (L2*(-2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*xDown[2] - xDownDerivative[9] + xDownDerivative[10] + xDownDerivative[13]))/2.;

            xMinusTerm[3]=-(L1*(2*qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 4*qCurrent[3] + 648*(q0*q0)*qCurrent[3] - qCurrentDerivative[6] + 18*q0*qCurrentDerivative[7] + 18*q0*qCurrentDerivative[11] + qCurrentDerivative[12] - 2*xDown[0] - 4*xDown[3] + 72*q0*xDown[4] - xDownDerivative[6] + xDownDerivative[12]))/2.;

            xMinusTerm[4]=(L1*(-4*qCurrent[4] - 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[7] + qCurrentDerivative[11] + 18*q0*(qCurrentDerivative[6] - qCurrentDerivative[12] + 2*xDown[0] + 4*xDown[3]) + 4*xDown[4] + xDownDerivative[7] + xDownDerivative[11]))/2.;
            }
        if(d_types[ixu] <= 0) //xPlus
            {
            xPlusTerm[0]=(L2*(-2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] + 2*xUp[0] + xUpDerivative[6] + xUpDerivative[12]))/2. - (L1*(648*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) - 36*q0*(-2*qCurrentDerivative[7] + qCurrentDerivative[11] + 2*xUp[4]) - 2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[12] + 2*xUp[0] + 2*xUp[3] + xUpDerivative[12])))/4.;

            xPlusTerm[1]=-(L1*(qCurrentDerivative[5] + 2*(qCurrent[1] + 324*(q0*q0)*qCurrent[1] - xUp[1] + 9*q0*(qCurrentDerivative[9] + qCurrentDerivative[10] - qCurrentDerivative[13] - 2*xUp[2])) + xUpDerivative[5]))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xUp[1] + xUpDerivative[8] + xUpDerivative[14]))/2.;

            xPlusTerm[2]=-(L1*(2*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[10] - 18*q0*(2*qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xUp[1]) - 2*xUp[2] + xUpDerivative[10]))/2. - (L2*(2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*xUp[2] - xUpDerivative[9] + xUpDerivative[10] + xUpDerivative[13]))/2.;

            xPlusTerm[3]=-(L1*(2*qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 4*qCurrent[3] + 648*(q0*q0)*qCurrent[3] + qCurrentDerivative[6] + 18*q0*qCurrentDerivative[7] + 18*q0*qCurrentDerivative[11] - qCurrentDerivative[12] - 2*xUp[0] - 4*xUp[3] - 72*q0*xUp[4] + xUpDerivative[6] - xUpDerivative[12]))/2.;

            xPlusTerm[4]=-(L1*(4*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[7] + qCurrentDerivative[11] - 18*q0*(qCurrentDerivative[6] - qCurrentDerivative[12] - 2*xUp[0] - 4*xUp[3]) - 4*xUp[4] + xUpDerivative[7] + xUpDerivative[11]))/2.;
            }

        if(d_types[iyd] <= 0) //yMinus
            {
            yMinusTerm[0]=-(L1*(4*qCurrent[0] + 648*(q0*q0)*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*qCurrent[3] - qCurrentDerivative[1] - 18*q0*qCurrentDerivative[4] - 18*q0*qCurrentDerivative[11] + qCurrentDerivative[14] - 4*yDown[0] - 72*q0*yDown[2] - 2*yDown[3] - yDownDerivative[1] + yDownDerivative[14]))/2.;

            yMinusTerm[1]=-(L1*(1296*(q0*q0)*qCurrent[1] + 36*q0*(-qCurrentDerivative[2] + qCurrentDerivative[10] - qCurrentDerivative[13] - 2*yDown[4]) - 2*(-2*qCurrent[1] + qCurrentDerivative[3] + 2*yDown[1] + yDownDerivative[3])))/4. - (L2*(2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] - 2*yDown[1] + yDownDerivative[0] + yDownDerivative[12]))/2.;

            yMinusTerm[2]=(L1*(-4*qCurrent[2] - 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[4] + qCurrentDerivative[11] + 4*yDown[2] - 18*q0*(qCurrentDerivative[1] - qCurrentDerivative[14] + 4*yDown[0] + 2*yDown[3]) + yDownDerivative[4] + yDownDerivative[11]))/2.;

            yMinusTerm[3]=-(L1*(2*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + qCurrentDerivative[14] - 2*yDown[0] + 18*q0*(qCurrentDerivative[11] - 2*(qCurrentDerivative[4] + yDown[2])) - 2*yDown[3] + yDownDerivative[14]))/2. - (L2*(2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] - 2*yDown[3] + yDownDerivative[1] + yDownDerivative[14]));

            yMinusTerm[4]=(L2*(-2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*yDown[4] - yDownDerivative[2] + yDownDerivative[10] + yDownDerivative[13]))/2. - (L1*(1296*(q0*q0)*qCurrent[4] + 36*q0*(qCurrentDerivative[0] + qCurrentDerivative[12] + 2*(qCurrentDerivative[3] + yDown[1])) - 2*(-2*qCurrent[4] + qCurrentDerivative[13] + 2*yDown[4] + yDownDerivative[13])))/4.;
            }

        if(d_types[iyu] <= 0) //yPlus
            {
            yPlusTerm[0]=-(L1*(4*qCurrent[0] + 648*(q0*q0)*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*qCurrent[3] + qCurrentDerivative[1] - 18*q0*qCurrentDerivative[4] - 18*q0*qCurrentDerivative[11] - qCurrentDerivative[14] - 4*yUp[0] + 72*q0*yUp[2] - 2*yUp[3] + yUpDerivative[1] - yUpDerivative[14]))/2.;

            yPlusTerm[1]=-(L1*(2*qCurrent[1] + 648*(q0*q0)*qCurrent[1] + qCurrentDerivative[3] - 2*yUp[1] + 18*q0*(-qCurrentDerivative[2] + qCurrentDerivative[10] - qCurrentDerivative[13] + 2*yUp[4]) + yUpDerivative[3]))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] + 2*yUp[1] + yUpDerivative[0] + yUpDerivative[12]))/2.;

            yPlusTerm[2]=-(L1*(4*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[4] + qCurrentDerivative[11] - 4*yUp[2] + 18*q0*(qCurrentDerivative[1] - qCurrentDerivative[14] - 4*yUp[0] - 2*yUp[3]) + yUpDerivative[4] + yUpDerivative[11]))/2.;

            yPlusTerm[3]=(L2*(-2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] + 2*yUp[3] + yUpDerivative[1] + yUpDerivative[14]))/2. - (L1*(648*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + 36*q0*(-2*qCurrentDerivative[4] + qCurrentDerivative[11] + 2*yUp[2]) - 2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[14] + 2*yUp[0] + 2*yUp[3] + yUpDerivative[14])))/4.;

            yPlusTerm[4]=-(L1*(2*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[13] + 18*q0*(qCurrentDerivative[0] + 2*qCurrentDerivative[3] + qCurrentDerivative[12] - 2*yUp[1]) - 2*yUp[4] + yUpDerivative[13]))/2. - (L2*(2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*yUp[4] - yUpDerivative[2] + yUpDerivative[10] + yUpDerivative[13]))/2.;
            }

        if(d_types[izd] <= 0) //zMinus
            {
            zMinusTerm[0]=(L1*(qCurrentDerivative[2] - 2*(qCurrent[0] + 162*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) - zDown[0] + 9*q0*(-qCurrentDerivative[4] + 2*qCurrentDerivative[7] + 2*zDown[1])) + zDownDerivative[2]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

            zMinusTerm[1]=-(L1*(1296*(q0*q0)*qCurrent[1] - 36*q0*(qCurrentDerivative[2] - qCurrentDerivative[9] + 2*zDown[0] - 2*zDown[3]) - 2*(-4*qCurrent[1] + qCurrentDerivative[4] + qCurrentDerivative[7] + 4*zDown[1] + zDownDerivative[4] + zDownDerivative[7])))/4.;

            zMinusTerm[2]=-(L1*(2*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[0] + 18*q0*qCurrentDerivative[1] + qCurrentDerivative[3] - 36*q0*qCurrentDerivative[5] - 18*q0*qCurrentDerivative[8] - 2*zDown[2] + 36*q0*zDown[4] + zDownDerivative[0] + zDownDerivative[3]))/2. - (L2*(2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] - 2*zDown[2] + zDownDerivative[0] + zDownDerivative[6]))/2.;

            zMinusTerm[3]=(L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2. - (L1*(648*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + 36*q0*(qCurrentDerivative[7] - 2*(qCurrentDerivative[4] + zDown[1])) - 2*(-2*qCurrent[3] + qCurrentDerivative[9] + 2*zDown[3] + zDownDerivative[9])))/4.;

            zMinusTerm[4]=-(L2*(2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] - 2*zDown[4] + zDownDerivative[1] + zDownDerivative[8]))/2. - (L1*(2*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[5] + qCurrentDerivative[8] + 18*q0*(qCurrentDerivative[0] + 2*qCurrentDerivative[3] - qCurrentDerivative[6] - 2*zDown[2]) - 2*zDown[4] + zDownDerivative[5] + zDownDerivative[8]))/2.;
            }

        if(d_types[izu] <= 0) //zPlus
            {
            zPlusTerm[0]=-(L1*(qCurrentDerivative[2] + 2*(qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 162*(q0*q0)*qCurrent[3] - 9*q0*qCurrentDerivative[4] + 18*q0*qCurrentDerivative[7] - zUp[0] - 18*q0*zUp[1]) + zUpDerivative[2]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

            zPlusTerm[1]=-(L1*(4*qCurrent[1] + 648*(q0*q0)*qCurrent[1] + qCurrentDerivative[4] + qCurrentDerivative[7] - 4*zUp[1] - 18*q0*(qCurrentDerivative[2] - qCurrentDerivative[9] - 2*zUp[0] + 2*zUp[3]) + zUpDerivative[4] + zUpDerivative[7]))/2.;

            zPlusTerm[2]=(L1*(-2*qCurrent[2] - 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[0] - 18*q0*qCurrentDerivative[1] + qCurrentDerivative[3] + 36*q0*qCurrentDerivative[5] + 18*q0*qCurrentDerivative[8] + 2*zUp[2] + 36*q0*zUp[4] + zUpDerivative[0] + zUpDerivative[3]))/2. + (L2*(-2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] + 2*zUp[2] + zUpDerivative[0] + zUpDerivative[6]))/2.;

            zPlusTerm[3]=-(L1*(2*qCurrent[3] + 324*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + qCurrentDerivative[9] + 18*q0*(-2*qCurrentDerivative[4] + qCurrentDerivative[7] + 2*zUp[1]) - 2*zUp[3] + zUpDerivative[9]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

            zPlusTerm[4]=(L2*(-2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[1] + zUpDerivative[8]))/2. - (L1*(1296*(q0*q0)*qCurrent[4] + 36*q0*(qCurrentDerivative[0] - qCurrentDerivative[6] + 2*(qCurrentDerivative[3] + zUp[2])) - 2*(-2*qCurrent[4] + qCurrentDerivative[5] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[5] + zUpDerivative[8])))/4.;
            }

        force += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;

        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
    }

__global__ void gpu_qTensor_threeConstantForce_kernel(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                scalar a,scalar b,scalar c,scalar L1,scalar L2, scalar L3,
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
        gpu_phase_force(qCurrent, a, b, c, force);

        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);
        xDown = d_spins[ixd]; xUp = d_spins[ixu];
        yDown = d_spins[iyd]; yUp = d_spins[iyu];
        zDown = d_spins[izd]; zUp = d_spins[izu];
        cubicLatticeDerivativeVector qCurrentDerivative = d_derivatives[idx];
        cubicLatticeDerivativeVector xDownDerivative = d_derivatives[ixd];
        cubicLatticeDerivativeVector xUpDerivative = d_derivatives[ixu];
        cubicLatticeDerivativeVector yDownDerivative = d_derivatives[iyd];
        cubicLatticeDerivativeVector yUpDerivative = d_derivatives[iyu];
        cubicLatticeDerivativeVector zDownDerivative = d_derivatives[izd];
        cubicLatticeDerivativeVector zUpDerivative = d_derivatives[izu];

        dVec xMinusTerm(0.0);
        dVec xPlusTerm(0.0);
        dVec yMinusTerm(0.0);
        dVec yPlusTerm(0.0);
        dVec zMinusTerm(0.0);
        dVec zPlusTerm(0.0);
        if(d_types[ixd] <= 0) //xMinus
            {
            xMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*xDown[0] - 4*xDown[3]))/4. - (L2*(2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] - 2*xDown[0] + xDownDerivative[6] + xDownDerivative[12]))/2. - (L3*(3*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + qCurrent[3]*qCurrent[3] + qCurrent[4]*qCurrent[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xDown[0]*xDown[0] + xDown[1]*xDown[1] + xDown[2]*xDown[2] + xDown[3]*xDown[3] + xDown[4]*xDown[4] + 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[5] + qCurrent[1]*qCurrentDerivative[8] + qCurrent[2]*qCurrentDerivative[13] - qCurrentDerivative[10]*(-2*qCurrent[2] + qCurrentDerivative[13]) - 2*qCurrent[0]*xDown[0] - 2*qCurrent[1]*xDown[1] - 2*qCurrent[2]*xDown[2] - 2*qCurrent[0]*xDown[3] - 2*qCurrent[3]*xDown[3] - 2*qCurrent[4]*xDown[4] + 2*xDown[1]*xDownDerivative[5] + xDown[1]*xDownDerivative[8] + 2*xDown[2]*xDownDerivative[10] + xDown[2]*xDownDerivative[13]))/2.;

            xMinusTerm[1]=-2*L1*(qCurrent[1] - xDown[1]) - (L3*(qCurrentDerivative[8]*(qCurrent[0] + 2*qCurrent[3] - xDown[0] - 2*xDown[3]) + qCurrentDerivative[5]*(2*qCurrent[0] + qCurrent[3] - 2*xDown[0] - xDown[3]) + 2*(qCurrent[0]*qCurrent[1] + qCurrent[2]*qCurrentDerivative[7] + qCurrent[4]*qCurrentDerivative[9] + qCurrent[2]*qCurrentDerivative[11] + qCurrent[1]*xDown[0] - qCurrent[0]*xDown[1] - xDown[0]*xDown[1] - qCurrentDerivative[6]*(-2*qCurrent[1] + xDown[1]) - qCurrentDerivative[7]*xDown[2] - qCurrentDerivative[9]*xDown[4] + xDown[1]*xDownDerivative[6] + xDown[2]*xDownDerivative[11])))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xDown[1] + xDownDerivative[8] + xDownDerivative[14]))/2.;

            xMinusTerm[2]=-2*L1*(qCurrent[2] - xDown[2]) - (L3*(qCurrentDerivative[13]*(qCurrent[0] + 2*qCurrent[3] - xDown[0] - 2*xDown[3]) + qCurrentDerivative[10]*(2*qCurrent[0] + qCurrent[3] - 2*xDown[0] - xDown[3]) + 2*(qCurrent[0]*qCurrent[2] + qCurrent[1]*qCurrentDerivative[7] + 2*qCurrent[2]*qCurrentDerivative[12] + qCurrent[4]*qCurrentDerivative[14] + qCurrent[2]*xDown[0] + qCurrentDerivative[11]*(qCurrent[1] - xDown[1]) - qCurrent[0]*xDown[2] - qCurrentDerivative[12]*xDown[2] - xDown[0]*xDown[2] - qCurrentDerivative[14]*xDown[4] + xDown[1]*xDownDerivative[7] + xDown[2]*xDownDerivative[12])))/2. + (L2*(-2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*xDown[2] - xDownDerivative[9] + xDownDerivative[10] + xDownDerivative[13]))/2.;

            xMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*xDown[0] - 8*xDown[3]))/4. - (L3*(qCurrent[0]*qCurrent[0] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xDown[0]*xDown[0] + 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[8] + qCurrentDerivative[5]*(qCurrent[1] + qCurrentDerivative[8]) + qCurrentDerivative[10]*(qCurrent[2] - qCurrentDerivative[13]) + 2*qCurrent[2]*qCurrentDerivative[13] + 2*qCurrent[3]*xDown[0] - 2*qCurrent[0]*xDown[3] - 2*xDown[0]*xDown[3] + xDown[1]*xDownDerivative[5] + 2*xDown[1]*xDownDerivative[8] + xDown[2]*xDownDerivative[10] + 2*xDown[2]*xDownDerivative[13]))/2.;

            xMinusTerm[4]=-2*L1*(qCurrent[4] - xDown[4]) - (L3*(2*qCurrent[1]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*qCurrentDerivative[10] + qCurrentDerivative[8]*qCurrentDerivative[10] + 2*qCurrentDerivative[6]*qCurrentDerivative[11] + 2*qCurrentDerivative[7]*qCurrentDerivative[12] + qCurrentDerivative[5]*qCurrentDerivative[13] + 2*qCurrentDerivative[8]*qCurrentDerivative[13] + 2*qCurrent[2]*qCurrentDerivative[14] + 2*qCurrentDerivative[9]*qCurrentDerivative[14] + 2*qCurrent[0]*(qCurrent[4] - xDown[4]) + 2*xDown[0]*(qCurrent[4] - xDown[4]) + 2*xDown[1]*xDownDerivative[9] + 2*xDown[2]*xDownDerivative[14]))/2.;
            }
        if(d_types[ixu] <= 0) //xPlus
            {
            xPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*xUp[0] - xUp[3])) + (L2*(-2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] + 2*xUp[0] + xUpDerivative[6] + xUpDerivative[12]))/2. + (L3*(-3*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - qCurrent[3]*qCurrent[3] - qCurrent[4]*qCurrent[4] + qCurrentDerivative[10]*qCurrentDerivative[10] + qCurrentDerivative[11]*qCurrentDerivative[11] + qCurrentDerivative[12]*qCurrentDerivative[12] + qCurrentDerivative[13]*qCurrentDerivative[13] + qCurrentDerivative[14]*qCurrentDerivative[14] + xUp[0]*xUp[0] - xUp[1]*xUp[1] - xUp[2]*xUp[2] - xUp[3]*xUp[3] - xUp[4]*xUp[4] - 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[5] + qCurrent[1]*qCurrentDerivative[8] + qCurrent[2]*qCurrentDerivative[13] + qCurrentDerivative[10]*(2*qCurrent[2] + qCurrentDerivative[13]) + 2*qCurrent[0]*xUp[0] + 2*qCurrent[1]*xUp[1] + 2*qCurrent[2]*xUp[2] + 2*qCurrent[0]*xUp[3] + 2*qCurrent[3]*xUp[3] + 2*qCurrent[4]*xUp[4] + 2*xUp[1]*xUpDerivative[5] + xUp[1]*xUpDerivative[8] + 2*xUp[2]*xUpDerivative[10] + xUp[2]*xUpDerivative[13]))/2.;

            xPlusTerm[1]=-2*L1*(qCurrent[1] - xUp[1]) - (L3*(qCurrentDerivative[5]*(-2*qCurrent[0] - qCurrent[3] + 2*xUp[0] + xUp[3]) + qCurrentDerivative[8]*(-qCurrent[0] - 2*qCurrent[3] + xUp[0] + 2*xUp[3]) - 2*(qCurrent[2]*qCurrentDerivative[7] + qCurrent[4]*qCurrentDerivative[9] + qCurrent[2]*qCurrentDerivative[11] - qCurrent[1]*xUp[0] + qCurrentDerivative[6]*(2*qCurrent[1] - xUp[1]) + xUp[0]*xUp[1] + qCurrent[0]*(-qCurrent[1] + xUp[1]) - qCurrentDerivative[7]*xUp[2] - qCurrentDerivative[9]*xUp[4] + xUp[1]*xUpDerivative[6] + xUp[2]*xUpDerivative[11])))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xUp[1] + xUpDerivative[8] + xUpDerivative[14]))/2.;

            xPlusTerm[2]=-2*L1*(qCurrent[2] - xUp[2]) - (L3*(qCurrentDerivative[10]*(-2*qCurrent[0] - qCurrent[3] + 2*xUp[0] + xUp[3]) + qCurrentDerivative[13]*(-qCurrent[0] - 2*qCurrent[3] + xUp[0] + 2*xUp[3]) - 2*(-(qCurrent[0]*qCurrent[2]) + qCurrent[1]*qCurrentDerivative[7] + 2*qCurrent[2]*qCurrentDerivative[12] + qCurrent[4]*qCurrentDerivative[14] - qCurrent[2]*xUp[0] + qCurrentDerivative[11]*(qCurrent[1] - xUp[1]) + qCurrent[0]*xUp[2] - qCurrentDerivative[12]*xUp[2] + xUp[0]*xUp[2] - qCurrentDerivative[14]*xUp[4] + xUp[1]*xUpDerivative[7] + xUp[2]*xUpDerivative[12])))/2. - (L2*(2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*xUp[2] - xUpDerivative[9] + xUpDerivative[10] + xUpDerivative[13]))/2.;

            xPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - xUp[0] - 2*xUp[3])) - (L3*(qCurrent[0]*qCurrent[0] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xUp[0]*xUp[0] + 2*qCurrent[0]*qCurrent[3] - 2*qCurrent[1]*qCurrentDerivative[8] + qCurrentDerivative[5]*(-qCurrent[1] + qCurrentDerivative[8]) - 2*qCurrent[2]*qCurrentDerivative[13] - qCurrentDerivative[10]*(qCurrent[2] + qCurrentDerivative[13]) + 2*qCurrent[3]*xUp[0] - 2*qCurrent[0]*xUp[3] - 2*xUp[0]*xUp[3] - xUp[1]*xUpDerivative[5] - 2*xUp[1]*xUpDerivative[8] - xUp[2]*xUpDerivative[10] - 2*xUp[2]*xUpDerivative[13]))/2.;

            xPlusTerm[4]=-2*L1*(qCurrent[4] - xUp[4]) - (L3*(-2*qCurrent[1]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*qCurrentDerivative[10] + qCurrentDerivative[8]*qCurrentDerivative[10] + 2*qCurrentDerivative[6]*qCurrentDerivative[11] + 2*qCurrentDerivative[7]*qCurrentDerivative[12] + qCurrentDerivative[5]*qCurrentDerivative[13] + 2*qCurrentDerivative[8]*qCurrentDerivative[13] - 2*qCurrent[2]*qCurrentDerivative[14] + 2*qCurrentDerivative[9]*qCurrentDerivative[14] + 2*qCurrent[0]*(qCurrent[4] - xUp[4]) + 2*xUp[0]*(qCurrent[4] - xUp[4]) - 2*xUp[1]*xUpDerivative[9] - 2*xUp[2]*xUpDerivative[14]))/2.;
            }
        if(d_types[iyd] <= 0) //yMinus
            {
            yMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*yDown[0] - 4*yDown[3]))/4. - (L3*(qCurrent[3]*qCurrent[3] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - yDown[3]*yDown[3] + 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[0]*(2*qCurrent[1] + qCurrentDerivative[3]) + qCurrent[4]*qCurrentDerivative[13] - qCurrentDerivative[10]*(-2*qCurrent[4] + qCurrentDerivative[13]) - 2*qCurrent[3]*yDown[0] + 2*qCurrent[0]*yDown[3] - 2*yDown[0]*yDown[3] + 2*yDown[1]*yDownDerivative[0] + yDown[1]*yDownDerivative[3] + 2*yDown[4]*yDownDerivative[10] + yDown[4]*yDownDerivative[13]))/2.;

            yMinusTerm[1]=-2*L1*(qCurrent[1] - yDown[1]) - (L3*(qCurrentDerivative[3]*(qCurrent[0] + 2*qCurrent[3] - yDown[0] - 2*yDown[3]) + qCurrentDerivative[0]*(2*qCurrent[0] + qCurrent[3] - 2*yDown[0] - yDown[3]) + 2*(qCurrent[1]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[2] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[11] - qCurrent[3]*yDown[1] - qCurrentDerivative[1]*(-2*qCurrent[1] + yDown[1]) - qCurrentDerivative[2]*yDown[2] + qCurrent[1]*yDown[3] - yDown[1]*yDown[3] - qCurrentDerivative[4]*yDown[4] + yDown[1]*yDownDerivative[1] + yDown[4]*yDownDerivative[11])))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] - 2*yDown[1] + yDownDerivative[0] + yDownDerivative[12]))/2.;

            yMinusTerm[2]=-2*L1*(qCurrent[2] - yDown[2]) - (L3*(2*qCurrent[1]*qCurrentDerivative[2] + 2*qCurrentDerivative[0]*qCurrentDerivative[10] + qCurrentDerivative[3]*qCurrentDerivative[10] + 2*qCurrentDerivative[1]*qCurrentDerivative[11] + 2*qCurrent[4]*qCurrentDerivative[12] + 2*qCurrentDerivative[2]*qCurrentDerivative[12] + qCurrentDerivative[0]*qCurrentDerivative[13] + 2*qCurrentDerivative[3]*qCurrentDerivative[13] + 2*qCurrentDerivative[4]*qCurrentDerivative[14] + 2*qCurrent[3]*(qCurrent[2] - yDown[2]) + 2*(qCurrent[2] - yDown[2])*yDown[3] + 2*yDown[1]*yDownDerivative[2] + 2*yDown[4]*yDownDerivative[12]))/2.;

            yMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*yDown[0] - 8*yDown[3]))/4. - (L3*(qCurrent[0]*qCurrent[0] + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 3*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] + yDown[0]*yDown[0] + yDown[1]*yDown[1] + yDown[2]*yDown[2] - yDown[3]*yDown[3] + yDown[4]*yDown[4] + 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[0] + 2*qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[10]*(qCurrent[4] - qCurrentDerivative[13]) + 2*qCurrent[4]*qCurrentDerivative[13] - 2*qCurrent[0]*yDown[0] - 2*qCurrent[3]*yDown[0] - 2*qCurrent[1]*yDown[1] - 2*qCurrent[2]*yDown[2] - 2*qCurrent[3]*yDown[3] - 2*qCurrent[4]*yDown[4] + yDown[1]*yDownDerivative[0] + 2*yDown[1]*yDownDerivative[3] + yDown[4]*yDownDerivative[10] + 2*yDown[4]*yDownDerivative[13]))/2. - (L2*(2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] - 2*yDown[3] + yDownDerivative[1] + yDownDerivative[14]))/2.;

            yMinusTerm[4]=-2*L1*(qCurrent[4] - yDown[4]) + (L2*(-2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*yDown[4] - yDownDerivative[2] + yDownDerivative[10] + yDownDerivative[13]))/2. - (L3*(qCurrentDerivative[13]*(qCurrent[0] + 2*qCurrent[3] - yDown[0] - 2*yDown[3]) + qCurrentDerivative[10]*(2*qCurrent[0] + qCurrent[3] - 2*yDown[0] - yDown[3]) + 2*(qCurrent[3]*qCurrent[4] + qCurrent[1]*qCurrentDerivative[4] + qCurrent[2]*qCurrentDerivative[12] + 2*qCurrent[4]*qCurrentDerivative[14] + qCurrentDerivative[11]*(qCurrent[1] - yDown[1]) - qCurrentDerivative[12]*yDown[2] + qCurrent[4]*yDown[3] - qCurrent[3]*yDown[4] - qCurrentDerivative[14]*yDown[4] - yDown[3]*yDown[4] + yDown[1]*yDownDerivative[4] + yDown[4]*yDownDerivative[14])))/2.;
            }

        if(d_types[iyu] <= 0) //yPlus
            {
            yPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*yUp[0] - yUp[3])) - (L3*(qCurrent[3]*qCurrent[3] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - yUp[3]*yUp[3] + 2*qCurrent[0]*qCurrent[3] - qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[0]*(-2*qCurrent[1] + qCurrentDerivative[3]) - qCurrent[4]*qCurrentDerivative[13] - qCurrentDerivative[10]*(2*qCurrent[4] + qCurrentDerivative[13]) - 2*qCurrent[3]*yUp[0] + 2*qCurrent[0]*yUp[3] - 2*yUp[0]*yUp[3] - 2*yUp[1]*yUpDerivative[0] - yUp[1]*yUpDerivative[3] - 2*yUp[4]*yUpDerivative[10] - yUp[4]*yUpDerivative[13]))/2.;

            yPlusTerm[1]=-2*L1*(qCurrent[1] - yUp[1]) - (L3*(qCurrentDerivative[0]*(-2*qCurrent[0] - qCurrent[3] + 2*yUp[0] + yUp[3]) + qCurrentDerivative[3]*(-qCurrent[0] - 2*qCurrent[3] + yUp[0] + 2*yUp[3]) - 2*(-(qCurrent[1]*qCurrent[3]) + qCurrent[2]*qCurrentDerivative[2] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[11] + qCurrentDerivative[1]*(2*qCurrent[1] - yUp[1]) + qCurrent[3]*yUp[1] - qCurrentDerivative[2]*yUp[2] - qCurrent[1]*yUp[3] + yUp[1]*yUp[3] - qCurrentDerivative[4]*yUp[4] + yUp[1]*yUpDerivative[1] + yUp[4]*yUpDerivative[11])))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] + 2*yUp[1] + yUpDerivative[0] + yUpDerivative[12]))/2.;

            yPlusTerm[2]=-2*L1*(qCurrent[2] - yUp[2]) - (L3*(-2*qCurrent[1]*qCurrentDerivative[2] + 2*qCurrentDerivative[0]*qCurrentDerivative[10] + qCurrentDerivative[3]*qCurrentDerivative[10] + 2*qCurrentDerivative[1]*qCurrentDerivative[11] - 2*qCurrent[4]*qCurrentDerivative[12] + 2*qCurrentDerivative[2]*qCurrentDerivative[12] + qCurrentDerivative[0]*qCurrentDerivative[13] + 2*qCurrentDerivative[3]*qCurrentDerivative[13] + 2*qCurrentDerivative[4]*qCurrentDerivative[14] + 2*qCurrent[3]*(qCurrent[2] - yUp[2]) + 2*(qCurrent[2] - yUp[2])*yUp[3] - 2*yUp[1]*yUpDerivative[2] - 2*yUp[4]*yUpDerivative[12]))/2.;

            yPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - yUp[0] - 2*yUp[3])) + (L3*(-(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 3*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[10]*qCurrentDerivative[10] + qCurrentDerivative[11]*qCurrentDerivative[11] + qCurrentDerivative[12]*qCurrentDerivative[12] + qCurrentDerivative[13]*qCurrentDerivative[13] + qCurrentDerivative[14]*qCurrentDerivative[14] - yUp[0]*yUp[0] - yUp[1]*yUp[1] - yUp[2]*yUp[2] + yUp[3]*yUp[3] - yUp[4]*yUp[4] - 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[0] + 2*qCurrent[1]*qCurrentDerivative[3] + 2*qCurrent[4]*qCurrentDerivative[13] + qCurrentDerivative[10]*(qCurrent[4] + qCurrentDerivative[13]) + 2*qCurrent[0]*yUp[0] + 2*qCurrent[3]*yUp[0] + 2*qCurrent[1]*yUp[1] + 2*qCurrent[2]*yUp[2] + 2*qCurrent[3]*yUp[3] + 2*qCurrent[4]*yUp[4] + yUp[1]*yUpDerivative[0] + 2*yUp[1]*yUpDerivative[3] + yUp[4]*yUpDerivative[10] + 2*yUp[4]*yUpDerivative[13]))/2. + (L2*(-2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] + 2*yUp[3] + yUpDerivative[1] + yUpDerivative[14]))/2.;

            yPlusTerm[4]=-2*L1*(qCurrent[4] - yUp[4]) - (L2*(2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*yUp[4] - yUpDerivative[2] + yUpDerivative[10] + yUpDerivative[13]))/2. - (L3*(qCurrentDerivative[10]*(-2*qCurrent[0] - qCurrent[3] + 2*yUp[0] + yUp[3]) + qCurrentDerivative[13]*(-qCurrent[0] - 2*qCurrent[3] + yUp[0] + 2*yUp[3]) - 2*(-(qCurrent[3]*qCurrent[4]) + qCurrent[1]*qCurrentDerivative[4] + qCurrent[2]*qCurrentDerivative[12] + 2*qCurrent[4]*qCurrentDerivative[14] + qCurrentDerivative[11]*(qCurrent[1] - yUp[1]) - qCurrentDerivative[12]*yUp[2] - qCurrent[4]*yUp[3] + qCurrent[3]*yUp[4] - qCurrentDerivative[14]*yUp[4] + yUp[3]*yUp[4] + yUp[1]*yUpDerivative[4] + yUp[4]*yUpDerivative[14])))/2.;
            }

        if(d_types[izd] <= 0) //zMinus
            {
            zMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*zDown[0] - 4*zDown[3]))/4. - (L3*(-3*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 2*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] + zDown[0]*zDown[0] - zDown[1]*zDown[1] - zDown[2]*zDown[2] - zDown[4]*zDown[4] - 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[3] + qCurrentDerivative[0]*(2*qCurrent[2] + qCurrentDerivative[3]) + 2*qCurrent[4]*qCurrentDerivative[5] + qCurrent[4]*qCurrentDerivative[8] + 2*qCurrent[0]*zDown[0] + 2*qCurrent[3]*zDown[0] + 2*qCurrent[1]*zDown[1] + 2*qCurrent[2]*zDown[2] + 2*qCurrent[3]*zDown[3] + 2*zDown[0]*zDown[3] + 2*qCurrent[4]*zDown[4] + 2*zDown[2]*zDownDerivative[0] + zDown[2]*zDownDerivative[3] + 2*zDown[4]*zDownDerivative[5] + zDown[4]*zDownDerivative[8]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

            zMinusTerm[1]=-2*L1*(qCurrent[1] - zDown[1]) - (L3*(2*qCurrent[2]*qCurrentDerivative[1] + 2*qCurrentDerivative[0]*qCurrentDerivative[5] + qCurrentDerivative[3]*qCurrentDerivative[5] + 2*qCurrent[4]*qCurrentDerivative[6] + 2*qCurrentDerivative[1]*qCurrentDerivative[6] + 2*qCurrentDerivative[2]*qCurrentDerivative[7] + qCurrentDerivative[0]*qCurrentDerivative[8] + 2*qCurrentDerivative[3]*qCurrentDerivative[8] + 2*qCurrentDerivative[4]*qCurrentDerivative[9] + 2*qCurrent[0]*(-qCurrent[1] + zDown[1]) + 2*qCurrent[3]*(-qCurrent[1] + zDown[1]) + 2*zDown[0]*(-qCurrent[1] + zDown[1]) + 2*(-qCurrent[1] + zDown[1])*zDown[3] + 2*zDown[2]*zDownDerivative[1] + 2*zDown[4]*zDownDerivative[6]))/2.;

            zMinusTerm[2]=-2*L1*(qCurrent[2] - zDown[2]) - (L2*(2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] - 2*zDown[2] + zDownDerivative[0] + zDownDerivative[6]))/2. - (L3*(2*qCurrent[2]*qCurrentDerivative[2] + 2*qCurrent[4]*qCurrentDerivative[7] + 2*qCurrentDerivative[0]*(qCurrent[0] - zDown[0]) + qCurrentDerivative[3]*(qCurrent[0] - zDown[0]) + 2*qCurrentDerivative[1]*(qCurrent[1] - zDown[1]) + 2*qCurrentDerivative[2]*(qCurrent[2] - zDown[2]) + 2*qCurrent[0]*(-qCurrent[2] + zDown[2]) + 2*qCurrent[3]*(-qCurrent[2] + zDown[2]) + 2*zDown[0]*(-qCurrent[2] + zDown[2]) + qCurrentDerivative[0]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[3]*(qCurrent[3] - zDown[3]) + 2*(-qCurrent[2] + zDown[2])*zDown[3] + 2*qCurrentDerivative[4]*(qCurrent[4] - zDown[4]) + 2*zDown[2]*zDownDerivative[2] + 2*zDown[4]*zDownDerivative[7]))/2.;

            zMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*zDown[0] - 8*zDown[3]))/4. - (L3*(-2*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 3*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - zDown[1]*zDown[1] - zDown[2]*zDown[2] + zDown[3]*zDown[3] - zDown[4]*zDown[4] - 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[0] + 2*qCurrent[2]*qCurrentDerivative[3] + 2*qCurrent[4]*qCurrentDerivative[8] + qCurrentDerivative[5]*(qCurrent[4] + qCurrentDerivative[8]) + 2*qCurrent[0]*zDown[0] + 2*qCurrent[1]*zDown[1] + 2*qCurrent[2]*zDown[2] + 2*qCurrent[0]*zDown[3] + 2*qCurrent[3]*zDown[3] + 2*zDown[0]*zDown[3] + 2*qCurrent[4]*zDown[4] + zDown[2]*zDownDerivative[0] + 2*zDown[2]*zDownDerivative[3] + zDown[4]*zDownDerivative[5] + 2*zDown[4]*zDownDerivative[8]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

            zMinusTerm[4]=-2*L1*(qCurrent[4] - zDown[4]) - (L2*(2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] - 2*zDown[4] + zDownDerivative[1] + zDownDerivative[8]))/2. - (L3*(2*qCurrent[2]*qCurrentDerivative[4] + 2*qCurrent[4]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*(qCurrent[0] - zDown[0]) + qCurrentDerivative[8]*(qCurrent[0] - zDown[0]) + 2*qCurrentDerivative[6]*(qCurrent[1] - zDown[1]) + 2*qCurrentDerivative[7]*(qCurrent[2] - zDown[2]) + qCurrentDerivative[5]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[8]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[9]*(qCurrent[4] - zDown[4]) + 2*qCurrent[0]*(-qCurrent[4] + zDown[4]) + 2*qCurrent[3]*(-qCurrent[4] + zDown[4]) + 2*zDown[0]*(-qCurrent[4] + zDown[4]) + 2*zDown[3]*(-qCurrent[4] + zDown[4]) + 2*zDown[2]*zDownDerivative[4] + 2*zDown[4]*zDownDerivative[9]))/2.;
            }

        if(d_types[izu] <= 0) //zPlus
            {
            zPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*zUp[0] - zUp[3])) + (L3*(3*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 2*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[0]*qCurrentDerivative[0] - qCurrentDerivative[1]*qCurrentDerivative[1] - qCurrentDerivative[2]*qCurrentDerivative[2] - qCurrentDerivative[3]*qCurrentDerivative[3] - qCurrentDerivative[4]*qCurrentDerivative[4] - zUp[0]*zUp[0] + zUp[1]*zUp[1] + zUp[2]*zUp[2] + zUp[4]*zUp[4] + 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[3] - qCurrentDerivative[0]*(-2*qCurrent[2] + qCurrentDerivative[3]) + 2*qCurrent[4]*qCurrentDerivative[5] + qCurrent[4]*qCurrentDerivative[8] - 2*qCurrent[0]*zUp[0] - 2*qCurrent[3]*zUp[0] - 2*qCurrent[1]*zUp[1] - 2*qCurrent[2]*zUp[2] - 2*qCurrent[3]*zUp[3] - 2*zUp[0]*zUp[3] - 2*qCurrent[4]*zUp[4] + 2*zUp[2]*zUpDerivative[0] + zUp[2]*zUpDerivative[3] + 2*zUp[4]*zUpDerivative[5] + zUp[4]*zUpDerivative[8]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

            zPlusTerm[1]=-2*L1*(qCurrent[1] - zUp[1]) - (L3*(-2*qCurrent[2]*qCurrentDerivative[1] + 2*qCurrentDerivative[0]*qCurrentDerivative[5] + qCurrentDerivative[3]*qCurrentDerivative[5] - 2*qCurrent[4]*qCurrentDerivative[6] + 2*qCurrentDerivative[1]*qCurrentDerivative[6] + 2*qCurrentDerivative[2]*qCurrentDerivative[7] + qCurrentDerivative[0]*qCurrentDerivative[8] + 2*qCurrentDerivative[3]*qCurrentDerivative[8] + 2*qCurrentDerivative[4]*qCurrentDerivative[9] + 2*qCurrent[0]*(-qCurrent[1] + zUp[1]) + 2*qCurrent[3]*(-qCurrent[1] + zUp[1]) + 2*zUp[0]*(-qCurrent[1] + zUp[1]) + 2*(-qCurrent[1] + zUp[1])*zUp[3] - 2*zUp[2]*zUpDerivative[1] - 2*zUp[4]*zUpDerivative[6]))/2.;

            zPlusTerm[2]=-2*L1*(qCurrent[2] - zUp[2]) + (L2*(-2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] + 2*zUp[2] + zUpDerivative[0] + zUpDerivative[6]))/2. - (L3*(qCurrentDerivative[0]*(-2*qCurrent[0] - qCurrent[3] + 2*zUp[0] + zUp[3]) + qCurrentDerivative[3]*(-qCurrent[0] - 2*qCurrent[3] + zUp[0] + 2*zUp[3]) - 2*(qCurrent[0]*qCurrent[2] + qCurrent[2]*qCurrent[3] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[7] + qCurrent[2]*zUp[0] + qCurrentDerivative[1]*(qCurrent[1] - zUp[1]) + qCurrentDerivative[2]*(2*qCurrent[2] - zUp[2]) - qCurrent[0]*zUp[2] - qCurrent[3]*zUp[2] - zUp[0]*zUp[2] + qCurrent[2]*zUp[3] - zUp[2]*zUp[3] - qCurrentDerivative[4]*zUp[4] + zUp[2]*zUpDerivative[2] + zUp[4]*zUpDerivative[7])))/2.;

            zPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - zUp[0] - 2*zUp[3])) + (L3*(2*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 3*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[5]*qCurrentDerivative[5] - qCurrentDerivative[6]*qCurrentDerivative[6] - qCurrentDerivative[7]*qCurrentDerivative[7] - qCurrentDerivative[8]*qCurrentDerivative[8] - qCurrentDerivative[9]*qCurrentDerivative[9] + zUp[1]*zUp[1] + zUp[2]*zUp[2] - zUp[3]*zUp[3] + zUp[4]*zUp[4] + 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[0] + 2*qCurrent[2]*qCurrentDerivative[3] + qCurrentDerivative[5]*(qCurrent[4] - qCurrentDerivative[8]) + 2*qCurrent[4]*qCurrentDerivative[8] - 2*qCurrent[0]*zUp[0] - 2*qCurrent[1]*zUp[1] - 2*qCurrent[2]*zUp[2] - 2*qCurrent[0]*zUp[3] - 2*qCurrent[3]*zUp[3] - 2*zUp[0]*zUp[3] - 2*qCurrent[4]*zUp[4] + zUp[2]*zUpDerivative[0] + 2*zUp[2]*zUpDerivative[3] + zUp[4]*zUpDerivative[5] + 2*zUp[4]*zUpDerivative[8]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

            zPlusTerm[4]=-2*L1*(qCurrent[4] - zUp[4]) + (L2*(-2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[1] + zUpDerivative[8]))/2. - (L3*(qCurrentDerivative[5]*(-2*qCurrent[0] - qCurrent[3] + 2*zUp[0] + zUp[3]) + qCurrentDerivative[8]*(-qCurrent[0] - 2*qCurrent[3] + zUp[0] + 2*zUp[3]) - 2*(qCurrent[0]*qCurrent[4] + qCurrent[3]*qCurrent[4] + qCurrent[2]*qCurrentDerivative[4] + 2*qCurrent[4]*qCurrentDerivative[9] + qCurrent[4]*zUp[0] + qCurrentDerivative[6]*(qCurrent[1] - zUp[1]) + qCurrentDerivative[7]*(qCurrent[2] - zUp[2]) + qCurrent[4]*zUp[3] - qCurrent[0]*zUp[4] - qCurrent[3]*zUp[4] - qCurrentDerivative[9]*zUp[4] - zUp[0]*zUp[4] - zUp[3]*zUp[4] + zUp[2]*zUpDerivative[4] + zUp[4]*zUpDerivative[9])))/2.;
            }

        force += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;

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

__global__ void gpu_qTensor_l24Force_kernel(dVec *d_force,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                int N,
                                scalar L24,
                                bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int3 target = latticeIndex.inverseIndex(idx);
    int3 latticeSizes = latticeIndex.getSizes();
    dVec force(0.0);

    if(d_types[idx] <= 0) //no force on sites that are part of boundaries
        {
        //get neighbor indices and data
        int ixd, ixu,iyd,iyu,izd,izu;
        gpu_get_six_neighbors(target,ixd, ixu,iyd,iyu,izd,izu,latticeIndex,latticeSizes);
        cubicLatticeDerivativeVector qCurrentDerivative = d_derivatives[idx];
        cubicLatticeDerivativeVector xDownDerivative = d_derivatives[ixd];
        cubicLatticeDerivativeVector xUpDerivative = d_derivatives[ixu];
        cubicLatticeDerivativeVector yDownDerivative = d_derivatives[iyd];
        cubicLatticeDerivativeVector yUpDerivative = d_derivatives[iyu];
        cubicLatticeDerivativeVector zDownDerivative = d_derivatives[izd];
        cubicLatticeDerivativeVector zUpDerivative = d_derivatives[izu];

        dVec xMinusTerm(0.0);
        dVec xPlusTerm(0.0);
        dVec yMinusTerm(0.0);
        dVec yPlusTerm(0.0);
        dVec zMinusTerm(0.0);
        dVec zPlusTerm(0.0);
        if(d_types[ixd] <= 0) //xMinus
            {
            xMinusTerm[0]=(3*L24*(qCurrentDerivative[6] + xDownDerivative[6] + 2*(qCurrentDerivative[12] + xDownDerivative[12])))/2.;

            xMinusTerm[1]=(-3*L24*(qCurrentDerivative[5] - qCurrentDerivative[8] - qCurrentDerivative[14] + xDownDerivative[5] - xDownDerivative[8] - xDownDerivative[14]))/2.;

            xMinusTerm[2]=(-3*L24*(-qCurrentDerivative[9] + 2*qCurrentDerivative[10] + qCurrentDerivative[13] - xDownDerivative[9] + 2*xDownDerivative[10] + xDownDerivative[13]))/2.;

            xMinusTerm[3]=(-3*L24*(qCurrentDerivative[6] - qCurrentDerivative[12] + xDownDerivative[6] - xDownDerivative[12]))/2.;

            xMinusTerm[4]=(-3*L24*(qCurrentDerivative[7] + qCurrentDerivative[11] + xDownDerivative[7] + xDownDerivative[11]))/2.;
            }
        if(d_types[ixu] <= 0) //xPlus
            {
            xPlusTerm[0]=(-3*L24*(qCurrentDerivative[6] + xUpDerivative[6] + 2*(qCurrentDerivative[12] + xUpDerivative[12])))/2.;

            xPlusTerm[1]=(-3*L24*(-qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] - xUpDerivative[5] + xUpDerivative[8] + xUpDerivative[14]))/2.;

            xPlusTerm[2]=(3*L24*(-qCurrentDerivative[9] + 2*qCurrentDerivative[10] + qCurrentDerivative[13] - xUpDerivative[9] + 2*xUpDerivative[10] + xUpDerivative[13]))/2.;

            xPlusTerm[3]=(3*L24*(qCurrentDerivative[6] - qCurrentDerivative[12] + xUpDerivative[6] - xUpDerivative[12]))/2.;

            xPlusTerm[4]=(3*L24*(qCurrentDerivative[7] + qCurrentDerivative[11] + xUpDerivative[7] + xUpDerivative[11]))/2.;
            }

        if(d_types[iyd] <= 0) //yMinus
            {
            yMinusTerm[0]=(-3*L24*(qCurrentDerivative[1] - qCurrentDerivative[14] + yDownDerivative[1] - yDownDerivative[14]))/2.;

            yMinusTerm[1]=(3*L24*(qCurrentDerivative[0] - qCurrentDerivative[3] + qCurrentDerivative[12] + yDownDerivative[0] - yDownDerivative[3] + yDownDerivative[12]))/2.;

            yMinusTerm[2]=(-3*L24*(qCurrentDerivative[4] + qCurrentDerivative[11] + yDownDerivative[4] + yDownDerivative[11]))/2.;

            yMinusTerm[3]=(3*L24*(qCurrentDerivative[1] + yDownDerivative[1] + 2*(qCurrentDerivative[14] + yDownDerivative[14])))/2.;

            yMinusTerm[4]=(-3*L24*(-qCurrentDerivative[2] + qCurrentDerivative[10] + 2*qCurrentDerivative[13] - yDownDerivative[2] + yDownDerivative[10] + 2*yDownDerivative[13]))/2.;
            }

        if(d_types[iyu] <= 0) //yPlus
            {
            yPlusTerm[0]=(3*L24*(qCurrentDerivative[1] - qCurrentDerivative[14] + yUpDerivative[1] - yUpDerivative[14]))/2.;

            yPlusTerm[1]=(-3*L24*(qCurrentDerivative[0] - qCurrentDerivative[3] + qCurrentDerivative[12] + yUpDerivative[0] - yUpDerivative[3] + yUpDerivative[12]))/2.;

            yPlusTerm[2]=(3*L24*(qCurrentDerivative[4] + qCurrentDerivative[11] + yUpDerivative[4] + yUpDerivative[11]))/2.;

            yPlusTerm[3]=(-3*L24*(qCurrentDerivative[1] + yUpDerivative[1] + 2*(qCurrentDerivative[14] + yUpDerivative[14])))/2.;

            yPlusTerm[4]=(3*L24*(-qCurrentDerivative[2] + qCurrentDerivative[10] + 2*qCurrentDerivative[13] - yUpDerivative[2] + yUpDerivative[10] + 2*yUpDerivative[13]))/2.;
            }

        if(d_types[izd] <= 0) //zMinus
            {
            zMinusTerm[0]=(-3*L24*(2*qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDownDerivative[2] + zDownDerivative[9]))/2.;

            zMinusTerm[1]=(-3*L24*(qCurrentDerivative[4] + qCurrentDerivative[7] + zDownDerivative[4] + zDownDerivative[7]))/2.;

            zMinusTerm[2]=(3*L24*(2*qCurrentDerivative[0] + qCurrentDerivative[3] + qCurrentDerivative[6] + 2*zDownDerivative[0] + zDownDerivative[3] + zDownDerivative[6]))/2.;

            zMinusTerm[3]=(-3*L24*(qCurrentDerivative[2] + zDownDerivative[2] + 2*(qCurrentDerivative[9] + zDownDerivative[9])))/2.;

            zMinusTerm[4]=(3*L24*(qCurrentDerivative[1] + qCurrentDerivative[5] + 2*qCurrentDerivative[8] + zDownDerivative[1] + zDownDerivative[5] + 2*zDownDerivative[8]))/2.;
            }

        if(d_types[izu] <= 0) //zPlus
            {
            zPlusTerm[0]=(3*L24*(2*qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zUpDerivative[2] + zUpDerivative[9]))/2.;

            zPlusTerm[1]=(3*L24*(qCurrentDerivative[4] + qCurrentDerivative[7] + zUpDerivative[4] + zUpDerivative[7]))/2.;

            zPlusTerm[2]=(-3*L24*(2*qCurrentDerivative[0] + qCurrentDerivative[3] + qCurrentDerivative[6] + 2*zUpDerivative[0] + zUpDerivative[3] + zUpDerivative[6]))/2.;

            zPlusTerm[3]=(3*L24*(qCurrentDerivative[2] + zUpDerivative[2] + 2*(qCurrentDerivative[9] + zUpDerivative[9])))/2.;

            zPlusTerm[4]=(-3*L24*(qCurrentDerivative[1] + qCurrentDerivative[5] + 2*qCurrentDerivative[8] + zUpDerivative[1] + zUpDerivative[5] + 2*zUpDerivative[8]))/2.;
            }

        force += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;
        };
    if(zeroForce)
        d_force[idx] = force;
    else
        d_force[idx] += force;
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
                                int *d_latticeNeighbors,
                                Index3D latticeIndex,
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
                                                                latticeIndex,neighborIndex,
                                                                a,b,c,l,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_twoConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                scalar A,scalar B,scalar C,scalar L1,scalar L2, scalar q0,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    gpu_qTensor_twoConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,d_derivatives,latticeIndex,
                                                             a,b,c,L1,L2,q0,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_threeConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                scalar A,scalar B,scalar C,scalar L1,scalar L2, scalar L3,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    gpu_qTensor_threeConstantForce_kernel<<<nblocks,block_size>>>(d_force,d_spins,d_types,d_derivatives,latticeIndex,
                                                             a,b,c,L1,L2,L3,N,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_qTensor_computeL24ForcesGPU(dVec *d_force,
                                    int *d_types,
                                    cubicLatticeDerivativeVector *d_derivatives,
                                    Index3D latticeIndex,
                                    int N, scalar L24,
                                    bool zeroOutForce,
                                    int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;
    gpu_qTensor_l24Force_kernel<<<nblocks,block_size>>>(d_force,d_types,d_derivatives,latticeIndex,
                                                                 N,L24,zeroOutForce);
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
    };;

/** @} */ //end of group declaration
