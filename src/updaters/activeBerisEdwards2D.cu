#include "activeBerisEdwards2D.cuh"
#include "../../inc/qTensorFunctions2D.h"



/*! \file activeBerisEdwards2D.cu

\addtogroup updaterKernels
@{
*/

__device__ dVec upwindAdvectiveDerivative(dVec &u, dVec &f, dVec &fxd, dVec &fyd, dVec &fxu, dVec &fyu, dVec &fxdd,
                                            dVec &fydd, dVec &fxuu, dVec &fyuu)
    {
    dVec ans;
    if(u[0] > 0)
        {
        ans = -0.5 * u[0] * (3. * f - 4 * fxd + 1. * fxdd);
        }
    else
        {
        ans = 0.5 * u[0] * (3. * f - 4 * fxu + 1. * fxuu);
        }

    if(u[1] > 0)
        {
        ans += -0.5 * u[1] * (3. * f - 4 * fyd + 1. * fydd);
        }
    else
        {
        ans += 0.5 * u[1] * (3. * f - 4 * fyu + 1. * fyuu);
        }
    return ans;
    }


__global__ void gpu_calculateMolecularFieldAdvectionStressGPU_kernel(dVec *Q, dVec *v, dVec *H, dVec *advection, dVec *PiS, 
                                                                        dVec *PiA, int *nearestNeighbors, scalar lambda, 
                                                                        scalar zeta, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Ndof) 
        {
        dVec q = Q[idx];
        dVec h = H[idx];

        // Lattice indices of four nearest neighbors
        int ixd = nearestNeighbors[8 * idx];
        int ixu = nearestNeighbors[8 * idx + 1];
        int iyd = nearestNeighbors[8 * idx + 2];
        int iyu = nearestNeighbors[8 * idx + 3];

        // Relevant strain and vorticity terms
        scalar dxux = 0.5 * (v[ixu].x[0] - v[ixd].x[0]);
        scalar dxuy = 0.5 * (v[ixu].x[1] - v[ixd].x[1]);
        scalar dyux = 0.5 * (v[iyu].x[0] - v[iyd].x[0]);
        scalar omegaxy = 0.5 * (dxuy - dyux);

        // Update the generalized advection and stress terms
        scalar localS = sqrt(q.x[0] * q.x[0] + q.x[1] * q.x[1]);
        advection[idx].x[0] = lambda * localS * dxux - 2.0 * omegaxy * q.x[1];
        advection[idx].x[1] = lambda * localS * 0.5 * (dxuy + dyux) + 2.0 * omegaxy * q.x[0];

        PiS[idx] = -lambda * h - zeta * q;
        PiA[idx].x[0] = 2.0 * (q.x[0] * h.x[1] - h.x[0] * q.x[1]);
        } 
    return;  
    }

__global__ void gpu_calculatePoissonPressure_kernel(dVec *v, dVec *PiS, int *nearestNeighbors, scalar *pRHS, int Ndof, 
                                                        scalar pseudotimestep)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    int ixd = nearestNeighbors[8 * idx];
    int ixu = nearestNeighbors[8 * idx + 1];
    int iyd = nearestNeighbors[8 * idx + 2];
    int iyu = nearestNeighbors[8 * idx + 3];
    int ixdyd = nearestNeighbors[8 * idx + 4];
    int ixdyu = nearestNeighbors[8 * idx + 5];
    int ixuyd = nearestNeighbors[8 * idx + 6];
    int ixuyu = nearestNeighbors[8 * idx + 7];

    scalar dudx = 0.5 * (v[ixu].x[0] - v[ixd].x[0]);
    scalar dudy = 0.5 * (v[iyu].x[0] - v[iyd].x[0]);
    scalar dvdy = 0.5 * (v[iyu].x[1] - v[iyd].x[1]);
    scalar dvdx = 0.5 * (v[ixu].x[1] - v[ixd].x[1]);


    pRHS[idx] = (1.0 / pseudotimestep)*(dudx + dvdy);

    //Add divergence of stress
    pRHS[idx] += (PiS[ixu].x[0] + PiS[ixd].x[0] - PiS[iyu].x[0] - PiS[iyd].x[0]) + 0.5*(
                                        PiS[ixuyu].x[1] - PiS[ixuyd].x[1] - PiS[ixdyu].x[1] + PiS[ixdyd].x[1]);
    
    //subtract dot product of velocity gradients
    pRHS[idx] += -(dudx*dudx + dvdy*dvdy + 2.0*dudy*dvdx);
    return;
    }


__global__ void gpu_updatePressureJacobi_kernel(scalar *p, scalar *pAux, scalar *pRHS, int *nearestNeighbors, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    int ixd = nearestNeighbors[8 * idx];
    int ixu = nearestNeighbors[8 * idx + 1];
    int iyd = nearestNeighbors[8 * idx + 2];
    int iyu = nearestNeighbors[8 * idx + 3];
    int ixdyd = nearestNeighbors[8 * idx + 4];
    int ixdyu = nearestNeighbors[8 * idx + 5];
    int ixuyd = nearestNeighbors[8 * idx + 6];
    int ixuyu = nearestNeighbors[8 * idx + 7];

    p[idx] = 0.05*(-6.0*pRHS[idx] 
                    + 4.0*(pAux[ixu] + pAux[iyu] + pAux[iyd] + pAux[ixd])
                    + pAux[ixdyd] + pAux[ixdyu] + pAux[ixuyd] + pAux[ixuyu]);
    return;
    }


__global__ void gpu_subtractpMeanPressureFromPressure_kernel(scalar *p, scalar *pMean, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    p[idx] = p[idx] - pMean[0]/((double)Ndof);
    return;
    }


__global__ void gpu_subtractPFromPAux_kernel(scalar *pAux, scalar *p, scalar *pAuxMinusPHolder, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    pAuxMinusPHolder[idx] = pAux[idx] - p[idx];
    return;
    }


__global__ void gpu_get_QFieldUpdate_kernel(dVec *disp, dVec *Q, dVec *v, dVec *H, dVec *advection, int *nearestNeighbors,
                                            int  *alternateNeighbors, scalar deltaT, scalar rotationalViscosity, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    int ixd = nearestNeighbors[8 * idx];
    int ixu = nearestNeighbors[8 * idx + 1];
    int iyd = nearestNeighbors[8 * idx + 2];
    int iyu = nearestNeighbors[8 * idx + 3];
    int ixdd = alternateNeighbors[4 * idx];
    int ixuu = alternateNeighbors[4 * idx + 1];
    int iydd = alternateNeighbors[4 * idx + 2];
    int iyuu = alternateNeighbors[4 * idx + 3];

    disp[idx] = deltaT * ((1.0/rotationalViscosity) * H[idx] + advection[idx] +
                                upwindAdvectiveDerivative(v[idx], Q[idx], Q[ixd], Q[iyd], Q[ixu], Q[iyu],
                                                            Q[ixdd], Q[iydd], Q[ixuu], Q[iyuu]));

    return;
    }


__global__ void gpu_get_velocityFieldUpdate_kernel(dVec *v, dVec *disp, dVec *PiS, dVec *PiA, scalar *p, int *nearestNeighbors, 
                                                    int *alternateNeighbors, scalar viscosity, scalar deltaT, scalar rho, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;

    dVec dudt;

    int ixd = nearestNeighbors[8 * idx];
    int ixu = nearestNeighbors[8 * idx + 1];
    int iyd = nearestNeighbors[8 * idx + 2];
    int iyu = nearestNeighbors[8 * idx + 3];
    int ixdyd = nearestNeighbors[8 * idx + 4];
    int ixdyu = nearestNeighbors[8 * idx + 5];
    int ixuyd = nearestNeighbors[8 * idx + 6];
    int ixuyu = nearestNeighbors[8 * idx + 7];
    int ixdd = alternateNeighbors[4 * idx];
    int ixuu = alternateNeighbors[4 * idx + 1];
    int iydd = alternateNeighbors[4 * idx + 2];
    int iyuu = alternateNeighbors[4 * idx + 3];
    
    //convective term
    dudt = upwindAdvectiveDerivative(v[idx], v[idx], v[ixd], v[iyd], v[ixu], v[iyu], v[ixdd], v[iydd], v[ixuu], v[iyuu]);
    //add viscous term
    dudt = dudt + laplacianStencil(viscosity, v[idx], v[ixd], v[ixu], v[iyd], v[iyu], v[ixdyd], v[ixuyd], v[ixdyu], v[ixuyu]);
    //add pressure and active/elastic stress terms: F_x = dx Pixx + dy Pixy
    dudt.x[0] += (0.5/rho)*(-(p[ixu] - p[ixd])
                            //F_x = dx Pixx + dy Pixy
                            + (PiS[ixu].x[0] - PiS[ixd].x[0]) 
                            + ((PiS[iyu].x[1] + PiA[iyu].x[0]) - (PiS[iyd].x[1] + PiA[iyd].x[0])));
    dudt.x[1] += (0.5/rho)*(-(p[iyu] - p[iyd])
                            //F_y = dx Piyx + dy Piyy = -dy Pixx + dx Piyx
                            - (PiS[iyu].x[0] - PiS[iyd].x[0]) 
                            + ((PiS[ixu].x[1] - PiA[ixu].x[0]) - (PiS[ixd].x[1] - PiA[ixd].x[0])));
    //scale by deltaT
    disp[idx] = deltaT * dudt;
    return;
    }


__global__ void gpu_updateAllVelocities_kernel(dVec *v, dVec *disp, int Ndof)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndof) return;
    v[idx] = v[idx] + disp[idx];
    } 


bool gpu_calculateMolecularFieldAdvectionStressGPU(dVec *Q, dVec *v, dVec *H, dVec *advection, dVec *PiS, dVec *PiA, 
                                                        int *nearestNeighbors, scalar lambda, scalar zeta, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_calculateMolecularFieldAdvectionStressGPU_kernel<<<nBlocks, blockSize>>>(Q, v, H, advection, PiS, PiA, 
                                                                                        nearestNeighbors, lambda, zeta,  Ndof);
    return cudaSuccess;
    }


bool gpu_calculatePoissonPressureRHS(dVec *v, dVec *PiS, int *nearestNeighbors, scalar *pRHS, int Ndof, scalar pseudotimestep)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_calculatePoissonPressure_kernel<<<nBlocks, blockSize>>>(v, PiS, nearestNeighbors, pRHS, Ndof, pseudotimestep);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }


bool gpu_updatePressureJacobi(scalar *p, scalar *pRHS, scalar *pAux, int *nearestNeighbors, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_updatePressureJacobi_kernel<<<nBlocks, blockSize>>>(p, pAux, pRHS, nearestNeighbors, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_subtractPFromPAux(scalar *pAux, scalar *p, scalar *pAuxMinusPHolder, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_subtractPFromPAux_kernel<<<nBlocks, blockSize>>>(pAux, p, pAuxMinusPHolder, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }


bool gpu_subtractpMeanPressureFromPressure(scalar *p, scalar *pMean, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_subtractpMeanPressureFromPressure_kernel<<<nBlocks, blockSize>>>(p, pMean, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_get_QField_update(dVec *disp, dVec *Q, dVec *v, dVec *H, dVec *advection, int *nearestNeighbors,
                                int *alternateNeighbors, scalar deltaT, scalar rotationalViscosity, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_get_QFieldUpdate_kernel<<<nBlocks, blockSize>>>(disp, Q, v, H, advection, nearestNeighbors, alternateNeighbors,
                                                        deltaT, rotationalViscosity, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_get_velocityFieldUpdate(dVec *v, dVec *disp, dVec *PiS, dVec *PiA, scalar *p, int *nearestNeighbors, int *alternateNeighbors,
                                     scalar viscosity, scalar deltaT, scalar rho, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_get_velocityFieldUpdate_kernel<<<nBlocks, blockSize>>>(v, disp, PiS, PiA, p, nearestNeighbors, alternateNeighbors, viscosity, deltaT,
                                                                rho, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }


bool gpu_updateAllVelocities(dVec *v, dVec *disp, int Ndof)
    {
    unsigned int blockSize = 128;
    if(Ndof < 128) blockSize = 16;
    unsigned int nBlocks = Ndof/blockSize + 1;
    gpu_updateAllVelocities_kernel<<<nBlocks, blockSize>>>(v, disp, Ndof);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
