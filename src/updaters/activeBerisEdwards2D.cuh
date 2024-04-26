#ifndef activeBerisEdwards2D_CUH
#define activeBerisEdwards2D_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file activeBerisEdwards2D.cuh */

/** addtogroup updaterKernels updater Kernels
@{
* \brief CUDA kernels and callers for updater classes
*/

//! calculate the vorticity tensor, strain tensor, generalized advection tensor and the symmetric/antisymmetric stress tensor
bool gpu_calculateMolecularFieldAdvectionStressGPU(dVec *Q, dVec *v, dVec *H, dVec *advection, dVec *PiS, dVec *PiA, 
                                                        int *nearestNeighbors, scalar lambda, scalar zeta, int Ndof);

//! adds divergence of stress minus the dot product of velocity gradient to rescaled (dudx + dvdy)
bool gpu_calculatePoissonPressureRHS(dVec *v, dVec *PiS, int *nearestNeighbors, scalar *pRHS, int Ndof, scalar pseudotimestep);

//! update pressure field using Auxiliary pressure and pRHS  
bool gpu_updatePressureJacobi(scalar *p, scalar *pRHS, scalar *pAux, int *nearestNeighbors, int Ndof);

//! to relax pressure, we need to subtract pressure from Auxiliary pressure
bool gpu_subtractPFromPAux(scalar *pAux, scalar *p, scalar *pAuxMinusPHolder, int Ndof);

//! p -> p - pMean
bool gpu_subtractpMeanPressureFromPressure(scalar *p, scalar pMean, int Ndof);

//! update Q field; use both nearest neighbors and alternate neighbors
bool gpu_get_QField_update(dVec *disp, dVec *Q, dVec *v, dVec *H, dVec *advection, int *nearestNeighbors,
                        int *alternateNeighbors, scalar deltaT, scalar rotationalViscosity, int Ndof);

//! calculate dv from the advective, viscous, pressure and active/elastic stress terms
bool gpu_get_velocityFieldUpdate(dVec *v, dVec *disp, dVec *PiS, dVec *PiA, scalar *p, int *nearestNeighbors, int *alternateNeighbors,
                                     scalar viscosity, scalar deltaT, scalar rho, int Ndof);

//! updates the velocity field for all field: v -> v + dv
bool gpu_updateAllVelocities(dVec *v, dVec *disp, int Ndof);

/** @} */ //end of group declaration
#endif
