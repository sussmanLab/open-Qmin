#ifndef landauDeGennesLC_CUH
#define landauDeGennesLC_CUH
#include "std_include.h"
#include "indexer.h"
#include "landauDeGennesLCBoundary.h"

/*! \file landauDeGennesLC.cuh */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers for force calculations
 */

bool gpuCorrectForceFromMetric(dVec *d_force,
                                int N,
                                int maxBlockSize);


bool gpu_qTensor_oneConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                int *d_latticeNeighbors,
                                Index2D neighborIndex,
                                scalar A,scalar B,scalar C,scalar L,
                                int N,
                                bool zeroForce,
                                int maxBlockSize);

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
                                int maxBlockSize);

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
                               int N);

bool gpu_qTensor_computeUniformFieldForcesGPU(dVec * d_force,
                                       int *d_types,
                                       int N,
                                       scalar3 field,
                                       scalar anisotropicSusceptibility,
                                       scalar vacuumPermeability,
                                       bool zeroOutForce,
                                       int maxBlockSize);

bool gpu_qTensor_computeSpatiallyVaryingFieldForcesGPU(dVec * d_force,
                                       int *d_types,
                                       int N,
                                       scalar3 *field,
                                       scalar anisotropicSusceptibility,
                                       scalar vacuumPermeability,
                                       bool zeroOutForce,
                                       int maxBlockSize);

bool gpu_qTensor_firstDerivatives(cubicLatticeDerivativeVector *d_derivatives,
                                dVec *d_spins,
                                int *d_types,
                                int *latticeNeighbors,
                                Index2D neighborIndex,
                                int N,
                                int maxBlockSize);

bool gpu_qTensor_computeBoundaryForcesGPU(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                boundaryObject *d_bounds,
                                Index3D latticeIndex,
                                int N,
                                bool zeroForce,
                                int maxBlockSize);

bool gpu_qTensor_computeObjectForceFromStresses(int *sites,
                                        int *latticeTypes,
                                        int *latticeNeighbors,
                                        Matrix3x3 *stress,
                                        scalar3 *objectForces,
                                        Index2D neighborIndex,
                                        int nSites,
                                        int maxBlockSize);

/** @} */ //end of group declaration
#endif
