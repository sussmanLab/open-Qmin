#ifndef landauDeGennesLC_CUH
#define landauDeGennesLC_CUH
#include "std_include.h"
#include "indexer.h"
#include "landauDeGennesLCBoundary.h"

/*! \file landauDeGennesLC.cuh */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

bool gpu_qTensor_oneConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                Index3D latticeIndex,
                                scalar A,scalar B,scalar C,scalar L,
                                int N,
                                bool zeroForce,
                                int maxBlockSize);

bool gpu_qTensor_threeConstantForce(dVec *d_force,
                                dVec *d_spins,
                                int *d_types,
                                cubicLatticeDerivativeVector *d_derivatives,
                                Index3D latticeIndex,
                                scalar A,scalar B,scalar C,scalar L1,scalar L2, scalar L3,
                                int N,
                                bool zeroForce,
                                int maxBlockSize);

bool gpu_qTensor_firstDerivatives(cubicLatticeDerivativeVector *d_derivatives,
                                dVec *d_spins,
                                int *d_types,
                                Index3D latticeIndex,
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

/** @} */ //end of group declaration
#endif
