#ifndef landauDeGennesLC_CUH
#define landauDeGennesLC_CUH
#include "std_include.h"
#include "indexer.h"

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

/** @} */ //end of group declaration
#endif
