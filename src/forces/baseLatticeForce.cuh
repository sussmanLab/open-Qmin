#ifndef baseLatticeForce_CUH
#define baseLatticeForce_CUH
#include "std_include.h"
#include "indexer.h"

/*! \file baseLatticeForce.cuh */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

bool gpu_lattice_spin_force_nn(dVec *d_force,
                                dVec *d_spins,
                                Index3D latticeIndex,
                                scalar J,
                                int N,
                                bool zeroForce,
                                int maxBlockSize);

/** @} */ //end of group declaration
#endif
