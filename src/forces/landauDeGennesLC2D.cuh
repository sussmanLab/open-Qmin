#ifndef landauDeGennesLC2D_CUH
#define landauDeGennesLC2D_CUH
#include "std_include.h"
#include "indexer.h"

/*! \file landauDeGennesLC.cuh */
/** @addtogroup forceKernels force kernels
 * @{
 * \brief CUDA kernels and callers for force calculations
*/

bool gpu_2DqTensor_oneConstantForce(dVec *d_force, dVec *d_spins, int *d_types, int *d_latticeNeighbors, Index2D neighborIndex,
                                    scalar A,scalar C,scalar L, int N, bool zeroOutForce, int maxBlockSize);

/** @} */ //end of group declaration
#endif