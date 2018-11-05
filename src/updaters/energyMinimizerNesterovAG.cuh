#ifndef energyMinimizerNesterovAG_CUH
#define energyMinimizerNesterovAG_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file energyMinimizerNesterovAG.cuh */

/** @addtogroup updaterKernels updater Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!run an adam minimization step
bool gpu_nesterovAG_step(dVec *force,
                   dVec *position,
                   dVec *alternatePosition,
                   scalar deltaT,
                   scalar mu,
                   int N,
                   int blockSize);

/** @} */ //end of group declaration
#endif
