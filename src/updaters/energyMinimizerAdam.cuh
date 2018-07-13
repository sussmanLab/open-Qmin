#ifndef energyMinimizerAdam_CUH
#define energyMinimizerAdam_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file energyMinimizerAdam.cuh */

/** @addtogroup updaterKernels updater Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!run an adam minimization step
bool gpu_adam_step(dVec *force,
                   dVec *biasedMomentum,
                   dVec *biasedMomentum2,
                   dVec *correctedMomentum,
                   dVec *correctedMomentum2,
                   dVec *displacement,
                   scalar deltaT,
                   scalar beta1,
                   scalar beta2,
                   scalar beta1t,
                   scalar beta2t,
                   int N,
                   int blockSize);

/** @} */ //end of group declaration
#endif
