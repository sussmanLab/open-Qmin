#ifndef energyMinimizerFIRE_CUH
#define energyMinimizerFIRE_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file energyMinimizerFIRE.cuh */

/** @addtogroup updaterKernels updater Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!velocity = (1-a)velocity +a*scaling*force
bool gpu_update_velocity_FIRE(dVec *d_velocity,
                      dVec *d_force,
                      scalar alpha,
                      scalar scaling,
                      int N
                      );

/** @} */ //end of group declaration
#endif
