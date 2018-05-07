#ifndef simpleModel_CUH
#define simpleModel_CUH

#include "std_include.h"
#include "periodicBoundaryConditions.h"
#include <cuda_runtime.h>

/*! \file simpleModel.cuh */

/** @addtogroup modelKernels model Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!pos += (scale)*disp, then put in box... done per thread
bool gpu_move_particles(dVec *d_pos,
                      dVec *d_disp,
                      periodicBoundaryConditions &Box,
                      scalar scale,
                      int N
                      );

/** @} */ //end of group declaration
#endif
