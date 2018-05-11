#ifndef noseHooverNVT_CUH
#define noseHooverNVT_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file noseHooverNVT.cuh */

/** @addtogroup updaterKernels updater Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!sequential update of the chain variables
bool gpu_propagate_noseHoover_chain(scalar *d_kes,
                                    scalar4 *d_bath,
                                    scalar deltaT,
                                    scalar temperature,
                                    int Nchain,
                                    int Ndof);

/** @} */ //end of group declaration
#endif

