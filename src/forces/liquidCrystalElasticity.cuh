#ifndef liquidCrystalElasticity_CUH
#define liquidCrystalElasticity_CUH

#include "std_include.h"
#include "indexer.h"

/*! \file liquidCrystalElasticity.cuh */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

bool gpu_LdG_energy(
                    int N);

/** @} */ //end of group declaration
#endif
