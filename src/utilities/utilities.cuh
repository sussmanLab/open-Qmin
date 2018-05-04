#ifndef utilities_CUH__
#define utilities_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
/*!
 \file utilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//!convenience function to zero out an array on the GPU
bool gpu_zero_array(int *arr,
                    int N
                    );
//!convenience function to zero out an array on the GPU
bool gpu_zero_array(unsigned int *arr,
                    int      N
                    );
//!A trivial reduction of an array by one thread in serial. Think before you use this.
bool gpu_serial_reduction(
                    scalar *array,
                    scalar *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm.
bool gpu_parallel_reduction(
                    scalar *input,
                    scalar *intermediate,
                    scalar *output,
                    int helperIdx,
                    int N);

/** @} */ //end of group declaration
#endif
