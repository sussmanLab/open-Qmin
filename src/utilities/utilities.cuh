#ifndef utilities_CUH__
#define utilities_CUH__

#include <cuda_runtime.h>
#include "std_include.h"
#include "gpuarray.h"

/*!
 \file utilities.cuh
A file providing an interface to the relevant cuda calls for some simple GPU array manipulations
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//!set every element of an array to the specified value
template<typename T>
bool gpu_set_array(T *arr,
                   T value,
                   int N,
                   int maxBlockSize=512);

//! (scalar) ans = (dVec) vec1 . vec2
bool gpu_dot_dVec_vectors(dVec *d_vec1,
                              dVec *d_vec2,
                              scalar  *d_ans,
                              int N);

//! (dVec) input *= factor
bool gpu_dVec_times_scalar(dVec *d_vec1,
                              scalar factor,
                              int N);
//! (dVec) ans = input * factor
bool gpu_dVec_times_scalar(dVec *d_vec1,
                              scalar factor,
                              dVec *d_ans,
                              int N);
//! ans = a*b[i]*c[i]^2r
bool gpu_scalar_times_dVec_squared(dVec *d_vec1,
                                   scalar *d_scalars,
                                   scalar factor,
                                   scalar *d_answer,
                                   int N);

//! vec1 += a*vec2
bool gpu_dVec_plusEqual_dVec(dVec *d_vec1,
                              dVec *d_vec2,
                              scalar factor,
                              int N,
                              int maxBlockSize = 512);

//!A trivial reduction of an array by one thread in serial. Think before you use this.
bool gpu_serial_reduction(
                    scalar *array,
                    scalar *output,
                    int helperIdx,
                    int N);

//!A straightforward two-step parallel reduction algorithm with block_size declared
bool gpu_parallel_reduction(
                    scalar *input,
                    scalar *intermediate,
                    scalar *output,
                    int helperIdx,
                    int N,
                    int block_size);

//!Take two vectors of dVecs and compute the sum of the dot products between them using thrust
bool gpu_dVec_dot_products(
                    dVec *input1,
                    dVec *input2,
                    scalar *output,
                    int helperIdx,
                    int N);

//!Take two vectors of dVecs and compute the sum of the dot products between them
bool gpu_dVec_dot_products(
                    dVec *input1,
                    dVec *input2,
                    scalar *intermediate,
                    scalar *intermediate2,
                    scalar *output,
                    int helperIdx,
                    int N,
                    int block_size);

//!Take two vectors of dVecs and compute the sum of the dot products between them on the host
scalar host_dVec_dot_products(dVec *input1,dVec *input2,int N);

//! vec1 += a*vec2... on the host!
void host_dVec_plusEqual_dVec(dVec *d_vec1,dVec *d_vec2,scalar factor,int N);

//! (dVec) ans = input * factor... on the host
void host_dVec_times_scalar(dVec *d_vec1,
                              scalar factor,
                              dVec *d_ans,
                              int N);

/** @} */ //end of group declaration
#endif
