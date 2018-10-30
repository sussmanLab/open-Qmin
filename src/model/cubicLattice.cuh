#ifndef cubicLattice_CUH
#define cubicLattice_CUH

#include "std_include.h"
#include <cuda_runtime.h>
#include "curand_kernel.h"

/*! \file cubicLattice.cuh */

/** @addtogroup modelKernels model Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//! move spins
bool gpu_update_spins(dVec *d_disp,
                      dVec *d_pos,
                      scalar scale,
                      int N,
                      bool normalize);

//! set spins to be random points on d-sphere
bool gpu_set_random_spins(dVec *d_pos,
                          curandState *rngs,
                          int blockSize,
                          int nBlocks,
                          int N
                          );

//! set 5-d spins to be random nematic Q tensors with a given amplitude
bool gpu_set_random_nematic_qTensors(dVec *d_pos,
                        int *d_types,
                        curandState *rngs,
                        scalar amplitude,
                        int blockSize,
                        int nBlocks,
                        bool globallyAligned,
                        scalar theta,
                        scalar phi,
                        int N
                        );

/** @} */ //end of group declaration
#endif
