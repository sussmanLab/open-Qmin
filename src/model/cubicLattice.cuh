#ifndef cubicLattice_CUH
#define cubicLattice_CUH

#include "std_include.h"
#include <cuda_runtime.h>
#include "curand_kernel.h"
#include "indexer.h"

/*! \file cubicLattice.cuh */

/** @addtogroup modelKernels model Kernels
 * @{
 * \brief CUDA kernels and callers for model classes
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
//!copy a boundary or surface to an assist array
bool gpu_copy_boundary_object(dVec *pos,
                              int *sites,
                              int *neighbors,
                              pair<int,dVec> *assistStructure,
                              int *types,
                              Index2D neighborIndex,
                              int motionDirection,
                              bool resetLattice,
                              int Nsites);

//!Move a boundary or surface via an assist structure
bool gpu_move_boundary_object(dVec *pos,
                              int *sites,
                              pair<int,dVec> *assistStructure,
                              int *types,
                              int newTypeValue,
                              int Nsites);

/** @} */ //end of group declaration
#endif
