#ifndef qTensorLatticeModel2D_CUH
#define qTensorLatticeModel2D_CUH

#include "std_include.h"
#include "periodicBoundaryConditions.h"
#include <cuda_runtime.h>
#include "curand_kernel.h"

/*! \file qTensorLatticeModel2D.cuh */

/** @addtogroup modelKernels model Kernels
 * @{
 * \brief CUDA kernels and callers for model classes
 */

//!move a qTensor, keep the components within the allowed range
bool gpu_update_2DqTensor(dVec *d_disp, dVec *Q, int N, int blockSize);

//!move a qTensor by a scaled amount, keeping the components within the allowed range
bool gpu_update_2DqTensor(dVec *d_disp, dVec *Q, scalar scale, int N, int blockSize);

//!GPU analog of function in cpp file
bool gpu_get_2DqTensor_DefectMeasures(dVec *Q, scalar *defects, int *t, int defectType, int N);

//! set 5-d spins to be random nematic Q tensors with a given amplitude
bool gpu_set_random_nematic_2DqTensors(dVec *d_pos, int *d_types, curandState *rngs, scalar amplitude, int blockSize,
                                            int nBlocks, bool globallyAligned, scalar phi, int N);



/** @} */ //end of group declaration
#endif
