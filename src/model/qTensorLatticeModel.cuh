#ifndef qTensorLatticeModel_CUH
#define qTensorLatticeModel_CUH

#include "std_include.h"
#include "periodicBoundaryConditions.h"
#include <cuda_runtime.h>
#include "curand_kernel.h"

/*! \file qTensorLatticeModel.cuh */

/** @addtogroup modelKernels model Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!move a qTensor, project it back to a traceless configuration
bool gpu_update_qTensor(dVec *d_disp,
                        dVec *Q,
                        int N);

//!GPU analog of function in cpp file
bool gpu_get_qtensor_DefectMeasures(dVec *Q,
                                    scalar *defects,
                                    int *t,
                                    int defectType,
                                    int N);

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
