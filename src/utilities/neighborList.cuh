#ifndef __neighborList_CUH__
#define __neighborList_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "periodicBoundaryConditions.h"

/*! \file neighborList.cuh
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers
 */

bool gpu_compute_neighbor_list(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *d_pt,
                               int *d_assist,
                               periodicBoundaryConditions &Box,
                               Index2D neighborIndexer,
                               Index2D cellListIndexer,
                               IndexDD cellIndexer,
                               iVec gridCellsPerSide,
                               dVec gridCellSizes,
                               scalar maxRange,
                               int nmax,
                               int Np);

/** @} */ //end of group declaration
#endif
