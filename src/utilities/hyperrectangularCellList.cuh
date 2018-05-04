#ifndef __GPUCELL_CUH__
#define __GPUCELL_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "periodicBoundaryConditions.h"

/*! \file hyperrectangularCellList.cuh
*/

/** @defgroup cellListGPUKernels cellListGPU Kernels
 * @{
 * \brief CUDA kernels and callers for the cellListGPU class
 */

//!Find the set indices of points in every cell bucket in the grid
bool gpu_compute_cell_list(dVec *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  int Np,
                                  int &Nmax,
                                  iVec gridCellsPerSide,
                                  dVec gridCellSizes,
                                  BoxPtr Box,
                                  IndexDD &ci,
                                  Index2D &cli,
                                  int *d_assist
                                  );

/** @} */ //end of group declaration

#endif
