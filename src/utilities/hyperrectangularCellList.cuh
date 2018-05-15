#ifndef __hyperRectangularCELL_CUH__
#define __hyperRectangularCELL_CUH__

#include "std_include.h"
#include <cuda_runtime.h>
#include "indexer.h"
#include "periodicBoundaryConditions.h"

/*! \file hyperrectangularCellList.cuh
*/

/** @defgroup utilityKernels utility Kernels
 * @{
 * \brief CUDA kernels and callers for the utilities base
 */

//!Find the set indices of points in every cell bucket in the grid
bool gpu_compute_cell_list(dVec *d_pt,
                                  unsigned int *d_cell_sizes,
                                  int *d_idx,
                                  dVec *d_cellParticlePos,
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
