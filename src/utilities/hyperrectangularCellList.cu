#define ENABLE_CUDA
#include "hyperrectangularCellList.cuh"
#include "indexer.h"
#include "periodicBoundaryConditions.h"
/*! \file hyperrectangularCellList.cu */

/*!
    \addtogroup cellListGPUKernels
    @{
*/

/*!
  Assign particles to bins, keep track of the number of particles per bin, etc.
  */
__global__ void gpu_compute_cell_list_kernel(dVec *d_pt,
                                              unsigned int *d_elementsPerCell,
                                              int *d_particleIndices,
                                              int Np,
                                              unsigned int Nmax,
                                              iVec gridCellsPerSide,
                                              dVec gridCellSizes,
                                              BoxPtr Box,
                                              IndexDD cellIndexer,
                                              Index2D cellListIndexer,
                                              int *d_assist
                                              )
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Np)
        return;

    dVec pos = d_pt[idx];
    iVec bin;
    for (int dd = 0; dd < DIMENSION; ++dd)
        bin.x[dd] = floor(pos.x[dd] / gridCellSizes.x[dd]);
    int binIndex = cellIndexer(bin);
    unsigned int offset = atomicAdd(&(d_elementsPerCell[binIndex]), 1);
    if(offset <= d_assist[0]+1)
        {
        unsigned int write_pos = min(cellListIndexer(offset,binIndex),cellListIndexer.getNumElements()-1);
        d_particleIndices[write_pos] = idx;
        }
    else
        {
        d_assist[0] = offset+1;
        d_assist[1] = 1;
        };
    };

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
                                  )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (Np < 128) block_size = 16;
    unsigned int nblocks  = Np/block_size + 1;


    unsigned int nmax = (unsigned int) Nmax;
    gpu_compute_cell_list_kernel<<<nblocks, block_size>>>(d_pt,
                                                          d_cell_sizes,
                                                          d_idx,
                                                          Np,
                                                          nmax,
                                                          gridCellsPerSide,
                                                          gridCellSizes,
                                                          Box,
                                                          ci,
                                                          cli,
                                                          d_assist
                                                          );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
