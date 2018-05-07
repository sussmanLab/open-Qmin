#include "neighborList.cuh"
/*! \file neighborList.cu */

/*!
    \addtogroup utilityKernels
    @{
*/

__global__ void gpu_compute_neighbor_list_kernel(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *d_pt,
                               int *d_assist,
                               periodicBoundaryConditions Box,
                               Index2D neighborIndexer,
                               Index2D cellListIndexer,
                               IndexDD cellIndexer,
                               iVec gridCellsPerSide,
                               dVec gridCellSizes,
                               scalar maxRange,
                               int nmax,
                               int Np)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Np)
        return;
    dVec target = d_pt[idx];
    //positionToCellIndex(target)
    iVec cellIndexVec;
    for (int dd =0; dd < DIMENSION; ++dd)
        cellIndexVec.x[dd] = max(0,min((int)gridCellsPerSide.x[dd]-1,(int) floor(target.x[dd]/gridCellSizes.x[dd])));
    int cell = cellIndexer(cellIndexVec);
    //iterate through neighboring cells
    iVec min(-1);
    iVec max(1);
    iVec it(-1); it.x[0]-=1;
    while(iVecIterate(it,min,max))
        {
        int currentCell = cellIndexer(modularAddition(cellIndexVec,it,gridCellsPerSide));
        int particlesInBin =  particlesPerCell[currentCell];
        for(int p1 = 0; p1 < particlesInBin; ++p1)
            {
            int neighborIndex = indices[cellListIndexer(p1,currentCell)];
            if (neighborIndex == idx) continue;
            dVec disp;
            Box.minDist(target,d_pt[neighborIndex],disp);
            if(norm(disp)>=maxRange) continue;

            int offset = d_npp[idx];
            if(offset < d_assist[0])
                {
                int nlpos = neighborIndexer(offset,idx);
                d_idx[nlpos] = neighborIndex;
                d_vec[nlpos] = disp;
                }
            else
                {
                //atomicAdd(&(d_assist[0]),1);
                d_assist[0] += 1;
                d_assist[1] = 1;
                };
            d_npp[idx]+=1;
            };
        };
    };

/*!
  compute neighbor list, one particle per thread
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
                               int Np)
    {
    //optimize block size later
    unsigned int block_size = 64;
    if (Np < 64) block_size = 16;
    unsigned int nblocks  = Np/block_size + 1;
    gpu_compute_neighbor_list_kernel<<<nblocks, block_size>>>(d_idx,
            d_npp,
            d_vec,
            particlesPerCell,
            indices,
            d_pt,
            d_assist,
            Box,
            neighborIndexer,
            cellListIndexer,
            cellIndexer,
            gridCellsPerSide,
            gridCellSizes,
            maxRange,
            nmax,
            Np);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
