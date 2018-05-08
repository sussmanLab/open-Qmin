#include "neighborList.cuh"
/*! \file neighborList.cu */

/*!
    \addtogroup utilityKernels
    @{
*/

/*!
  compute a neighbor list with one thread for each cell to scan for every particle
  (i.e, # threads = N_cells*adjacentCellsPerCell
*/
__global__ void gpu_compute_neighbor_list_TPC_kernel(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *d_pt,
                               int *d_assist,
                               int *d_adj,
                               periodicBoundaryConditions Box,
                               Index2D neighborIndexer,
                               Index2D cellListIndexer,
                               IndexDD cellIndexer,
                               Index2D adjacentCellIndexer,
                               int adjacentCellsPerCell,
                               iVec gridCellsPerSide,
                               dVec gridCellSizes,
                               scalar maxRange,
                               int nmax,
                               int Np)
    {
    // read in the index of this thread
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int particleIdx = tidx / adjacentCellsPerCell;
    if (particleIdx >= Np)
        return;
    int cellIdx = tidx%adjacentCellsPerCell;
    dVec target = d_pt[particleIdx];
    //positionToCellIndex(target)
    iVec cellIndexVec;
    for (int dd =0; dd < DIMENSION; ++dd)
        cellIndexVec.x[dd] = max(0,min((int)gridCellsPerSide.x[dd]-1,(int) floor(target.x[dd]/gridCellSizes.x[dd])));
    int cell = cellIndexer(cellIndexVec);

    //iterate through the given cell
    int currentCell = d_adj[adjacentCellIndexer(cellIdx,cell)];
    int particlesInBin = particlesPerCell[currentCell];
    for(int p1 = 0; p1 < particlesInBin; ++p1)
        {
        int neighborIndex = indices[cellListIndexer(p1,currentCell)];
        if (neighborIndex == particleIdx) continue;
        dVec disp;
        Box.minDist(target,d_pt[neighborIndex],disp);
        if(norm(disp)>=maxRange) continue;
        int offset = atomicAdd(&(d_npp[particleIdx]),1);
        if(offset < d_assist[0])
            {
            int nlpos = neighborIndexer(offset,particleIdx);
            d_idx[nlpos] = neighborIndex;
            d_vec[nlpos] = disp;
            }
        else
            {
            d_assist[0] += 1;
            d_assist[1] = 1;
            }
        };
    };



/*!
compute a neighbor list with one thread per particle
*/
__global__ void gpu_compute_neighbor_list_kernel(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *d_pt,
                               int *d_assist,
                               int *d_adj,
                               periodicBoundaryConditions Box,
                               Index2D neighborIndexer,
                               Index2D cellListIndexer,
                               IndexDD cellIndexer,
                               Index2D adjacentCellIndexer,
                               int adjacentCellsPerCell,
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
    for (int cc = 0; cc < adjacentCellsPerCell; ++cc)
        {
        int currentCell = d_adj[adjacentCellIndexer(cc,cell)];
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
                               int *d_adj,
                               periodicBoundaryConditions &Box,
                               Index2D neighborIndexer,
                               Index2D cellListIndexer,
                               IndexDD cellIndexer,
                               Index2D adjacentCellIndexer,
                               int adjacentCellsPerCell,
                               iVec gridCellsPerSide,
                               dVec gridCellSizes,
                               scalar maxRange,
                               int nmax,
                               int Np,
                               bool threadPerCell)
    {
    if(!threadPerCell)
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
            d_adj,
            Box,
            neighborIndexer,
            cellListIndexer,
            cellIndexer,
            adjacentCellIndexer,
            adjacentCellsPerCell,
            gridCellsPerSide,
            gridCellSizes,
            maxRange,
            nmax,
            Np);

        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        }
    else
        {
        //optimize block size later
        unsigned int max_block = 512;
        unsigned int block_size = max_block;
        if (Np*adjacentCellsPerCell < max_block) block_size = 16;
        unsigned int nblocks  = (adjacentCellsPerCell*Np)/block_size + 1;
        gpu_compute_neighbor_list_TPC_kernel<<<nblocks, block_size>>>(d_idx,
            d_npp,
            d_vec,
            particlesPerCell,
            indices,
            d_pt,
            d_assist,
            d_adj,
            Box,
            neighborIndexer,
            cellListIndexer,
            cellIndexer,
            adjacentCellIndexer,
            adjacentCellsPerCell,
            gridCellsPerSide,
            gridCellSizes,
            maxRange,
            nmax,
            Np);

        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        };
    };

/** @} */ //end of group declaration
