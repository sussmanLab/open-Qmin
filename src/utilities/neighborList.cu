#include "neighborList.cuh"
/*! \file neighborList.cu */

/*!
    \addtogroup utilityKernels
    @{
*/

/*!
  compute a neighbor list with some value of threads per particle
  (i.e, # threads = N_cells*adjacentCellsPerCell
*/
__global__ void gpu_compute_neighbor_list_TPP_kernel(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *cellParticlePos,
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
                               int cellListNmax,
                               scalar maxRange,
                               int nmax,
                               int Np,
                               int threadsPerParticle)
    {
    // read in the index of this thread
    unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int particleIdx = tidx / threadsPerParticle;
    if (particleIdx >= Np)
        return;
    int workIdx = tidx % threadsPerParticle;

    dVec target = d_pt[particleIdx];
    //positionToCellIndex(target)
    iVec cellIndexVec;
    for (int dd =0; dd < DIMENSION; ++dd)
        cellIndexVec.x[dd] = max(0,min((int)gridCellsPerSide.x[dd]-1,(int) floor(target.x[dd]/gridCellSizes.x[dd])));
    //the cell index of the target particle
    int cell = cellIndexer(cellIndexVec);

    //work per thread is integer division, rounding up to guarantee all cells are fully checked
    int workPerParticleThread = (adjacentCellsPerCell*cellListNmax + threadsPerParticle-1) / threadsPerParticle;
    //start the loop with the actual work
    int cellToScan = workIdx*workPerParticleThread / cellListNmax;
    int cellIndexToScan = workIdx*workPerParticleThread % cellListNmax;
    if(cellToScan >=adjacentCellsPerCell)
        return;
    int currentCell = d_adj[adjacentCellIndexer(cellToScan,cell)];
    int particlesInBin = particlesPerCell[currentCell];

    for (int ww = 0; ww < workPerParticleThread; ++ww)
        {
        if(cellIndexToScan < particlesInBin)
            {
            int cellListIdx = cellListIndexer(cellIndexToScan,currentCell);
            int neighborIndex = indices[cellListIdx];
            if (neighborIndex != particleIdx)
                {
                dVec otherParticle = cellParticlePos[cellListIdx];
                dVec disp;
                Box.minDist(target,otherParticle,disp);
                if(dot(disp,disp) < maxRange*maxRange)
                    {
                    int offset = atomicAdd(&(d_npp[particleIdx]),1);
                    if(offset<nmax)
                        {
                        int nlpos = neighborIndexer(offset,particleIdx);
                        d_idx[nlpos] = neighborIndex;
                        d_vec[nlpos] = disp;
                        }
                    else
                        {
                        //atomicAdd(&(d_assist[0]),1);
                        atomicCAS(&(d_assist)[0],offset,offset+1);
                        d_assist[1]=1;
                        }
                    }
                }
            }
        cellIndexToScan += 1;
        if(cellIndexToScan >=cellListNmax)
            {
            cellIndexToScan = 0;
            cellToScan += 1;
            if (cellToScan >= adjacentCellsPerCell)
                break;
            currentCell = d_adj[adjacentCellIndexer(cellToScan,cell)];
            particlesInBin = particlesPerCell[currentCell];
            }
        }
    };
/*!
  compute a neighbor list with one thread for each cell to scan for every particle
  (i.e, # threads = N_cells*adjacentCellsPerCell
*/
__global__ void gpu_compute_neighbor_list_TPC_kernel(int *d_idx,
                               unsigned int *d_npp,
                               dVec *d_vec,
                               unsigned int *particlesPerCell,
                               int *indices,
                               dVec *cellParticlePos,
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
                               scalar maxRange2,
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
    dVec disp;
    for(int p1 = 0; p1 < particlesInBin; ++p1)
        {
        int cellListIdx = cellListIndexer(p1,currentCell);
        int neighborIndex = indices[cellListIdx];
        if (neighborIndex == particleIdx) continue;
        dVec otherParticle = cellParticlePos[cellListIndexer(p1,currentCell)];
        Box.minDist(target,otherParticle,disp);
        if(dot(disp,disp)>=maxRange2) continue;
        int offset = atomicAdd(&(d_npp[particleIdx]),1);
        if(offset<nmax)
            {
            int nlpos = neighborIndexer(offset,particleIdx);
            d_idx[nlpos] = neighborIndex;
            d_vec[nlpos] = disp;
            }
        else
            {
            atomicCAS(&(d_assist)[0],offset,offset+1);
            d_assist[1]=1;
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
                               scalar maxRange2,
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
            if(dot(disp,disp)>=maxRange2) continue;

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
                               dVec *cellParticlePos,
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
                               int cellListNmax,
                               scalar maxRange,
                               int nmax,
                               int Np,
                               int maxBlockSize,
                               bool threadPerCell)
    {
    /*
       //testing varying thread per particle stuff
    unsigned int block_size = 128;
    int threadsPerParticle = maxBlockSize;
    unsigned int nblocks = (threadsPerParticle*Np)/block_size+1;
    dim3 blocks(nblocks,1,1);
    dim3 grids(block_size,1,1);
        gpu_compute_neighbor_list_TPP_kernel<<<blocks, grids>>>(d_idx,
            d_npp,
            d_vec,
            particlesPerCell,
            indices,
            cellParticlePos,
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
            cellListNmax,
            maxRange,
            nmax,
            Np,
            threadsPerParticle);

        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        */
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = (adjacentCellsPerCell*Np)/block_size+1;

    if(!threadPerCell || nblocks > 2147483647) // max blocks for compute 3.0 and higher
        {
        if (Np < maxBlockSize) block_size = 16;
        nblocks  = Np/block_size + 1;
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
            maxRange*maxRange,
            nmax,
            Np);

        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        }
    else
        {
        if (Np*adjacentCellsPerCell < maxBlockSize) block_size = 16;
        nblocks  = (adjacentCellsPerCell*Np)/block_size + 1;
        gpu_compute_neighbor_list_TPC_kernel<<<nblocks, block_size>>>(d_idx,
            d_npp,
            d_vec,
            particlesPerCell,
            indices,
            cellParticlePos,
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
            maxRange*maxRange,
            nmax,
            Np);

        HANDLE_ERROR(cudaGetLastError());
        return cudaSuccess;
        };
    };

/** @} */ //end of group declaration
