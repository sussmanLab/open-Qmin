#include "hyperrectangularCellList.h"
#include "hyperrectangularCellList.cuh"
#include "utilities.cuh"
#include "cuda_runtime.h"

/*! \file hyperrectangularCellList.cpp */

/*!
\param a the approximate side length of the cells
\param points the positions of points to populate the cell list with
\param bx the period box for the system
 */
hyperrectangularCellList::hyperrectangularCellList(scalar a, BoxPtr _box)
    {
    useGPU = false;
    Nmax = 2;
    Box = _box;
    setGridSize(a);
    }

/*!
\param a the approximate side length of all of the cells.
This routine currently picks an even integer of cells in each dimension, close to the desired size, that fit in the box.
 */
void hyperrectangularCellList::setGridSize(scalar a)
    {
    dVec bDims;
    Box->getBoxDims(bDims);

    totalCells = 1;
    for (int dd = 0; dd < DIMENSION; ++dd)
        {
        gridCellsPerSide.x[dd] = (unsigned int)floor(bDims.x[dd]/a);
        if(gridCellsPerSide.x[dd]%2==1) gridCellsPerSide.x[dd]-=1;
        totalCells *= gridCellsPerSide.x[dd];
        gridCellSizes = bDims.x[dd]/gridCellsPerSide.x[dd];
        };

    elementsPerCell.resize(totalCells); //number of elements in each cell...initialize to zero

    cellIndexer = IndexDD(gridCellsPerSide);
    Nmax = 2;
    cellListIndexer = Index2D(Nmax,totalCells);
    resetCellSizesCPU();
    adjCellsComputed = false;
    };

/*!
Sets all cell sizes to zero, all cell indices to zero, and resets the "assist" utility structure,
all on the CPU (so that no expensive copies are needed)
 */
void hyperrectangularCellList::resetCellSizesCPU()
    {
    //set all cell sizes to zero
    if(elementsPerCell.getNumElements() != totalCells)
        elementsPerCell.resize(totalCells);

    ArrayHandle<unsigned int> h_elementsPerCell(elementsPerCell,access_location::host,access_mode::overwrite);
    for (int i = 0; i <totalCells; ++i)
        h_elementsPerCell.data[i]=0;

    //set all cell indexes to zero
    cellListIndexer = Index2D(Nmax,totalCells);
    if(particleIndices.getNumElements() != cellListIndexer.getNumElements())
        {
        particleIndices.resize(cellListIndexer.getNumElements());
        particlePositions.resize(cellListIndexer.getNumElements());
        }

    ArrayHandle<int> h_idx(particleIndices,access_location::host,access_mode::overwrite);
    for (int i = 0; i < cellListIndexer.getNumElements(); ++i)
        h_idx.data[i]=0;

    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
    };


/*!
Sets all cell sizes to zero, all cell indices to zero, and resets the "assist" utility structure,
all on the GPU so that arrays don't need to be copied back to the host
*/
void hyperrectangularCellList::resetCellSizes()
    {
NVTXPUSH("cell list resetting");
    //set all cell sizes to zero
    if(elementsPerCell.getNumElements() != totalCells)
        elementsPerCell.resize(totalCells);

    ArrayHandle<unsigned int> d_elementsPerCell(elementsPerCell,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_elementsPerCell.data,totalCells);

    //set all cell indexes to zero
    cellListIndexer = Index2D(Nmax,totalCells);
    if(particleIndices.getNumElements() != cellListIndexer.getNumElements())
        {
        particleIndices.resize(cellListIndexer.getNumElements());
        particlePositions.resize(cellListIndexer.getNumElements());
        };

    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
NVTXPOP();
    };

/*!
\param pos the dVec coordinate of the position
returns the cell index that (pos) would be contained in for the current cell list
 */
int hyperrectangularCellList::positionToCellIndex(const dVec &pos)
    {
    iVec cellIndexVec;
    for (int dd = 0; dd < DIMENSION;++dd)
        cellIndexVec.x[dd] = max(0,min((int)gridCellsPerSide.x[dd]-1,(int) floor(pos.x[dd]/gridCellSizes.x[dd])));
    return cellIndexer(cellIndexVec);
    };

/*!
\param cellIndex the base cell index to find the neighbors of
\param width the distance (in cells) to search
\param cellNeighbors a vector of all cell indices that are neighbors of cellIndex
 */
void hyperrectangularCellList::getCellNeighbors(int cellIndex, int width, std::vector<int> &cellNeighbors)
    {
    int w = min(width,(int)gridCellsPerSide.x[0]);
    iVec cellIndexVec = cellIndexer.inverseIndex(cellIndex);

    cellNeighbors.clear();
    cellNeighbors.reserve(idPow(2*w+1));
    iVec min(-w);
    iVec max(w);
    iVec it(-w);it.x[0]-=1;
    while(iVecIterate(it,min,max))
        {

        cellNeighbors.push_back(cellIndexer(modularAddition(cellIndexVec,it,gridCellsPerSide)));
        };
    };

/*!
\param points the set of points to assign to cells
 */
void hyperrectangularCellList::computeCPU(GPUArray<dVec> &points)
    {
    //will loop through particles and put them in cells...
    //if there are more than Nmax particles in any cell, will need to recompute.
    bool recompute = true;
    bool reset = false;
    ArrayHandle<dVec> h_pt(points,access_location::host,access_mode::read);
    iVec bin;
    int Np = points.getNumElements();
    resetCellSizesCPU();
    int nmax = Nmax;
    computations = 0;
    while (recompute)
        {
        //reset particles per cell, reset cellListIndexer, resize particleIndices
        if(reset) resetCellSizesCPU();
        {
        ArrayHandle<unsigned int> h_elementsPerCell(elementsPerCell,access_location::host,access_mode::readwrite);
        ArrayHandle<int> h_idx(particleIndices,access_location::host,access_mode::readwrite);
        ArrayHandle<dVec> h_cellParticlePos(particlePositions,access_location::host,access_mode::readwrite);
        recompute=false;

        for (int nn = 0; nn < Np; ++nn)
            {
            //get the correct cell of the current particle
            for (int dd = 0; dd < DIMENSION; ++dd)
                bin.x[dd] = floor(h_pt.data[nn].x[dd] / gridCellSizes.x[dd]);

            int binIndex = cellIndexer(bin);
            int offset = h_elementsPerCell.data[binIndex];
            if (offset < Nmax && !recompute)
                {
                int clpos = cellListIndexer(offset,binIndex);
                h_idx.data[clpos]=nn;
                h_cellParticlePos.data[clpos] = h_pt.data[nn];
                }
            else
                {
                nmax = max(Nmax,offset+1);
                Nmax=nmax;
                recompute=true;
                reset=true;
                };
            h_elementsPerCell.data[binIndex]++;
            };
        computations++;
        };
        };
    cellListIndexer = Index2D(Nmax,totalCells);
    };

/*!
 *width*gridsize should be a cutoff scale in the problem... this will evaluate all cells that might
 containt a neighbor within gridsize*width; *not* all cells in a hypercube of sidelength width
 */
void hyperrectangularCellList::computeAdjacentCells(int width)
    {
    //compute the number of adjacent cells per cell:
    int neighs = 0;
    vector<iVec> mask;
    {
    iVec min(-width);
    iVec max(width);
    iVec it(-width);it.x[0]-=1;
    while(iVecIterate(it,min,max))
        {
        iVec gridPosition = it;
        scalar rmin = 0.0;
        for (int dd = 0; dd < DIMENSION; ++dd)
            {
            scalar temp = gridCellSizes.x[dd]*(std::max(abs(gridPosition.x[dd])-1.,0.0));
            rmin += temp*temp;
            }
        rmin = sqrt(rmin);
        if(rmin < gridCellSizes.x[0]*width+1e-6)
            {
            neighs += 1;
            mask.push_back(it);
            }
        };
    }
    adjacentCellsPerCell = neighs;
    cout << "building a cell list with "<< adjacentCellsPerCell << " adj cells per cell" << endl;

    if(adjacentCellsPerCell == adjacentCells.getNumElements()) return;

    adjacentCells.resize(totalCells*adjacentCellsPerCell);
    adjacentCellIndexer = Index2D(adjacentCellsPerCell,totalCells);
    ArrayHandle<int> adj(adjacentCells);
    iVec min(-width);
    iVec max(width);
    for (int cellIndex = 0; cellIndex < totalCells; ++cellIndex)
        {
        iVec cellIndexVec = cellIndexer.inverseIndex(cellIndex);
        for (int cc = 0; cc < mask.size();++cc)
            {
            int location = adjacentCellIndexer(cc,cellIndex);
            adj.data[location] = cellIndexer(modularAddition(cellIndexVec,mask[cc],gridCellsPerSide));
            }
        };
    adjCellsComputed = true;
    };

/*!
\param points the set of points to assign to cells...on the GPU
 */
void hyperrectangularCellList::computeGPU(GPUArray<dVec> &points)
    {
    bool recompute = true;
    int Np = points.getNumElements();
    if(Nmax <2)
        {
        Nmax = 2;
        };

    while (recompute)
        {
        //cout << "computing cell list on the gpu with Nmax = " << Nmax << endl;
        resetCellSizes();
        //scope for arrayhandles
        if (true)
            {
            //get particle data
            ArrayHandle<dVec> d_pt(points,access_location::device,access_mode::read);

            //get cell list arrays...readwrite so things are properly zeroed out
            ArrayHandle<unsigned int> d_elementsPerCell(elementsPerCell,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_idx(particleIndices,access_location::device,access_mode::readwrite);
            ArrayHandle<dVec> d_cellParticlePos(particlePositions,access_location::device,access_mode::readwrite);
            ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);

            //call the gpu function
            gpu_compute_cell_list(d_pt.data,        //particle positions...
                          d_elementsPerCell.data,//particles per cell
                          d_idx.data,       //cell list
                          d_cellParticlePos.data,       //cell list particle positions
                          Np,               //number of particles
                          Nmax,             //maximum particles per cell
                          gridCellsPerSide, //number of cells in each direction
                          gridCellSizes,    //size of cells in each direction
                          Box,
                          cellIndexer,
                          cellListIndexer,
                          d_assist.data
                          );
            }
        //get cell list arrays
        recompute = false;
NVTXPUSH("cell list thing");
        if (true)
            {
            ArrayHandle<unsigned int> h_elementsPerCell(elementsPerCell,access_location::host,access_mode::read);
            for (int cc = 0; cc < totalCells; ++cc)
                {
                int cs = h_elementsPerCell.data[cc] ;
                if(cs > Nmax)
                    {
                    Nmax =cs ;
                    if (Nmax%2 == 0 ) Nmax +=2;
                    if (Nmax%2 == 1 ) Nmax +=1;
                    recompute = true;
                    };
                };
            };
NVTXPOP();
        };
    cellListIndexer = Index2D(Nmax,totalCells);
    };
