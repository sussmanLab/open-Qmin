#include "hyperrectangularCellList.h"
#include "hyperrectangularCellList.cuh"
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
    Nmax = 0;
    Box = make_shared<periodicBoundaryConditions>();
    setBox(_box);
    setGridSize(a);
    }

/*!
\param bx the box defining the periodic unit cell
 */
void hyperrectangularCellList::setBox(BoxPtr _box)
    {
    dVec bDims;
    _box->getBoxDims(bDims);
    Box->setBoxDims(bDims);
    };

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
        gridCellsPerSide.x[dd] = (unsigned int)floor(bDims.x[dd]);
        if(gridCellsPerSide.x[dd]%2==1) gridCellsPerSide.x[dd]+=1;
        totalCells *= gridCellsPerSide.x[dd];
        gridCellSizes = bDims.x[dd]/gridCellsPerSide.x[dd];
        };

    elementsPerCell.resize(totalCells); //number of elements in each cell...initialize to zero

    cellIndexer = IndexDD(gridCellsPerSide);

    Nmax = 1;
    resetCellSizesCPU();
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
        particleIndices.resize(cellListIndexer.getNumElements());

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
    //set all cell sizes to zero
    if(elementsPerCell.getNumElements() != totalCells)
        elementsPerCell.resize(totalCells);

    ArrayHandle<unsigned int> d_elementsPerCell(elementsPerCell,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_elementsPerCell.data,totalCells);

    //set all cell indexes to zero
    cellListIndexer = Index2D(Nmax,totalCells);
    if(particleIndices.getNumElements() != cellListIndexer.getNumElements())
        particleIndices.resize(cellListIndexer.getNumElements());

    ArrayHandle<int> d_idx(particleIndices,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_idx.data,(int) cellListIndexer.getNumElements());


    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
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
    cellNeighbors.reserve(idPow(w));
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
    ArrayHandle<dVec> h_pt(points,access_location::host,access_mode::read);
    iVec bin;
    int Np = points.getNumElements();
    if(Nmax == 1)
        {
        Nmax = ceil(Np/totalCells);
        };
    int nmax = Nmax;
    computations = 0;
    while (recompute)
        {
        //reset particles per cell, reset cellListIndexer, resize particleIndices
        resetCellSizesCPU();
        ArrayHandle<unsigned int> h_elementsPerCell(elementsPerCell,access_location::host,access_mode::readwrite);
        ArrayHandle<int> h_idx(particleIndices,access_location::host,access_mode::readwrite);
        recompute=false;

        for (int nn = 0; nn < Np; ++nn)
            {
            //get the correct cell of the current particle
            for (int dd = 0; dd < DIMENSION; ++dd)
                bin.x[dd] = floor(h_pt.data[nn].x[dd] / gridCellSizes.x[dd]);

            int binIndex = cellIndexer(bin);
            int offset = h_elementsPerCell.data[binIndex];
            if (offset < Nmax)
                {
                int clpos = cellListIndexer(offset,binIndex);
                h_idx.data[cellListIndexer(offset,binIndex)]=nn;
                }
            else
                {
                nmax = max(Nmax,offset+1);
                Nmax=nmax;
                recompute=true;
                };
            h_elementsPerCell.data[binIndex]++;
            };
        computations++;
        };
    cellListIndexer = Index2D(Nmax,totalCells);
    };


/*!
\param points the set of points to assign to cells...on the GPU
 */
void hyperrectangularCellList::computeGPU(GPUArray<dVec> &points)
    {
    bool recompute = true;
    int Np = points.getNumElements();
    if(Nmax == 1)
        {
        Nmax = ceil(Np/totalCells);
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
            ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);

            //call the gpu function
            gpu_compute_cell_list(d_pt.data,        //particle positions...broken
                          d_elementsPerCell.data,//particles per cell
                          d_idx.data,       //cell list
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
        };
    cellListIndexer = Index2D(Nmax,totalCells);
    };
