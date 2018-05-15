#ifndef hyperrectangularCellList_H
#define hyperrectangularCellList_H

#include "periodicBoundaryConditions.h"
#include "gpuarray.h"
#include "indexer.h"
#include "std_include.h"


/*! \file orthorhombicCellList.h */
//! Construct simple cell/bucket structures on the CPU or GPU given a hyper-rectangular domain with PBCs
//
/*!
 * A class that can sort points into a grid of buckets. This enables local searches for particle neighbors, etc.
 * Note that at the moment this class can only handle hyper-rectangular boxes.
 */
class hyperrectangularCellList
    {
    public:
        //!Blank constructor
        hyperrectangularCellList(){Nmax=0;Box = make_shared<periodicBoundaryConditions>();};
        //! a box, and a size for the underlying grid
        hyperrectangularCellList(scalar a, BoxPtr _box);

        //!Set the BoxPtr to point to an existing one
        void setBox(BoxPtr bx){Box=bx;};
        //!call setGridSize if the particles and box already set, as this doubles as a general initialization of data structures
        void setGridSize(scalar a);
        //!Get an upper bound on the maximum number of particles in a given bucket
        int getNmax() {return Nmax;};
        //!The number of cells in each direction
        iVec getBinsPerSide() {return gridCellsPerSide;};
        //!Returns the length of the square that forms the base grid size
        dVec getCellSize() {return gridCellSizes;};

        //!Return the index of the cell that contains the given point
        int positionToCellIndex(const dVec &pos);

        //!return the iVec corresponding to a cell index
        iVec indexToiVec(const int cellIndex){return cellIndexer.inverseIndex(cellIndex);};

        //! given a target cell and a width, get all cell indices that sit in the surrounding square
        void getCellNeighbors(int cellIndex, int width, vector<int> &cellNeighbors);

        //!Initialization and helper without using the GPU
        void resetCellSizesCPU();
        //!Initialization and helper
        void resetCellSizes();
        //!Return the array of particles per cell
        const GPUArray<unsigned int>& getCellSizeArray() const
            {
            return elementsPerCell;
            };
        //!Return the array of cell indices in the different cells
        const GPUArray<int>& getIdxArray() const
            {
            return particleIndices;
            };

        //! compute the cell list of the gpuarry passed to it.
        void computeCellList(GPUArray<dVec> &points)
            {
            if(!adjCellsComputed)
                computeAdjacentCells();
            if(useGPU)
                computeGPU(points);
            else
                computeCPU(points);
            };
        //! compute the cell list of the gpuarry passed to it. GPU function
        void computeGPU(GPUArray<dVec> &points);
        //! compute the cell list of the gpuarry passed to it. CPU function
        void computeCPU(GPUArray<dVec> &points);


        //! Indexes the cells in the grid themselves (so the index of the bin corresponding to (iVec) is bin = cellIndexer(iVec)
        IndexDD cellIndexer;
        //!Indexes elements in the cell list (i.e., the third element of cell index i is cellListIndexer(3,i)
        Index2D cellListIndexer;

        //! An array containing the number of elements in each cell
        GPUArray<unsigned int> elementsPerCell;
        //!An array containing the indices of particles in various cells. So, particleIndices[cellListIndexer(nn,bin)] gives the index of the nth particle in the bin "bin" of the cell list
        GPUArray<int> particleIndices;
        //!An array containing the positions of particles in various cells. Aligned with the particleIndices array, so that derived methods can choose whatever is most convenient,so particlePositions[cellListIndexer(nn,bin)] gives the positionof the nth particle in the bin "bin" of the cell list
        GPUArray<dVec> particlePositions;

        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU=true){useGPU = _useGPU;};

        //!A reporter index of how hard it was to compute the cell list
        int computations;

        //!return the list of adjacent cells
        GPUArray<int> & returnAdjacentCells(){return adjacentCells;};
        //!compute the adjacent cells up to a scale
        void computeAdjacentCells(int width = 1);
        //!the number of cells in the adjacency list for each cell
        int adjacentCellsPerCell;
        //! indexes the adjacent cells of each cell
        Index2D adjacentCellIndexer;

    protected:
        //!first index is Nmax, second is whether to recompute
        GPUArray<int> assist;

        //!an array that stores the list cell adjacent to every cell
        GPUArray<int> adjacentCells;

        //!The number of bins in each of the different dimensions
        iVec gridCellsPerSide;
        //!The size of each cell (e.g., each cell is a small hyperrectangle of these dimensions
        dVec gridCellSizes;

        int totalCells;
        //! the maximum number of particles found in any bin
        int Nmax;
        //!The Box used to compute periodic distances
        BoxPtr Box;
        //!whether the updater does its work on the GPU or not
        bool useGPU;

        //!have we already computed the adjacent cell lists?
        bool adjCellsComputed;
    };

#endif
