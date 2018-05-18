#include "neighborList.h"
#include "utilities.cuh"
#include "neighborList.cuh"
/*! \file neighborList.cpp */

neighborList::neighborList(scalar range, BoxPtr _box, int subGridReduction)
    {
    useGPU = false;
    saveDistanceData = true;
    Box = _box;
    scalar gridScale = 1./(scalar)subGridReduction;
    int width = subGridReduction;
    cellList = make_shared<hyperrectangularCellList>(range*gridScale,Box);
    cellList->computeAdjacentCells(width);
    Nmax = 4;
    maxRange = range;
    nlistTuner = make_shared<kernelTuner>(16,1024,16,5,200000);
    };

void neighborList::resetNeighborsGPU(int size,int _nmax)
    {
NVTXPUSH("resetting neighbor structures1");
    if(neighborsPerParticle.getNumElements() != size)
        neighborsPerParticle.resize(size);
    ArrayHandle<unsigned int> d_npp(neighborsPerParticle,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_npp.data,size);

    Nmax = _nmax;
    neighborIndexer = Index2D(_nmax,size);
    if(particleIndices.getNumElements() != neighborIndexer.getNumElements())
        {
        particleIndices.resize(neighborIndexer.getNumElements());
        if(saveDistanceData)
            {
            neighborVectors.resize(neighborIndexer.getNumElements());
            neighborDistances.resize(neighborIndexer.getNumElements());
            };
        };

NVTXPOP();
NVTXPUSH("resetting neighbor structures2");
    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=_nmax;
    h_assist.data[1] = 0;
NVTXPOP();
    };

void neighborList::resetNeighborsCPU(int size, int _nmax)
    {
    if(neighborsPerParticle.getNumElements() != size)
        neighborsPerParticle.resize(size);
    ArrayHandle<unsigned int> h_npp(neighborsPerParticle,access_location::host,access_mode::overwrite);
    for (int i = 0; i < size; ++i)
        h_npp.data[i] = 0;

    Nmax = _nmax;
    neighborIndexer = Index2D(_nmax,size);
    if(particleIndices.getNumElements() != neighborIndexer.getNumElements())
        {
        particleIndices.resize(neighborIndexer.getNumElements());
        if(saveDistanceData)
            {
            neighborVectors.resize(neighborIndexer.getNumElements());
            neighborDistances.resize(neighborIndexer.getNumElements());
            };
        };

    ArrayHandle<int> h_idx(particleIndices,access_location::host,access_mode::overwrite);
    ArrayHandle<dVec> h_vec(neighborVectors,access_location::host,access_mode::overwrite);
        ArrayHandle<scalar> h_dist(neighborDistances,access_location::host,access_mode::overwrite);
    for (int i = 0; i < neighborIndexer.getNumElements(); ++i)
        {
        h_idx.data[i]=0;
        if(saveDistanceData)
            {
            h_vec.data[i] = make_dVec(0.);
            h_dist.data[i] = 0.;
            };
        };

    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=_nmax;
    h_assist.data[1] = 0;
    };

/*!
\param points the set of points to find neighbors for
 */
void neighborList::computeCPU(GPUArray<dVec> &points)
    {
    //put the points in a cell list structure
    cellList->computeCellList(points);
    ArrayHandle<unsigned int> particlesPerCell(cellList->elementsPerCell);
    ArrayHandle<int> indices(cellList->particleIndices);

    bool recompute = true;
    int Np = points.getNumElements();
    ArrayHandle<dVec> h_pt(points,access_location::host,access_mode::read);
    iVec bin;
    int nmax = Nmax;

    computations = 0;
    while(recompute)
        {
        computations += 1;
        resetNeighborsCPU(Np,nmax);
        ArrayHandle<unsigned int> h_npp(neighborsPerParticle);
        ArrayHandle<int> h_idx(particleIndices);
        ArrayHandle<dVec> h_vec(neighborVectors);
        ArrayHandle<scalar> h_dist(neighborDistances);
        ArrayHandle<int> h_adj(cellList->returnAdjacentCells());
        recompute = false;
        vector<int> cellsToScan;
        for (int pp = 0; pp < Np; ++pp)
            {
            dVec target = h_pt.data[pp];
            int cell = cellList->positionToCellIndex(target);
            for (int cc = 0; cc < cellList->adjacentCellsPerCell;++cc)
                {
                int currentCell = h_adj.data[cellList->adjacentCellIndexer(cc,cell)];
                int particlesInBin =  particlesPerCell.data[currentCell];
                for (int p1 = 0; p1 < particlesInBin; ++p1)
                    {
                    int neighborIndex = indices.data[cellList->cellListIndexer(p1,currentCell)];
                    if (neighborIndex == pp) continue;
                    dVec disp;
                    Box->minDist(target,h_pt.data[neighborIndex],disp);
                    scalar dist = norm(disp);
                    if(dist>=maxRange) continue;
                    int offset = h_npp.data[pp];
                    if(offset < Nmax && !recompute)
                        {
                        int nlpos = neighborIndexer(offset,pp);
                        h_idx.data[nlpos] = neighborIndex;
                        if(saveDistanceData)
                            {
                            h_vec.data[nlpos] = disp;
                            h_dist.data[nlpos] = dist;
                            }
                        }
                    else
                        {
                        nmax=max(nmax,offset+1);
                        Nmax=nmax;
                        recompute = true;
                        };
                    h_npp.data[pp] += 1;
                    };
                };
            };
        };
        neighborIndexer = Index2D(Nmax,Np);
    };

/*!
\param points the set of points to find neighbors for
 */
void neighborList::computeGPU(GPUArray<dVec> &points)
    {
NVTXPUSH("cell list");
    cellList->computeCellList(points);
NVTXPOP();
    ArrayHandle<unsigned int> particlesPerCell(cellList->elementsPerCell,access_location::device,access_mode::read);
    ArrayHandle<int> indices(cellList->particleIndices,access_location::device,access_mode::read);
    ArrayHandle<dVec> cellParticlePos(cellList->particlePositions,access_location::device,access_mode::read);
    ArrayHandle<int> d_adj(cellList->returnAdjacentCells(),access_location::device,access_mode::read);

    bool recompute = true;
    int Np = points.getNumElements();
    ArrayHandle<dVec> d_pt(points,access_location::device,access_mode::read);
    int nmax = Nmax;
    while(recompute)
        {
NVTXPUSH("primary neighborlist computation");
        resetNeighborsGPU(Np,nmax);
        {//scope
        ArrayHandle<unsigned int> d_npp(neighborsPerParticle,access_location::device,access_mode::readwrite);
        ArrayHandle<int> d_idx(particleIndices,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> d_vec(neighborVectors,access_location::device,access_mode::overwrite);
        ArrayHandle<int> d_assist(assist,access_location::device,access_mode::readwrite);
        //!call gpu function
        nlistTuner->begin();
        gpu_compute_neighbor_list(d_idx.data,
                                  d_npp.data,
                                  d_vec.data,
                                  particlesPerCell.data,
                                  indices.data,
                                  cellParticlePos.data,
                                  d_pt.data,
                                  d_assist.data,
                                  d_adj.data,
                                  *(Box),
                                  neighborIndexer,
                                  cellList->cellListIndexer,
                                  cellList->cellIndexer,
                                  cellList->adjacentCellIndexer,
                                  cellList->adjacentCellsPerCell,
                                  cellList->getBinsPerSide(),
                                  cellList->getCellSize(),
                                  cellList->getNmax(),
                                  maxRange,
                                  nmax,
                                  Np,
                                  nlistTuner->getParameter());
        nlistTuner->end();
        }//scope
NVTXPOP();
NVTXPUSH("neighborList assist checking");
        {
        ArrayHandle<int> h_assist(assist,access_location::host,access_mode::readwrite);
//        printf("h[0]=%i, h[1]=%i\n",h_assist.data[0],h_assist.data[1]);
        if(h_assist.data[1] == 1)
            {
            Nmax = h_assist.data[0];
            nmax = Nmax;
            h_assist.data[1] =0;
            recompute = true;
            }
        else
            recompute = false;
        };
NVTXPOP();
        };
    };

