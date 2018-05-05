#include "neighborList.h"
#include "utilities.cuh"
//#include "neighborList.h"
/*! \file neighborList.cpp */

neighborList::neighborList(scalar range, BoxPtr _box)
    {
    useGPU = false;
    saveDistanceData = true;
    Box = _box;
    cellList = make_shared<hyperrectangularCellList>(range,Box);
    Nmax = 3;
    maxRange = range;
    };

void neighborList::resetNeighborsGPU(int size)
    {
    if(neighborsPerParticle.getNumElements() != size)
        neighborsPerParticle.resize(size);
    ArrayHandle<unsigned int> d_npp(neighborsPerParticle,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_npp.data,size);
    
    neighborIndexer = Index2D(Nmax,size);
    if(particleIndices.getNumElements() != neighborIndexer.getNumElements())
        {
        particleIndices.resize(neighborIndexer.getNumElements());
        if(saveDistanceData)
            {
            neighborVectors.resize(neighborIndexer.getNumElements());
            neighborDistances.resize(neighborIndexer.getNumElements());
            };
        };

    ArrayHandle<int> d_idx(particleIndices,access_location::device,access_mode::overwrite);
    gpu_zero_array(d_idx.data,size);
    if(saveDistanceData)
        {
        ArrayHandle<dVec> d_vec(neighborVectors,access_location::device,access_mode::overwrite);
        ArrayHandle<scalar> d_dist(neighborDistances,access_location::device,access_mode::overwrite);
        gpu_zero_array(d_vec.data,size);
        gpu_zero_array(d_dist.data,size);
        };

    if(assist.getNumElements()!= 2)
        assist.resize(2);
    ArrayHandle<int> h_assist(assist,access_location::host,access_mode::overwrite);
    h_assist.data[0]=Nmax;
    h_assist.data[1] = 0;
    };

void neighborList::resetNeighborsCPU(int size)
    {
    if(neighborsPerParticle.getNumElements() != size)
        neighborsPerParticle.resize(size);
    ArrayHandle<unsigned int> h_npp(neighborsPerParticle,access_location::host,access_mode::overwrite);
    for (int i = 0; i < size; ++i)
        h_npp.data[i] = 0;
    
    neighborIndexer = Index2D(Nmax,size);
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
    h_assist.data[0]=Nmax;
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
        resetNeighborsCPU(Np);
        ArrayHandle<unsigned int> h_npp(neighborsPerParticle);
        ArrayHandle<int> h_idx(particleIndices);
        ArrayHandle<dVec> h_vec(neighborVectors);
        ArrayHandle<scalar> h_dist(neighborDistances);
        recompute = false;
        vector<int> cellsToScan;
        for (int pp = 0; pp < Np; ++pp)
            {
            dVec target = h_pt.data[pp];
            int cell = cellList->positionToCellIndex(target);
            cellList->getCellNeighbors(cell,1,cellsToScan);
            for (int cc = 0; cc < cellsToScan.size();++cc)
                {
                int currentCell = cellsToScan[cc];
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
        printf("neighIndexer size %i\t %i\t %i\n", neighborIndexer.getNumElements(),Nmax,Np);
    };

/*!
\param points the set of points to find neighbors for
 */
void neighborList::computeGPU(GPUArray<dVec> &points)
    {
    cellList->computeCellList(points);
    UNWRITTENCODE("gpu neighbor list stuff");
    };

