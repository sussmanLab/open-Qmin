#include "squareLattice.h"
#include "functions.h"
#include "cubicLattice.cuh"//some basic spin functionality has already been written, such as simple spin updating GPU calls -- refactor code later
/*
no gpu functions written yet
#include "squareLattice.cuh"
*/
/*! \file squareLattice.cpp */

squareLattice::squareLattice(int l, bool _slice, bool _useGPU, bool _neverGPU)
    {
    useGPU=_useGPU;
    neverGPU = _neverGPU;
    sliceSites = _slice;
    N = l*l;
    L=l;
    Box = make_shared<periodicBoundaryConditions>(L);
    selfForceCompute = false;
    initializeNSites();
    normalizeSpins = true;
    latticeIndex = Index2D(l);
    };

squareLattice::squareLattice(int lx, int ly, bool _slice, bool _useGPU, bool _neverGPU)
    {
    useGPU=_useGPU;
    neverGPU = _neverGPU;
    sliceSites = _slice;
    N = lx*ly;
    Box = make_shared<periodicBoundaryConditions>(lx);//should not be used; lattices carry around a Box for compatibility reasons with other parts of the code.
    latticeIndex = Index2D(lx, ly);
    selfForceCompute = false;
    normalizeSpins = true;
    initializeNSites();
    };

void squareLattice::initializeNSites()
    {
    if(neverGPU)
        {
        neighboringSites.noGPU = true;
        boundaries.noGPU = true;
        boundaryMoveAssist1.noGPU = true;
        boundaryMoveAssist2.noGPU = true;
        }
    initializeSimpleModel(N);

    moveParticlesTuner = make_shared<kernelTuner>(512,1024,128,10,200000);
    };

void squareLattice::moveParticles(GPUArray<dVec> &dofs,GPUArray<dVec> &displacements,scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(dofs);
        for(int pp = 0; pp < N; ++pp)
            {
            h_pos.data[pp] += scale*h_disp.data[pp];
            if(normalizeSpins)
                {
                scalar nrm = norm(h_pos.data[pp]);
                h_pos.data[pp] = (1/nrm)*h_pos.data[pp];
                }
            }
        }
    else
        {//gpu branch
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(dofs,access_location::device,access_mode::readwrite);
        gpu_update_spins(d_disp.data,d_pos.data,scale,N,normalizeSpins);
        };
    };


void squareLattice::moveParticles(GPUArray<dVec> &displacements, scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(positions);
        for(int pp = 0; pp < N; ++pp)
            {
            h_pos.data[pp] += scale*h_disp.data[pp];
            if(normalizeSpins)
                {
                scalar nrm = norm(h_pos.data[pp]);
                h_pos.data[pp] = (1/nrm)*h_pos.data[pp];
                }
            }
        }
    else
        {//gpu branch
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        gpu_update_spins(d_disp.data,d_pos.data,scale,N,normalizeSpins);
        };
    };
     
void squareLattice::setSpinsRandomly(noiseSource &noise)
    {
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        for(int pp = 0; pp < N; ++pp)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                pos.data[pp][dd] = noise.getRealNormal();
            scalar lambda = dot(pos.data[pp],pos.data[pp]);
            pos.data[pp] = (1/sqrt(lambda))*pos.data[pp];
            };
        }
    else
        {
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::overwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(nBlocks);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_spins(pos.data,d_curandRNGs.data, blockSize,nBlocks,N);
        }
    };

int squareLattice::latticeSiteToLinearIndex(const int2 &target)
    {
    //sliceSites functionality not currently implemented... always just return the latticeIndex of the target
    return latticeIndex(target);
    }

//Filling the neighbor lists is currently *always* done on the cpu at the beginning of the simulation
void squareLattice::fillNeighborLists(int stencilType)
    {
    vector<int> neighs;
    int nNeighs;
    int temp = getNeighbors(0, neighs,nNeighs,stencilType);

    neighborIndex = Index2D(nNeighs,N);
    neighboringSites.resize(nNeighs*N);
    {//array handle scope
    ArrayHandle<int> neighbors(neighboringSites);
    for (int ii = 0; ii < N; ++ii)
        {
        temp = getNeighbors(ii, neighs,nNeighs,stencilType);
        for (int jj = 0; jj < nNeighs; ++jj)
            {
            neighbors.data[neighborIndex(jj,ii)] = neighs[jj];
            }
        }
    }

    };

/*!
returns, in the vector "neighbors" a list of lattice neighbors of the target site.
If stencilType ==0, the result will be
neighs = 4;
neighbors = {xMinus, xPlus, yMinus, yPlus};
If stencilType ==1 neighbors will be suitable for computing 9-point stencil laplacians on the 2D lattice:
neighs = 8;
neighbors = {xMinus, xPlus, yMinus, yPlus,
            xMinus_yMinus,xMinus_yPlus,xPlus_yMinus,xPlus_yPlus}
*/
int squareLattice::getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType)
    {
    if(stencilType==0)
        {
        neighs = 4;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        if(!sliceSites)
            {
            int2 position = latticeIndex.inverseIndex(target);
            if(position.x >0 && position.x < latticeIndex.width-1)
                {
                neighbors[0] = latticeIndex(position.x-1,position.y);
                neighbors[1] = latticeIndex(position.x+1,position.y);
                }
            else if(position.x ==0)
                {
                neighbors[0] = latticeIndex(latticeIndex.width-1,position.y);
                neighbors[1] = latticeIndex(1,position.y);
                }
            else
                {
                neighbors[0] = latticeIndex(latticeIndex.width-2,position.y);
                neighbors[1] = latticeIndex(0,position.y);
                };
            if(position.y >0 && position.y < latticeIndex.height-1)
                {
                neighbors[2] = latticeIndex(position.x,position.y-1);
                neighbors[3] = latticeIndex(position.x,position.y+1);
                }
            else if(position.y ==0)
                {
                neighbors[2] = latticeIndex(position.x,latticeIndex.height-1);
                neighbors[3] = latticeIndex(position.x,1);
                }
            else
                { neighbors[2] = latticeIndex(position.x,latticeIndex.height-2);
                neighbors[3] = latticeIndex(position.x,0);
                };
           
            return target;
            };
        }

    if(stencilType==1)
        {
        int2 position = latticeIndex.inverseIndex(target);
        neighs = 8;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        neighbors[0] = latticeIndex(wrap(position.x-1,latticeIndex.width),position.y);
        neighbors[1] = latticeIndex(wrap(position.x+1,latticeIndex.width),position.y);
        neighbors[2] = latticeIndex(position.x,wrap(position.y-1,latticeIndex.height));
        neighbors[3] = latticeIndex(position.x,wrap(position.y+1,latticeIndex.height));

        neighbors[4] = latticeIndex(wrap(position.x-1,latticeIndex.width),wrap(position.y-1,latticeIndex.height));
        neighbors[5] = latticeIndex(wrap(position.x-1,latticeIndex.width),wrap(position.y+1,latticeIndex.height));;
        neighbors[6] = latticeIndex(wrap(position.x+1,latticeIndex.width),wrap(position.y-1,latticeIndex.height));
        neighbors[7] = latticeIndex(wrap(position.x+1,latticeIndex.width),wrap(position.y+1,latticeIndex.height));
        }

    return target; 
};



void squareLattice::createBoundaryObject(vector<int> &latticeSites, boundaryType _type, scalar Param1, scalar Param2)
{
    //throw std::runtime_error("createBoundaryObject function not yet written");
    UNWRITTENCODE("position to index in squareLattice... currently this function is unwritten");
};


void squareLattice::displaceBoundaryObject(int objectIndex, int motionDirection, int magnitude)
{
    //throw std::runtime_error("displayBoundaryObject function not yet written");
    UNWRITTENCODE("position to index in squareLattice... currently this function is unwritten");
};


