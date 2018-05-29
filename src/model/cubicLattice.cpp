#include "cubicLattice.h"
//#include "cubicLattice.cuh"
/*! /file cubicLattice.cpp */

cubicLattice::cubicLattice(int l, bool _slice, bool _useGPU)
    {
    latticeIndex = Index3D(l);
    sliceSites = _slice;
    N = l*l*l;
    L=l;
    selfForceCompute = false;
    positions.resize(N);
    types.resize(N);
    forces.resize(N);
    vector<dVec> zeroes(N,make_dVec(0.0));
    vector<int> units(N,0);
    fillGPUArrayWithVector(units,types);
    fillGPUArrayWithVector(zeroes,positions);
    fillGPUArrayWithVector(zeroes,forces);
    };

void cubicLattice::setSpinsRandomly(noiseSource &noise)
    {
    ArrayHandle<dVec> pos(positions);
    for(int pp = 0; pp < N; ++pp)
        {
        for (int dd = 0; dd < DIMENSION; ++dd)
            pos.data[pp][dd] = noise.getRealNormal();
        scalar lambda = dot(pos.data[pp],pos.data[pp]);
        pos.data[pp] = (1/sqrt(lambda))*pos.data[pp];
        };
    };

int cubicLattice::getNeighbors(int target, int4 &xyNeighbors, int2 &zNeighbors)
    {
    if(!sliceSites)
        {
        int3 position = latticeIndex.inverseIndex(target);
        if(position.x >0 && position.x < L-1)
            {
            xyNeighbors.x = latticeIndex(position.x-1,position.y,position.z);
            xyNeighbors.y = latticeIndex(position.x+1,position.y,position.z);
            }
        else if(position.x ==0)
            {
            xyNeighbors.x = latticeIndex(L-1,position.y,position.z);
            xyNeighbors.y = latticeIndex(1,position.y,position.z);
            }
        else
            {
            xyNeighbors.x = latticeIndex(L-2,position.y,position.z);
            xyNeighbors.y = latticeIndex(0,position.y,position.z);
            };
        if(position.y >0 && position.y < L-1)
            {
            xyNeighbors.z = latticeIndex(position.x,position.y-1,position.z);
            xyNeighbors.w = latticeIndex(position.x,position.y+1,position.z);
            }
        else if(position.y ==0)
            {
            xyNeighbors.z = latticeIndex(position.x,L-1,position.z);
            xyNeighbors.w = latticeIndex(position.x,1 ,position.z);
            }
        else
            {
            xyNeighbors.z = latticeIndex(position.x,L-2,position.z);
            xyNeighbors.w = latticeIndex(position.x,0,position.z);
            };
        if(position.z >0 && position.z < L-1)
            {
            zNeighbors.x = latticeIndex(position.x,position.y,position.z-1);
            zNeighbors.y = latticeIndex(position.x,position.y,position.z+1);
            }
        else if(position.z ==0)
            {
            zNeighbors.x = latticeIndex(position.x,position.y,L-1);
            zNeighbors.y = latticeIndex(position.x,position.y,1);
            }
        else
            {
            zNeighbors.x = latticeIndex(position.x,position.y,L-2);
            zNeighbors.y = latticeIndex(position.x,position.y,0);
            };
        return target;
        };
    return target; //nope
    };
