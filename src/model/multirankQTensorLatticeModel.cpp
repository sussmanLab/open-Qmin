#include "multirankQTensorLatticeModel.h"
/*! \file multirankQTensorLatticeModel.cpp" */

multirankQTensorLatticeModel::multirankQTensorLatticeModel(int lx, int ly, int lz, bool _xHalo, bool _yHalo, bool _zHalo, bool _useGPU)
    : qTensorLatticeModel(lx,ly,lz,_useGPU), xHalo(_xHalo), yHalo(_yHalo), zHalo(_zHalo)
    {
    int Lx = lx;
    int Ly = ly;
    int Lz = lz;
    if(xHalo)
        Lx +=2;
    if(yHalo)
        Ly +=2;
    if(zHalo)
        Lz +=2;

    totalSites = Lx*Ly*Lz;
    expandedLatticeSites.x = Lx; 
    expandedLatticeSites.y = Ly; 
    expandedLatticeSites.z = Lz; 
    expandedLatticeIndex = Index3D(expandedLatticeSites);
    positions.resize(totalSites);
    types.resize(totalSites);
    forces.resize(totalSites);
    velocities.resize(totalSites);
    }

int multirankQTensorLatticeModel::getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType)
    {
    if(stencilType==0)
        {
        neighs = 6;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        if(!sliceSites)
            {
            int3 position = expandedLatticeIndex.inverseIndex(target);
            if(position.x >0 && position.x < expandedLatticeSites.x-1)
                {
                neighbors[0] = expandedLatticeIndex(position.x-1,position.y,position.z);
                neighbors[1] = expandedLatticeIndex(position.x+1,position.y,position.z);
                }
            else if(position.x ==0)
                {
                neighbors[0] = expandedLatticeIndex(expandedLatticeSites.x-1,position.y,position.z);
                neighbors[1] = expandedLatticeIndex(1,position.y,position.z);
                }
            else
                {
                neighbors[0] = expandedLatticeIndex(expandedLatticeSites.x-2,position.y,position.z);
                neighbors[1] = expandedLatticeIndex(0,position.y,position.z);
                };
            if(position.y >0 && position.y < expandedLatticeSites.y-1)
                {
                neighbors[2] = expandedLatticeIndex(position.x,position.y-1,position.z);
                neighbors[3] = expandedLatticeIndex(position.x,position.y+1,position.z);
                }
            else if(position.y ==0)
                {
                neighbors[2] = expandedLatticeIndex(position.x,expandedLatticeSites.y-1,position.z);
                neighbors[3] = expandedLatticeIndex(position.x,1 ,position.z);
                }
            else
                {
                neighbors[2] = expandedLatticeIndex(position.x,expandedLatticeSites.y-2,position.z);
                neighbors[3] = expandedLatticeIndex(position.x,0,position.z);
                };
            if(position.z >0 && position.z < expandedLatticeSites.z-1)
                {
                neighbors[4] = expandedLatticeIndex(position.x,position.y,position.z-1);
                neighbors[5] = expandedLatticeIndex(position.x,position.y,position.z+1);
                }
            else if(position.z ==0)
                {
                neighbors[4] = expandedLatticeIndex(position.x,position.y,expandedLatticeSites.z-1);
                neighbors[5] = expandedLatticeIndex(position.x,position.y,1);
                }
            else
                {
                neighbors[4] = expandedLatticeIndex(position.x,position.y,expandedLatticeSites.z-2);
                neighbors[5] = expandedLatticeIndex(position.x,position.y,0);
                };
            }
        return target;
        };
    if(stencilType==1)
        {
        int3 position = expandedLatticeIndex.inverseIndex(target);
        neighs = 18;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        neighbors[0] = expandedLatticeIndex(wrap(position.x-1,expandedLatticeSites.x),position.y,position.z);
        neighbors[1] = expandedLatticeIndex(wrap(position.x+1,expandedLatticeSites.x),position.y,position.z);
        neighbors[2] = expandedLatticeIndex(position.x,wrap(position.y-1,expandedLatticeSites.y),position.z);
        neighbors[3] = expandedLatticeIndex(position.x,wrap(position.y+1,expandedLatticeSites.y),position.z);
        neighbors[4] = expandedLatticeIndex(position.x,position.y,wrap(position.z-1,expandedLatticeSites.z));
        neighbors[5] = expandedLatticeIndex(position.x,position.y,wrap(position.z+1,expandedLatticeSites.z));

        neighbors[6] = expandedLatticeIndex(wrap(position.x-1,expandedLatticeSites.x),wrap(position.y-1,expandedLatticeSites.y),position.z);
        neighbors[7] = expandedLatticeIndex(wrap(position.x-1,expandedLatticeSites.x),wrap(position.y+1,expandedLatticeSites.y),position.z);
        neighbors[8] = expandedLatticeIndex(wrap(position.x-1,expandedLatticeSites.x),position.y,wrap(position.z-1,expandedLatticeSites.z));
        neighbors[9] = expandedLatticeIndex(wrap(position.x-1,expandedLatticeSites.x),position.y,wrap(position.z+1,expandedLatticeSites.z));
        neighbors[10] = expandedLatticeIndex(wrap(position.x+1,expandedLatticeSites.x),wrap(position.y-1,expandedLatticeSites.y),position.z);
        neighbors[11] = expandedLatticeIndex(wrap(position.x+1,expandedLatticeSites.x),wrap(position.y+1,expandedLatticeSites.y),position.z);
        neighbors[12] = expandedLatticeIndex(wrap(position.x+1,expandedLatticeSites.x),position.y,wrap(position.z-1,expandedLatticeSites.z));
        neighbors[13] = expandedLatticeIndex(wrap(position.x+1,expandedLatticeSites.x),position.y,wrap(position.z+1,expandedLatticeSites.z));

        neighbors[14] = expandedLatticeIndex(position.x,wrap(position.y-1,expandedLatticeSites.y),wrap(position.z-1,expandedLatticeSites.z));
        neighbors[15] = expandedLatticeIndex(position.x,wrap(position.y-1,expandedLatticeSites.y),wrap(position.z+1,expandedLatticeSites.z));
        neighbors[16] = expandedLatticeIndex(position.x,wrap(position.y+1,expandedLatticeSites.y),wrap(position.z-1,expandedLatticeSites.z));
        neighbors[17] = expandedLatticeIndex(position.x,wrap(position.y+1,expandedLatticeSites.y),wrap(position.z+1,expandedLatticeSites.z));
        return target;
        }

    return target; //nope
    };

