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

void multirankQTensorLatticeModel::parseDirectionType(int directionType, int &xyz, int &size1, int &size2, int &plane, bool sending)
    {
    xyz = 0;//the plane has fixed x
    size1 =expandedLatticeIndex.sizes.y;
    size2 =expandedLatticeIndex.sizes.z;
    if(directionType == 2 || directionType ==3)
        {
        xyz = 1;//the plane has fixed y
        size1 =expandedLatticeIndex.sizes.x;
        size2 =expandedLatticeIndex.sizes.z;
        }
    if(directionType == 4 || directionType ==5)
        {
        xyz = 2;//the plane has fixed z
        size1 =expandedLatticeIndex.sizes.x;
        size2 =expandedLatticeIndex.sizes.y;
        }
    if(sending)
        {
        switch(directionType)
            {
            case 0: plane = 0; break;//smallest x plane
            case 1: plane = expandedLatticeSites.x-3; break;//largest x plane
            case 2: plane = 0; break;//smallest y plane
            case 3: plane = expandedLatticeSites.y-3; break;//largest y plane
            case 4: plane = 0; break;//smallest z plane
            case 5: plane = expandedLatticeSites.z-3; break;//largest z plane
            default:
                throw std::runtime_error("negative directionTypes are not valid");
            }
        }
    else
        {
        switch(directionType)
            {
            case 0: plane = expandedLatticeSites.x-1; break;//smallest x plane
            case 1: plane = expandedLatticeSites.x-2; break;//largest x plane
            case 2: plane = expandedLatticeSites.y-1; break;//smallest y plane
            case 3: plane = expandedLatticeSites.y-2; break;//largest y plane
            case 4: plane = expandedLatticeSites.z-1; break;//smallest z plane
            case 5: plane = expandedLatticeSites.z-2; break;//largest z plane
            default:
                throw std::runtime_error("negative directionTypes are not valid");
            }
        };
    }

void multirankQTensorLatticeModel::prepareSendData(int directionType)
    {
    if(directionType <=5 )//send entire faces
        {
        int xyz,size1,size2,plane;
        parseDirectionType(directionType,xyz,size1,size2,plane,true);
        int nTot = size1*size2;
        if(intTransferBuffer.getNumElements() < nTot)
            {
            intTransferBuffer.resize(nTot);
            dvecTransferBuffer.resize(nTot);
            }
        int currentSite;
        //prepare to send the y-z plane at x=0 to the left
        if(!useGPU)
            {
            ArrayHandle<int> ht(types,access_location::host,access_mode::read);
            ArrayHandle<dVec> hp(positions,access_location::host,access_mode::read);
            ArrayHandle<int> iBuf(intTransferBuffer,access_location::host,access_mode::overwrite);
            ArrayHandle<dVec> dBuf(dvecTransferBuffer,access_location::host,access_mode::overwrite);
            int idx = 0;
            for (int ii = 0; ii < size1; ++ii)
                for (int jj = 0; jj < size2; ++jj)
                    {
                    if(xyz ==0)
                        currentSite = expandedLatticeIndex(plane,ii,jj);
                    if(xyz ==1)
                        currentSite = expandedLatticeIndex(ii,plane,jj);
                    if(xyz ==2)
                        currentSite = expandedLatticeIndex(ii,jj,plane);
                    iBuf.data[idx] = ht.data[currentSite];
                    dBuf.data[idx] = hp.data[currentSite];
                    idx+=1;
                    }
            }
        else
            {//GPU copy routine
            }
        }//end construction for sending faces
    }

void multirankQTensorLatticeModel::receiveData(int directionType)
    {
    if(directionType <=5 )//send entire faces
        {
        int xyz,size1,size2,plane;
        parseDirectionType(directionType,xyz,size1,size2,plane,false);
        int nTot = size1*size2;
        if(intTransferBuffer.getNumElements() < nTot)
            {
            intTransferBuffer.resize(nTot);
            dvecTransferBuffer.resize(nTot);
            }
        int currentSite;
        //prepare to send the y-z plane at x=0 to the left
        if(!useGPU)
            {
            ArrayHandle<int> ht(types,access_location::host,access_mode::readwrite);
            ArrayHandle<dVec> hp(positions,access_location::host,access_mode::readwrite);
            ArrayHandle<int> iBuf(intTransferBuffer,access_location::host,access_mode::read);
            ArrayHandle<dVec> dBuf(dvecTransferBuffer,access_location::host,access_mode::read);
            int idx = 0;
            for (int ii = 0; ii < size1; ++ii)
                for (int jj = 0; jj < size2; ++jj)
                    {
                    if(xyz ==0)
                        currentSite = expandedLatticeIndex(plane,ii,jj);
                    if(xyz ==1)
                        currentSite = expandedLatticeIndex(ii,plane,jj);
                    if(xyz ==2)
                        currentSite = expandedLatticeIndex(ii,jj,plane);
                    ht.data[currentSite] = iBuf.data[idx];
                    hp.data[currentSite] = dBuf.data[idx];
                    idx+=1;
                    }
            }
        else
            {//GPU copy routine
            }
        }//end construction for sending faces
    }
