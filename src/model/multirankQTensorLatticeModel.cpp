#include "multirankQTensorLatticeModel.h"
#include "multirankQTensorLatticeModel.cuh"
/*! \file multirankQTensorLatticeModel.cpp" */

multirankQTensorLatticeModel::multirankQTensorLatticeModel(int lx, int ly, int lz, bool _xHalo, bool _yHalo, bool _zHalo, bool _useGPU)
    : qTensorLatticeModel(lx,ly,lz,_useGPU), xHalo(_xHalo), yHalo(_yHalo), zHalo(_zHalo)
    {
    int Lx = lx;
    int Ly = ly;
    int Lz = lz;
    latticeSites.x = Lx;
    latticeSites.y = Ly;
    latticeSites.z = Lz;
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

/*!
maps between positions in the expanded lattice (base + halo sites) and the linear index of the position that site should reside in data.
respects the fact that the first N sites should be the base lattice, and the halo sites should all follow
*/
int multirankQTensorLatticeModel::indexInExpandedDataArray(int3 pos)
    {
    if(pos.x <0 && !xHalo)
        pos.x = latticeSites.x-1;
    if(pos.x ==latticeSites.x && !xHalo)
        pos.x = 0;
    if(pos.y <0 && !yHalo)
        pos.y = latticeSites.y-1;
    if(pos.y ==latticeSites.y && !yHalo)
        pos.y = 0;
    if(pos.z <0 && !zHalo)
        pos.z = latticeSites.z-1;
    if(pos.z ==latticeSites.z && !zHalo)
        pos.z = 0;

    if(pos.x < latticeSites.x && pos.y < latticeSites.y && pos.z < latticeSites.z && pos.x >=0 && pos.y >= 0 && pos.z >= 0)
        return latticeIndex(pos);

    //next consider the x = -1 face (total Ly * total Lz)
    int base = N;
    if(pos.x == -1)
        return base + pos.y + expandedLatticeSites.y*pos.z;
    //next the x + latticeSites.x + 1 face (note the off-by one fenceposting)
    base +=expandedLatticeSites.y*expandedLatticeSites.z;
    if(pos.x == latticeSites.x)
        return base + pos.y + expandedLatticeSites.y*pos.z;
    base +=expandedLatticeSites.y*expandedLatticeSites.z;
    //next consider the y = -1 face...  0 <=x < latticeSites, by -1 <= z <= latticeSites.z+1
    if(pos.y == -1)
        return base + pos.x + latticeSites.x*pos.z;
    base +=latticeSites.x*expandedLatticeSites.z;
    if(pos.y == latticeSites.y)
        return base + pos.x + latticeSites.x*pos.z;
    base +=latticeSites.x*expandedLatticeSites.z;

    //finally, the z-faces, for which x and y can only be 0 <= letter < latticeSites
    if(pos.z == -1)
        return base + pos.x + latticeSites.x*pos.y;
    base +=latticeSites.x*latticeSites.y;
    if(pos.z == latticeSites.z)
        return base + pos.x + latticeSites.x*pos.y;

    char message[256];
    sprintf(message,"inadmissible... {%i,%i,%i} = %i",pos.x,pos.y,pos.z,base);
    throw std::runtime_error(message);
    return -1;
    };

int multirankQTensorLatticeModel::getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType)
    {
    if(stencilType==0)
        {
        neighs = 6;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        if(!sliceSites)
            {
            int3 pos = latticeIndex.inverseIndex(target);
            neighbors[0] = indexInExpandedDataArray(pos.x-1,pos.y,pos.z);
            neighbors[1] = indexInExpandedDataArray(pos.x+1,pos.y,pos.z);
            neighbors[2] = indexInExpandedDataArray(pos.x,pos.y-1,pos.z);
            neighbors[3] = indexInExpandedDataArray(pos.x,pos.y+1,pos.z);
            neighbors[4] = indexInExpandedDataArray(pos.x,pos.y,pos.z-1);
            neighbors[5] = indexInExpandedDataArray(pos.x,pos.y,pos.z+1);

            }
        return target;
        };
    if(stencilType==1) //very wrong at the moment
        {
        UNWRITTENCODE("broken");
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

void multirankQTensorLatticeModel::parseDirectionType(int directionType, int &xyz, int &size1start, int &size1end, int &size2start, int &size2end,int &plane, bool sending)
    {
    xyz = 0;//the plane has fixed x
    size1start=0; size1end = latticeIndex.sizes.y;
    size2start=0; size2end = latticeIndex.sizes.z;
    if(directionType == 2 || directionType ==3)
        {
        xyz = 1;//the plane has fixed y
        size1end = latticeIndex.sizes.x;
        }
    if(directionType == 4 || directionType ==5)
        {
        xyz = 2;//the plane has fixed y
        size1end = latticeIndex.sizes.x;
        size2end = latticeIndex.sizes.y;
        }
    /* //When communicating faces, don't send extra lattice sites...
    if(yHalo)
        {size1start = -1; size1end = latticeIndex.sizes.y+1;}
    else
        {size1start = 0;  size1end = latticeIndex.sizes.y ;}
    if(zHalo)
        {size2start = -1; size2end = latticeIndex.sizes.z+1;}
    else
        {size2start = 0;  size2end = latticeIndex.sizes.z ;}
    if(directionType == 2 || directionType ==3)
        {
        xyz = 1;//the plane has fixed y
        if(xHalo)
            {size1start = -1; size1end = latticeIndex.sizes.x+1;}
        else
            {size1start = 0;  size1end = latticeIndex.sizes.x ;}
        }
    if(directionType == 4 || directionType ==5)
        {
        xyz = 2;//the plane has fixed z
        if(xHalo)
            {size1start = -1; size1end = latticeIndex.sizes.x+1;}
        else
            {size1start = 0;  size1end = latticeIndex.sizes.x ;}
        if(yHalo)
            {size2start = -1; size2end = latticeIndex.sizes.y+1;}
        else
            {size2start = 0;  size2end = latticeIndex.sizes.y ;}
        }
    */
    if(sending)
        {
        switch(directionType)
            {
            case 0: plane = 0; break;//smallest x plane
            case 1: plane = latticeSites.x-1; break;//largest x plane
            case 2: plane = 0; break;//smallest y plane
            case 3: plane = latticeSites.y-1; break;//largest y plane
            case 4: plane = 0; break;//smallest z plane
            case 5: plane = latticeSites.z-1; break;//largest z plane
            default:
                throw std::runtime_error("negative directionTypes are not valid");
            }
        }
    else //receiving
        {
        switch(directionType)
            {
            case 0: plane = -1; break;//smallest x plane
            case 1: plane = latticeSites.x; break;//largest x plane
            case 2: plane = -1; break;//smallest y plane
            case 3: plane = latticeSites.y; break;//largest y plane
            case 4: plane = -1; break;//smallest z plane
            case 5: plane = latticeSites.z; break;//largest z plane
            default:
                throw std::runtime_error("negative directionTypes are not valid");
            }
        };
    }

void multirankQTensorLatticeModel::prepareSendData(int directionType)
    {
    if(directionType <=5 )//send entire faces
        {
        int xyz,size1start,size1end,size2start,size2end,plane;
        parseDirectionType(directionType,xyz,size1start,size1end,size2start,size2end,plane,true);
        int nTot = (size1end-size1start)*(size2end-size2start);
        transferElementNumber = nTot;
        if(intTransferBufferSend.getNumElements() < nTot)
            {
            intTransferBufferSend.resize(nTot);
            doubleTransferBufferSend.resize(DIMENSION*nTot);
            intTransferBufferReceive.resize(nTot);
            doubleTransferBufferReceive.resize(DIMENSION*nTot);
            }
        int currentSite;
        //if(!useGPU)
        if(true)
            {
            ArrayHandle<int> ht(types,access_location::host,access_mode::read);
            ArrayHandle<dVec> hp(positions,access_location::host,access_mode::read);
            ArrayHandle<int> iBuf(intTransferBufferSend,access_location::host,access_mode::readwrite);
            ArrayHandle<scalar> dBuf(doubleTransferBufferSend,access_location::host,access_mode::readwrite);
            int idx = 0;
            int3 lPos;
            for (int ii = size1start; ii < size1end; ++ii)
                for (int jj = size2start; jj < size2end; ++jj)
                    {
                    if(xyz ==0)
                        {lPos.x = plane; lPos.y = ii; lPos.z = jj;}
                    if(xyz ==1)
                        {lPos.x = ii; lPos.y = plane; lPos.z = jj;}
                    if(xyz ==2)
                        {lPos.x = ii; lPos.y = jj; lPos.z = plane;}
                    currentSite = indexInExpandedDataArray(lPos);
                    iBuf.data[idx] = ht.data[currentSite];
                    for(int dd = 0; dd < DIMENSION; ++dd)
                        dBuf.data[DIMENSION*idx+dd] = hp.data[currentSite][dd];
                    idx+=1;
                    }
            }
        else
            {//GPU copy routine
            bool sending = true;
            ArrayHandle<int> dt(types,access_location::device,access_mode::readwrite);
            ArrayHandle<dVec> dp(positions,access_location::device,access_mode::readwrite);
            ArrayHandle<int> iBuf(intTransferBufferReceive,access_location::device,access_mode::read);
            ArrayHandle<scalar> dBuf(doubleTransferBufferReceive,access_location::device,access_mode::read);
            gpu_mrqtlm_buffer_data_exchange(sending,dt.data,dp.data,iBuf.data,dBuf.data,
                    size1start,size1end,size2start,size2end,
                    xyz,plane,latticeSites,expandedLatticeSites,latticeIndex,
                    xHalo,yHalo,zHalo);
            }
        }//end construction for sending faces
    }

void multirankQTensorLatticeModel::receiveData(int directionType)
    {
    if(directionType <=5 )//send entire faces
        {
        int xyz,size1start,size1end,size2start,size2end,plane;
        parseDirectionType(directionType,xyz,size1start,size1end,size2start,size2end,plane,false);
        int currentSite;
        if(!useGPU)
        //if(true)
            {
            ArrayHandle<int> ht(types,access_location::host,access_mode::readwrite);
            ArrayHandle<dVec> hp(positions,access_location::host,access_mode::readwrite);
            ArrayHandle<int> iBuf(intTransferBufferReceive,access_location::host,access_mode::read);
            ArrayHandle<scalar> dBuf(doubleTransferBufferReceive,access_location::host,access_mode::read);
            int idx = 0;
            int3 lPos;
            for (int ii = size1start; ii < size1end; ++ii)
                for (int jj = size2start; jj < size2end; ++jj)
                    {
                    if(xyz ==0)
                        {lPos.x = plane; lPos.y = ii; lPos.z = jj;}
                    if(xyz ==1)
                        {lPos.x = ii; lPos.y = plane; lPos.z = jj;}
                    if(xyz ==2)
                        {lPos.x = ii; lPos.y = jj; lPos.z = plane;}
                    currentSite = indexInExpandedDataArray(lPos);
                    ht.data[currentSite] = iBuf.data[idx];
                    for(int dd = 0; dd < DIMENSION; ++dd)
                        hp.data[currentSite][dd] = dBuf.data[DIMENSION*idx+dd];
                    idx+=1;
                    }
            }
        else
            {//GPU copy routine
            bool sending = false;
            ArrayHandle<int> dt(types,access_location::device,access_mode::readwrite);
            ArrayHandle<dVec> dp(positions,access_location::device,access_mode::readwrite);
            ArrayHandle<int> iBuf(intTransferBufferReceive,access_location::device,access_mode::read);
            ArrayHandle<scalar> dBuf(doubleTransferBufferReceive,access_location::device,access_mode::read);
            gpu_mrqtlm_buffer_data_exchange(sending,dt.data,dp.data,iBuf.data,dBuf.data,
                    size1start,size1end,size2start,size2end,
                    xyz,plane,latticeSites,expandedLatticeSites,latticeIndex,
                    xHalo,yHalo,zHalo);
            }
        }//end construction for sending faces
    }
