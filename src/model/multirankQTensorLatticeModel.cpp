#include "multirankQTensorLatticeModel.h"
#include "multirankQTensorLatticeModel.cuh"
/*! \file multirankQTensorLatticeModel.cpp" */

multirankQTensorLatticeModel::multirankQTensorLatticeModel(int lx, int ly, int lz, bool _xHalo, bool _yHalo, bool _zHalo, bool _useGPU, bool _neverGPU)
    : qTensorLatticeModel(lx,ly,lz,_useGPU,_neverGPU), xHalo(_xHalo), yHalo(_yHalo), zHalo(_zHalo)
    {
    int Lx = lx;
    int Ly = ly;
    int Lz = lz;
    latticeSites.x = Lx;
    latticeSites.y = Ly;
    latticeSites.z = Lz;
    if(neverGPU)
        {
        intTransferBufferSend.noGPU = true;
        intTransferBufferReceive.noGPU = true;
        doubleTransferBufferSend.noGPU = true;
        doubleTransferBufferReceive.noGPU = true;
        }
    determineBufferLayout();

    if(xHalo)
        Lx +=2;
    if(yHalo)
        Ly +=2;
    if(zHalo)
        Lz +=2;

    totalSites = N;
    if(xHalo || yHalo || zHalo)
        totalSites = N+transferStartStopIndexes[25].y;
    //printf("total sites: %i\n",totalSites);
    positions.resize(totalSites);
    types.resize(totalSites);
    forces.resize(totalSites);
    velocities.resize(totalSites);

    //by default, set sites that interface with the other ranks to a negative type
    int tTest = 0;
    ArrayHandle<int> h_t(types);
    for (int ii = 0; ii < N; ++ii)
        {
        int3 site = indexToPosition(ii);
        if( xHalo && (site.x ==0 || site.x == latticeSites.x-1))
            h_t.data[ii]=-2;
        if( yHalo && (site.y ==0 || site.y == latticeSites.y-1))
            h_t.data[ii]=-2;
        if( zHalo && (site.z ==0 || site.z == latticeSites.z-1))
            h_t.data[ii]=-2;
        if(h_t.data[ii] == -2) tTest +=1;
        }
    }

void multirankQTensorLatticeModel::setRandomDirectors(noiseSource &noise, scalar s0, bool globallyAligned)
    {
    scalar globalTheta = acos(2.0*noise.getRealUniform()-1);
    scalar globalPhi = 2.0*PI*noise.getRealUniform();
    ArrayHandle<dVec> pos(positions);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        scalar theta = acos(2.0*noise.getRealUniform()-1);
        scalar phi = 2.0*PI*noise.getRealUniform();
        if(globallyAligned)
            {
            theta = globalTheta;
            phi = globalPhi;
            }
        if(t.data[ii] <=0)
            {
            scalar3 n;
            n.x = cos(phi)*sin(theta);
            n.y = sin(phi)*sin(theta);
            n.z = cos(theta);
            qTensorFromDirector(n, s0, pos.data[ii]);
            };
        }

    };

void multirankQTensorLatticeModel::setUniformDirectors(scalar3 targetDirector, scalar s0)
    {
    ArrayHandle<dVec> pos(positions);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        if(t.data[ii] <=0)
            qTensorFromDirector(targetDirector,s0,pos.data[ii]);
        };
    };
    
void multirankQTensorLatticeModel::setDirectorFromFunction(std::function<scalar4(scalar,scalar,scalar)> func)
    {
    ArrayHandle<dVec> p(positions);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    for (int ii = 0; ii < N; ++ii)
        {
        if(t.data[ii] <=0)
            {
            //get global, not local. position if lattice site
            int3 pos = indexToPosition(ii);
            pos.x += latticeMinPosition.x;
            pos.y += latticeMinPosition.y;
            pos.z += latticeMinPosition.z;
            scalar4 functionResult = func(pos.x,pos.y,pos.z);
            scalar3 n;
            n.x = functionResult.x;
            n.y = functionResult.y;
            n.z = functionResult.z;
            scalar s0 = functionResult.w;
            qTensorFromDirector(n,s0,p.data[ii]);
            }
        }
    };

int3 multirankQTensorLatticeModel::indexToPosition(int idx)
    {

    if (idx < N)
        return latticeIndex.inverseIndex(idx);

    if(idx >=totalSites)
        throw std::runtime_error("invalid index requested");
    int ii = idx - N;
    int directionType = 0;
    int3 ans;
    while(ii > transferStartStopIndexes[directionType].y)
        directionType += 1;
    getBufferInt3FromIndex(ii, ans, directionType, false);
    if(ans.x < -1) printf("%i, %i, %i\n",idx,ii,directionType);
    return ans;
    }

/*!
Meant to be used with idx in (transferStartStopIndexes[directionType].x to ".y)
\param idx the linear index in between 0 and the maximum number of extended sites
\param pos gets filled with the right lattice position, with correct send/receive dependence
\param directionType an int specifying the type of face/edge/corner. See comments in determingBufferLayout() for the mapping between 0 and 25 to the possibilities
\param sending flag the either restricts pos to be withing 0 and latticeSites, or the halo sites if false
*/
void multirankQTensorLatticeModel::getBufferInt3FromIndex(int idx, int3 &pos,int directionType, bool sending)
    {
    int startIdx = transferStartStopIndexes[directionType].x;
    int index = idx - startIdx;
    switch(directionType)
        {
        case 0:
            pos.z = index / latticeSites.y; pos.y = index % latticeSites.y;
            pos.x = sending ? 0 : -1;
            break;
        case 1:
            pos.z = index / latticeSites.y; pos.y = index % latticeSites.y;
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            break;
        case 2:
            pos.z = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.y = sending ? 0 : -1;
            break;
        case 3:
            pos.z = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            break;
        case 4:
            pos.y = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.z = sending ? 0 : -1;
            break;
        case 5:
            pos.y = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            break;
        //edges
        case 6:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? 0 : -1;
            pos.z = index;
            break;
        case 7:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = index;
            break;
        case 8:
            pos.x = sending ? 0 : -1;
            pos.z = sending ? 0 : -1;
            pos.y = index;
            break;
        case 9:
            pos.x = sending ? 0 : -1;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            pos.y = index;
            break;
        case 10:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? 0 : -1;
            pos.z = index;
            break;
        case 11:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = index;
            break;
        case 12:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.z = sending ? 0 : -1;
            pos.y = index;
            break;
        case 13:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            pos.y = index;
            break;
        case 14:
            pos.y = sending ? 0 : -1;
            pos.z = sending ? 0 : -1;
            pos.x = index;
            break;
        case 15:
            pos.y = sending ? 0 : -1;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            pos.x = index;
            break;
        case 16:
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? 0 : -1;
            pos.x = index;
            break;
        case 17:
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            pos.x = index;
            break;
        //corners
        case 18:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? 0 : -1;
            pos.z = sending ? 0 : -1;
            break;
        case 19:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? 0 : -1;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            break;
        case 20:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? 0 : -1;
            break;
        case 21:
            pos.x = sending ? 0 : -1;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            break;
        case 22:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? 0 : -1;
            pos.z = sending ? 0 : -1;
            break;
        case 23:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? 0 : -1;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            break;
        case 24:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? 0 : -1;
            break;
        case 25:
            pos.x = sending ? latticeSites.x-1 : latticeSites.x;
            pos.y = sending ? latticeSites.y-1 : latticeSites.y;
            pos.z = sending ? latticeSites.z-1 : latticeSites.z;
            break;
        default:
            printf("direction type %i, position %i %i %i is invalid",directionType,pos.x,pos.y,pos.z);
            throw std::runtime_error("Fail");
        }
    }


/*!
Given the buffer layout below, determine the data array index of a given position
*/
int multirankQTensorLatticeModel::positionToIndex(int3 &pos)
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
    int base = N;
    //0: x = -1 face
    base = N + transferStartStopIndexes[0].x;
    if(pos.x == -1 && ordered(0, pos.y, latticeSites.y-1) && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.y+latticeSites.y*pos.z;
    //1: x = max face +1
    base = N + transferStartStopIndexes[1].x;
    if(pos.x == latticeSites.x && ordered(0, pos.y, latticeSites.y-1) && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.y+latticeSites.y*pos.z;
    //2: y = -1
    base = N + transferStartStopIndexes[2].x;
    if(pos.y == -1 && ordered(0, pos.x, latticeSites.x-1) && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.x+latticeSites.x*pos.z;
    //3: y = max face
    base = N + transferStartStopIndexes[3].x;
    if(pos.y == latticeSites.y && ordered(0, pos.x, latticeSites.x-1) && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.x+latticeSites.x*pos.z;
    //4: z = -1
    base = N + transferStartStopIndexes[4].x;
    if(pos.z == -1 && ordered(0, pos.x, latticeSites.x-1) && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.x+latticeSites.x*pos.y;
    //5: z = max face+1
    base = N + transferStartStopIndexes[5].x;
    if(pos.z == latticeSites.z && ordered(0, pos.x, latticeSites.x-1) && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.x+latticeSites.x*pos.y;

    //6: x = -1, y = -1  edge
    base = N + transferStartStopIndexes[6].x;
    if(pos.x==-1 && pos.y==-1 && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.z;
    //7: x = -1, y = max  edge
    base = N + transferStartStopIndexes[7].x;
    if(pos.x==-1 && pos.y==latticeSites.y && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.z;
    //8: x = -1, z = -1  edge
    base = N + transferStartStopIndexes[8].x;
    if(pos.x==-1 && pos.z==-1 && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.y;
    //9: x = -1, z = max  edge
    base = N + transferStartStopIndexes[9].x;
    if(pos.x==-1 && pos.z==latticeSites.z  && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.y;
    //10: x = max, y = -1  edge
    base = N + transferStartStopIndexes[10].x;
    if(pos.x== latticeSites.x && pos.y==-1 && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.z;
    //11: x = max, y = max  edge
    base = N + transferStartStopIndexes[11].x;
    if(pos.x== latticeSites.x && pos.y==latticeSites.y && ordered(0,pos.z,latticeSites.z-1))
        return base + pos.z;
    //12: x = max, z = -1  edge
    base = N + transferStartStopIndexes[12].x;
    if(pos.x==latticeSites.x && pos.z==-1 && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.y;
    //13: x = max, z = max  edge
    base = N + transferStartStopIndexes[13].x;
    if(pos.x==latticeSites.x && pos.z==latticeSites.z && ordered(0,pos.y,latticeSites.y-1))
        return base + pos.y;
    //14: y = -1, z = -1  edge
    base = N + transferStartStopIndexes[14].x;
    if(pos.y==-1 && pos.z==-1 && ordered(0,pos.x,latticeSites.x-1))
        return base + pos.x;
    //15: y = -1, z = max  edge
    base = N + transferStartStopIndexes[15].x;
    if(pos.y==-1 && pos.z==latticeSites.z && ordered(0,pos.x,latticeSites.x-1))
        return base + pos.x;
    //16: y = max, z = -1  edge
    base = N + transferStartStopIndexes[16].x;
    if(pos.y==latticeSites.y && pos.z==-1 && ordered(0,pos.x,latticeSites.x-1))
        return base + pos.x;
    //17: y = max, z = max  edge
    base = N + transferStartStopIndexes[17].x;
    if(pos.y==latticeSites.y && pos.z==latticeSites.z && ordered(0,pos.x,latticeSites.x-1))
        return base + pos.x;

    //18: x = -1, y = -1, z=-1 corner
    base = N + transferStartStopIndexes[18].x;
    if(pos.x == -1)
        {
        if(pos.y == -1)
            {
            if (pos.z == -1)
                return base;
            else if (pos.z == latticeSites.z)
                return base + 1;
            }
        else if(pos.y == latticeSites.y)
            {
            if (pos.z == -1)
                return base+2;
            else if (pos.z == latticeSites.z)
                return base+3;
            }
        }
    else if (pos.x == latticeSites.x)
        {
        if(pos.y == -1)
            {
            if (pos.z == -1)
                return base+4;
            else if (pos.z == latticeSites.z)
                return base+5;
            }
        else if(pos.y == latticeSites.y)
            {
            if (pos.z == -1)
                return base+6;
            else if (pos.z == latticeSites.z)
                return base+7;
            }
        }

    printf("(%i %i %i)\n",pos.x,pos.y,pos.z);
    throw std::runtime_error("invalid site requested");
    }

void multirankQTensorLatticeModel::prepareSendingBuffer(int directionType)
    {
    if(!useGPU)
        {
        ArrayHandle<int> ht(types,access_location::host,access_mode::read);
        ArrayHandle<dVec> hp(positions,access_location::host,access_mode::read);
        ArrayHandle<int> iBuf(intTransferBufferSend,access_location::host,access_mode::readwrite);
        ArrayHandle<scalar> dBuf(doubleTransferBufferSend,access_location::host,access_mode::readwrite);
        int2 startStop = transferStartStopIndexes[directionType];
        int3 pos;
        int currentSite;
        for (int ii = startStop.x; ii <=startStop.y; ++ii)
            {
            getBufferInt3FromIndex(ii,pos,directionType,true);
            currentSite = positionToIndex(pos);
            iBuf.data[ii] = ht.data[currentSite];
            for(int dd = 0; dd < DIMENSION; ++dd)
                dBuf.data[DIMENSION*ii+dd] = hp.data[currentSite][dd];
            }
        }//end of CPU part
    else
        {
        ArrayHandle<int> ht(types,access_location::device,access_mode::read);
        ArrayHandle<dVec> hp(positions,access_location::device,access_mode::read);
        ArrayHandle<int> iBuf(intTransferBufferSend,access_location::device,access_mode::readwrite);
        ArrayHandle<scalar> dBuf(doubleTransferBufferSend,access_location::device,access_mode::readwrite);
        int maxIndex = transferStartStopIndexes[transferStartStopIndexes.size()-1].y;
        gpu_prepareSendingBuffer(ht.data,hp.data,iBuf.data,dBuf.data,latticeSites,latticeIndex,maxIndex);
        }
    }

void multirankQTensorLatticeModel::readReceivingBuffer(int directionType)
    {
    int2 startStop = transferStartStopIndexes[directionType];
    int3 pos;
    int currentSite;
    if(!useGPU)
        {
        ArrayHandle<int> ht(types,access_location::host,access_mode::readwrite);
        ArrayHandle<dVec> hp(positions,access_location::host,access_mode::readwrite);
        ArrayHandle<int> iBuf(intTransferBufferReceive,access_location::host,access_mode::read);
        ArrayHandle<scalar> dBuf(doubleTransferBufferReceive,access_location::host,access_mode::read);
        for (int ii = startStop.x; ii <=startStop.y; ++ii)
            {
            //The data layout has been chosen so that ii+N is the right position
            /*
            getBufferInt3FromIndex(ii,pos,directionType,false);
            currentSite = positionToIndex(pos);
            */
            currentSite = ii+N;
            ht.data[currentSite] = iBuf.data[ii];
            for(int dd = 0; dd < DIMENSION; ++dd)
                hp.data[currentSite][dd] = dBuf.data[DIMENSION*ii+dd];
            }
        }//end of CPU part
    else
        {
        //GPU branch reads *the entire* buffer in one kernel call
        ArrayHandle<int> ht(types,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> hp(positions,access_location::device,access_mode::readwrite);
        ArrayHandle<int> iBuf(intTransferBufferReceive,access_location::device,access_mode::read);
        ArrayHandle<scalar> dBuf(doubleTransferBufferReceive,access_location::device,access_mode::read);
        int maxIndex = transferStartStopIndexes[transferStartStopIndexes.size()-1].y;
        gpu_copyReceivingBuffer(ht.data,hp.data,iBuf.data,dBuf.data,N,maxIndex);
        }
    }

/*!
During halo site communication, each rank will first completely fill the send buffer, and then send/receives within the buffers will be performed.
Finally, the entire receive buffer will be transfered into the expanded data arrays.
To facilitate this, a specific layout of the transfer buffers will be adopted for easy package/send/receive patterns:
the x = 0 face will be the first (Ly*Lz) elemetents, followed by the other faces, the edges, and finally the 8 corners.
 */
void multirankQTensorLatticeModel::determineBufferLayout()
    {
    int xFaces = latticeSites.z*latticeSites.y;
    int yFaces = latticeSites.z*latticeSites.x;
    int zFaces = latticeSites.x*latticeSites.y;
    //0: x = 0 face
    int2 startStop; startStop.x = 0; startStop.y = xFaces - 1;
    transferStartStopIndexes.push_back(startStop);
    //1: x = max face
    startStop.x = startStop.y+1; startStop.y += xFaces;
    transferStartStopIndexes.push_back(startStop);
    //2: y = 0
    startStop.x = startStop.y+1; startStop.y += yFaces;
    transferStartStopIndexes.push_back(startStop);
    //3: y = max face
    startStop.x = startStop.y+1; startStop.y += yFaces;
    transferStartStopIndexes.push_back(startStop);
    //4: z = 0
    startStop.x = startStop.y+1; startStop.y += zFaces;
    transferStartStopIndexes.push_back(startStop);
    //5: z = max face
    startStop.x = startStop.y+1; startStop.y += zFaces;
    transferStartStopIndexes.push_back(startStop);

    //6: x = 0, y = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.z;
    transferStartStopIndexes.push_back(startStop);
    //7: x = 0, y = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.z;
    transferStartStopIndexes.push_back(startStop);
    //8: x = 0, z = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.y;
    transferStartStopIndexes.push_back(startStop);
    //9: x = 0, z = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.y;
    transferStartStopIndexes.push_back(startStop);
    //10: x = max, y = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.z;
    transferStartStopIndexes.push_back(startStop);
    //11: x = max, y = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.z;
    transferStartStopIndexes.push_back(startStop);
    //12: x = max, z = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.y;
    transferStartStopIndexes.push_back(startStop);
    //13: x = max, z = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.y;
    transferStartStopIndexes.push_back(startStop);
    //14: y = 0, z = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.x;
    transferStartStopIndexes.push_back(startStop);
    //15: y = 0, z = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.x;
    transferStartStopIndexes.push_back(startStop);
    //16: y = max, z = 0  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.x;
    transferStartStopIndexes.push_back(startStop);
    //17: y = max, z = max  edge
    startStop.x = startStop.y+1; startStop.y += latticeSites.x;
    transferStartStopIndexes.push_back(startStop);

    //18: x = 0, y = 0, z=0 corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //19: x = 0, y = 0, z=max corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //20: x = 0, y = max, z=0 corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //21: x = 0, y = max, z=max corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //22: x = max, y = 0, z=0 corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //23: x = max, y = 0, z=max corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //24: x = max, y = max, z=0 corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);
    //25: x = max, y = max, z=max corner
    startStop.x = startStop.y+1; startStop.y += 1;
    transferStartStopIndexes.push_back(startStop);

    //printf("number of entries: %i\n",transferStartStopIndexes.size());
    //for (int ii = 0; ii < transferStartStopIndexes.size(); ++ii)
    //    printf("%i, %i\n", transferStartStopIndexes[ii].x,transferStartStopIndexes[ii].y);
    intTransferBufferSend.resize(startStop.y);
    intTransferBufferReceive.resize(startStop.y);
    doubleTransferBufferSend.resize(DIMENSION*startStop.y);
    doubleTransferBufferReceive.resize(DIMENSION*startStop.y);
    }

int multirankQTensorLatticeModel::getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType)
    {
    if(stencilType==0)
        {
        neighs = 6;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        if(!sliceSites)
            {
            int3 pos = latticeIndex.inverseIndex(target);
            neighbors[0] = positionToIndex(pos.x-1,pos.y,pos.z);
            neighbors[1] = positionToIndex(pos.x+1,pos.y,pos.z);
            neighbors[2] = positionToIndex(pos.x,pos.y-1,pos.z);
            neighbors[3] = positionToIndex(pos.x,pos.y+1,pos.z);
            neighbors[4] = positionToIndex(pos.x,pos.y,pos.z-1);
            neighbors[5] = positionToIndex(pos.x,pos.y,pos.z+1);
            }
        return target;
        };
    if(stencilType==1) //very wrong at the moment
        {
        neighs = 18;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        int3 pos = latticeIndex.inverseIndex(target);
        neighbors[0] = positionToIndex(pos.x-1,pos.y,pos.z);
        neighbors[1] = positionToIndex(pos.x+1,pos.y,pos.z);
        neighbors[2] = positionToIndex(pos.x,pos.y-1,pos.z);
        neighbors[3] = positionToIndex(pos.x,pos.y+1,pos.z);
        neighbors[4] = positionToIndex(pos.x,pos.y,pos.z-1);
        neighbors[5] = positionToIndex(pos.x,pos.y,pos.z+1);

        neighbors[6] = positionToIndex(pos.x-1,pos.y-1,pos.z);
        neighbors[7] = positionToIndex(pos.x-1,pos.y+1,pos.z);
        neighbors[8] = positionToIndex(pos.x-1,pos.y,pos.z-1);
        neighbors[9] = positionToIndex(pos.x-1,pos.y,pos.z+1);
        neighbors[10] = positionToIndex(pos.x+1,pos.y-1,pos.z);
        neighbors[11] = positionToIndex(pos.x+1,pos.y+1,pos.z);
        neighbors[12] = positionToIndex(pos.x+1,pos.y,pos.z-1);
        neighbors[13] = positionToIndex(pos.x+1,pos.y,pos.z+1);
        neighbors[14] = positionToIndex(pos.x,pos.y-1,pos.z-1);
        neighbors[15] = positionToIndex(pos.x,pos.y-1,pos.z+1);
        neighbors[16] = positionToIndex(pos.x,pos.y+1,pos.z-1);
        neighbors[17] = positionToIndex(pos.x,pos.y+1,pos.z+1);

        return target;
        }

    return target; //nope
    };
