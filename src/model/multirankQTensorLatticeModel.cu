#include "multirankQTensorLatticeModel.cuh"
/*! \file multirankQTensorLatticeModel.cu */
/*!
    \addtogroup modelKernels
    @{
*/

__device__ int deviceIndexInExpandedDataArray(int3 &pos,int3 &latticeSites, int3 &expandedLatticeSites, Index3D &latticeIndex,bool xHalo, bool yHalo, bool zHalo )
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
    int base = latticeIndex.getNumElements();
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

    return -1;
    };

__global__ void gpu_mrqtlm_data_to_buffer_kernel(int *type,
                               dVec *position,
                               int *iBuf,
                               scalar *dBuf,
                               int size1start,
                               int size1end,
                               int size2start,
                               int size2end,
                               int xyz,
                               int plane,
                               int3 latticeSites,
                               int3 expandedLatticeSites,
                               Index3D latticeIndex,
                               bool xHalo,
                               bool yHalo,
                               bool zHalo)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    int size1 = size1end-size1start;
    int size2 = size2end-size2start;
    //idx = ii * size2 + jj, with possible offsets if the start is negative
    int jj = (idx % size2) + size2start;
    int ii = (idx / size2) + size1start;
    int3 lPos;
    if(xyz ==0)
        {lPos.x = plane; lPos.y = ii; lPos.z = jj;}
    else if(xyz ==1)
        {lPos.x = ii; lPos.y = plane; lPos.z = jj;}
    else if (xyz ==2)
        {lPos.x = ii; lPos.y = jj; lPos.z = plane;}
    int currentSite = deviceIndexInExpandedDataArray(lPos,latticeSites,expandedLatticeSites,latticeIndex,xHalo,yHalo,zHalo);

    iBuf[idx] = type[currentSite];
    for(int dd = 0; dd < DIMENSION; ++dd)
        dBuf[DIMENSION*idx+dd] = position[currentSite][dd];
    };

__global__ void gpu_mrqtlm_buffer_to_data_kernel(int *type,
                               dVec *position,
                               int *iBuf,
                               scalar *dBuf,
                               int size1start,
                               int size1end,
                               int size2start,
                               int size2end,
                               int xyz,
                               int plane,
                               int3 latticeSites,
                               int3 expandedLatticeSites,
                               Index3D latticeIndex,
                               bool xHalo,
                               bool yHalo,
                               bool zHalo)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
//    int size1 = size1end-size1start;
    int size2 = size2end-size2start;
    int size1 = size1end-size1start;
    if(idx >=size1*size2)
        return;
    //idx = ii * size2 + jj, with possible offsets if the start is negative
    int jj = (idx % size2) + size2start;
    int ii = (idx / size2) + size1start;
    int3 lPos;
    if(xyz ==0)
        {lPos.x = plane; lPos.y = ii; lPos.z = jj;}
    else if(xyz ==1)
        {lPos.x = ii; lPos.y = plane; lPos.z = jj;}
    else if (xyz ==2)
        {lPos.x = ii; lPos.y = jj; lPos.z = plane;}
    int currentSite = deviceIndexInExpandedDataArray(lPos,latticeSites,expandedLatticeSites,latticeIndex,xHalo,yHalo,zHalo);

    type[currentSite] = iBuf[idx];
    for(int dd = 0; dd < DIMENSION; ++dd)
        position[currentSite][dd] = dBuf[DIMENSION*idx+dd];
    };

bool gpu_mrqtlm_buffer_data_exchange(bool sending,
                               int *type,
                               dVec *position,
                               int *iBuf,
                               scalar *dBuf,
                               int size1start,
                               int size1end,
                               int size2start,
                               int size2end,
                               int xyz,
                               int plane,
                               int3 latticeSites,
                               int3 expandedLatticeSites,
                               Index3D &latticeIndex,
                               bool xHalo,
                               bool yHalo,
                               bool zHalo,
                               int blockSize)
    {
    int block_size = blockSize;
    int Size1 = size1end-size1start;
    int Size2 = size2end-size2start;
    int N = Size1*Size2;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    if(sending)
        {
        gpu_mrqtlm_data_to_buffer_kernel<<<nblocks,block_size>>>(type,position,iBuf,dBuf,size1start,size1end,size2start,size2end,
            xyz,plane,latticeSites,expandedLatticeSites,latticeIndex,xHalo,yHalo,zHalo);
        }
    else
        {
        gpu_mrqtlm_buffer_to_data_kernel<<<nblocks,block_size>>>(type,position,iBuf,dBuf,size1start,size1end,size2start,size2end,
            xyz,plane,latticeSites,expandedLatticeSites,latticeIndex,xHalo,yHalo,zHalo);
        };

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

__device__ int inferDirectionFromIndex(int i,int3 latticeSites, int &startIdx)
    {
    startIdx = 0;
    int base = latticeSites.y*latticeSites.z;
    if (i < base)
        return 0;
    startIdx = base;
    base += latticeSites.y*latticeSites.z;
    if (i < base)
        return 1;
    startIdx = base;
    base += latticeSites.x*latticeSites.z;
    if (i < base)
        return 2;
    startIdx = base;
    base += latticeSites.x*latticeSites.z;
    if (i < base)
        return 3;
    startIdx = base;
    base += latticeSites.x*latticeSites.y;
    if (i < base)
        return 4;
    startIdx = base;
    base += latticeSites.x*latticeSites.y;
    if (i < base)
        return 5;

    startIdx = base;
    base += latticeSites.z;
    if (i < base)
        return 6;
    startIdx = base;
    base += latticeSites.z;
    if (i < base)
        return 7;
    startIdx = base;
    base += latticeSites.y;
    if (i < base)
        return 8;
    startIdx = base;
    base += latticeSites.y;
    if (i < base)
        return 9;
    startIdx = base;
    base += latticeSites.z;
    if (i < base)
        return 10;
    startIdx = base;
    base += latticeSites.z;
    if (i < base)
        return 11;
    startIdx = base;
    base += latticeSites.y;
    if (i < base)
        return 12;
    startIdx = base;
    base += latticeSites.y;
    if (i < base)
        return 13;
    startIdx = base;
    base += latticeSites.x;
    if (i < base)
        return 14;
    startIdx = base;
    base += latticeSites.x;
    if (i < base)
        return 15;
    startIdx = base;
    base += latticeSites.x;
    if (i < base)
        return 16;
    startIdx = base;
    base += latticeSites.x;
    if (i < base)
        return 17;

    startIdx = base;

    return 18 + i-base;
    }

__device__ void getBufferInt3(int idx, int3 &pos,int directionType,int startIndex,int3 latticeSites)
    {
    int index = idx - startIndex;
    switch(directionType)
        {
        case 0:
            pos.z = index / latticeSites.y; pos.y = index % latticeSites.y;
            pos.x = 0;
            break;
        case 1:
            pos.z = index / latticeSites.y; pos.y = index % latticeSites.y;
            pos.x = latticeSites.x-1;
            break;
        case 2:
            pos.z = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.y =  0;
            break;
        case 3:
            pos.z = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.y =  latticeSites.y-1;
            break;
        case 4:
            pos.y = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.z =  0;
            break;
        case 5:
            pos.y = index / latticeSites.x; pos.x = index % latticeSites.x;
            pos.z =  latticeSites.z-1;
            break;
        //edges
        case 6:
            pos.x =  0;
            pos.y =  0;
            pos.z = index;
            break;
        case 7:
            pos.x =  0;
            pos.y =  latticeSites.y-1;
            pos.z = index;
            break;
        case 8:
            pos.x =  0;
            pos.z =  0;
            pos.y = index;
            break;
        case 9:
            pos.x =  0 ;
            pos.z =  latticeSites.z-1;
            pos.y = index;
            break;
        case 10:
            pos.x =  latticeSites.x-1;
            pos.y =  0;
            pos.z = index;
            break;
        case 11:
            pos.x =  latticeSites.x-1;
            pos.y =  latticeSites.y-1;
            pos.z = index;
            break;
        case 12:
            pos.x =  latticeSites.x-1;
            pos.z =  0;
            pos.y = index;
            break;
        case 13:
            pos.x =  latticeSites.x-1;
            pos.z =  latticeSites.z-1;
            pos.y = index;
            break;
        case 14:
            pos.y =  0;
            pos.z =  0;
            pos.x = index;
            break;
        case 15:
            pos.y =  0;
            pos.z =  latticeSites.z-1;
            pos.x = index;
            break;
        case 16:
            pos.y =  latticeSites.y-1;
            pos.z =  0;
            pos.x = index;
            break;
        case 17:
            pos.y =  latticeSites.y-1;
            pos.z =  latticeSites.z-1;
            pos.x = index;
            break;
        //corners
        case 18:
            pos.x =  0;
            pos.y =  0;
            pos.z =  0;
            break;
        case 19:
            pos.x =  0;
            pos.y =  0;
            pos.z =  latticeSites.z-1;
            break;
        case 20:
            pos.x =  0 ;
            pos.y =  latticeSites.y-1;
            pos.z =  0;
            break;
        case 21:
            pos.x =  0 ;
            pos.y =  latticeSites.y-1 ;
            pos.z =  latticeSites.z-1 ;
            break;
        case 22:
            pos.x =  latticeSites.x-1 ;
            pos.y =  0;
            pos.z =  0;
            break;
        case 23:
            pos.x =  latticeSites.x-1;
            pos.y =  0 ;
            pos.z =  latticeSites.z-1;
            break;
        case 24:
            pos.x =  latticeSites.x-1;
            pos.y =  latticeSites.y-1;
            pos.z =  0 ;
            break;
        case 25:
            pos.x =  latticeSites.x-1;
            pos.y =  latticeSites.y-1;
            pos.z =  latticeSites.z-1;
            break;
        }
    }

__global__ void gpu_prepareSendingBuffer_kernel(int *type,
                               dVec *position,
                               int *iBuf,
                               scalar *dBuf,
                               int3 latticeSites,
                               Index3D latticeIndex,
                               int maxIndex)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= maxIndex)
        return;

    int startIndex;
    int directionType = inferDirectionFromIndex(idx,latticeSites,startIndex);
    int3 pos;
    getBufferInt3(idx,pos,directionType,startIndex,latticeSites);
    int currentSite = latticeIndex(pos);


    iBuf[idx] = type[currentSite];
    for (int dd = 0; dd < DIMENSION; ++dd)
        dBuf[DIMENSION*idx+dd] = position[currentSite][dd];
    };

bool gpu_prepareSendingBuffer(int *type,
                            dVec *position,
                            int *iBuf,
                            scalar *dBuf,
                            int3 latticeSites,
                            Index3D latticeIndex,
                            int maxIndex,
                            int blockSize)
    {
    int block_size = blockSize;

    if (maxIndex < 128) block_size = 16;
    unsigned int nblocks  = maxIndex/block_size + 1;

    gpu_prepareSendingBuffer_kernel<<<nblocks,block_size>>>(type,position,iBuf,dBuf,latticeSites,latticeIndex,maxIndex);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

__global__ void gpu_copyReceivingBuffer_kernel(int *type,
                               dVec *position,
                               int *iBuf,
                               scalar *dBuf,
                               int N,
                               int maxIndex)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= maxIndex)
        return;
    int currentSite = idx + N;
    type[currentSite] = iBuf[idx];
    for (int dd = 0; dd < DIMENSION; ++dd)
        position[currentSite][dd] = dBuf[DIMENSION*idx+dd];
    };

bool gpu_copyReceivingBuffer(int *type,
                            dVec *position,
                            int *iBuf,
                            scalar *dBuf,
                            int N,
                            int maxIndex,
                            int blockSize)
    {
    int block_size = blockSize;

    if (maxIndex < 128) block_size = 16;
    unsigned int nblocks  = maxIndex/block_size + 1;

    gpu_copyReceivingBuffer_kernel<<<nblocks,block_size>>>(type,position,iBuf,dBuf,N,maxIndex);

    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
