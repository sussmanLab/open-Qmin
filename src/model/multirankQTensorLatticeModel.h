#ifndef multirankQTensorLatticeModel_H
#define multirankQTensorLatticeModel_H

#include "qTensorLatticeModel.h"
/*! \file multirankQTensorLatticeModel.h */

class multirankQTensorLatticeModel : public qTensorLatticeModel
    {
    public:
        multirankQTensorLatticeModel(int lx, int ly, int lz, bool _xHalo, bool _yHalo, bool _zHalo, bool _useGPU = false);

        //! N is the number of sites controlled, totalSites includes halo sites
        int totalSites;
        bool xHalo;
        bool yHalo;
        bool zHalo;

        int myRank;
        int3 expandedLatticeSites;
        int3 latticeSites;
        Index3D expandedLatticeIndex;

        //! list of start/stop elements in the transfer arrays for the halo sites
        vector<int2> transferStartStopIndexes;

        int transferElementNumber;
        GPUArray<int> intTransferBufferSend;
        GPUArray<scalar> doubleTransferBufferSend;
        GPUArray<int> intTransferBufferReceive;
        GPUArray<scalar> doubleTransferBufferReceive;

        void determineBufferLayout();

        //!this implementation uses the expandedLatticeIndex
        virtual int getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType = 0);

        //!load the transfer buffers (type and position) with the data from the correct plane/edge/point specified by "direction type"
        virtual void prepareSendData(int directionType);
        //!assuming the tranfer buffers are full of the right data, copy them to the correct elements of positions and types
        virtual void receiveData(int directionType);

        //!map between the int3 in the expanded (base + halo) lattice frame and the 1-d index of position within the data arrays
        int indexInExpandedDataArray(int3 position);
        int indexInExpandedDataArray(int px, int py, int pz)
            {
            int3 temp; temp.x = px; temp.y = py; temp.z=pz;
            return indexInExpandedDataArray(temp);
            };

    protected:
        void parseDirectionType(int directionType, int &xyz, int &size1start, int &size2start, int &size1end, int &size2end, int &plane,bool sending);
    };
typedef shared_ptr<multirankQTensorLatticeModel> MConfigPtr;
typedef weak_ptr<multirankQTensorLatticeModel> WeakMConfigPtr;
#endif
