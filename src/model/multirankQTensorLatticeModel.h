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
        Index3D expandedLatticeIndex;

        int transferElementNumber;
        GPUArray<int> intTransferBuffer;
        GPUArray<dVec> dvecTransferBuffer;

        //!this implementation uses the expandedLatticeIndex
        virtual int getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType = 0);

        //!load the transfer buffers (type and position) with the data from the correct plane/edge/point specified by "direction type"
        virtual void prepareSendData(int directionType);
        //!assuming the tranfer buffers are full of the right data, copy them to the correct elements of positions and types
        virtual void receiveData(int directionType);

    protected:
        void parseDirectionType(int directionType, int &xyz, int &size1, int &size2, int &plane,bool sending);
    };
#endif
