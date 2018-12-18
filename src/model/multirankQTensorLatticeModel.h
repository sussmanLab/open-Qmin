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
        int3 expandedLatticeSites;
        Index3D expandedLatticeIndex;

        GPUArray<int> intTransferBuffer;
        GPUArray<dVec> dvecTransferBuffer;

        //!this implementation uses the expandedLatticeIndex
        virtual int getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType = 0);
    };
#endif
