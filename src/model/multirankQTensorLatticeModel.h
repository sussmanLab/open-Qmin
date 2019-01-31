#ifndef multirankQTensorLatticeModel_H
#define multirankQTensorLatticeModel_H

#include "qTensorLatticeModel.h"
#include <mpi.h>
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
        int3 latticeSites;

        //! list of start/stop elements in the transfer arrays for the halo sites
        vector<int2> transferStartStopIndexes;

        GPUArray<int> intTransferBufferSend;
        GPUArray<scalar> doubleTransferBufferSend;
        GPUArray<int> intTransferBufferReceive;
        GPUArray<scalar> doubleTransferBufferReceive;

        virtual scalar getClassSize()
            {
            scalar thisClassSize = 0.000000001*(sizeof(scalar)*(doubleTransferBufferSend.getNumElements()+doubleTransferBufferReceive.getNumElements()) + 
            3*sizeof(bool)
            +(3+ intTransferBufferSend.getNumElements()+intTransferBufferReceive.getNumElements() + 2*transferStartStopIndexes.size())*sizeof(int));
            return thisClassSize + qTensorLatticeModel::getClassSize();
            }
        void determineBufferLayout();

        //! given an  0 <= index < totalSites, return the local lattice position
        int3 indexToPosition(int idx);
        //!given a local lattice position, return the index in the expanded data arrays
        int positionToIndex(int3 &pos);
        int positionToIndex(int px, int py, int pz)
            {
            int3 temp; temp.x = px; temp.y = py; temp.z=pz;
            return positionToIndex(temp);
            };
        void getBufferInt3FromIndex(int idx, int3 &pos, int directionType, bool sending);

        //!Fill the appropriate part of the sending buffer...if GPU, fill it all in one function call
        void prepareSendingBuffer(int directionType = -1);
        //!Fill the appropriate part of data from the receiving  buffer...if GPU, fill it all in one function call
        void readReceivingBuffer(int directionType = -1);

        //!this implementation knows that extra neighbors are after N in the data arrays
        virtual int getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType = 0);
    };
typedef shared_ptr<multirankQTensorLatticeModel> MConfigPtr;
typedef weak_ptr<multirankQTensorLatticeModel> WeakMConfigPtr;
#endif
