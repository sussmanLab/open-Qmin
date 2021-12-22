#ifndef multirankQTensorLatticeModel_H
#define multirankQTensorLatticeModel_H

#include "qTensorLatticeModel.h"
#include <functional>
#include <mpi.h>
/*! \file multirankQTensorLatticeModel.h */

/*!
 In the nomenclature of multi-rank models, the "type" of lattice sites corresponds to:
-2 --> sites that might be communicated to other ranks (this value can be overwritten)
-1 --> sites that border at least one boundary (colloid, wall, etc.)
0 --> bulk sites
[positive number] --> sites that are part of a boundary object
*/
class multirankQTensorLatticeModel : public qTensorLatticeModel
    {
    public:
        multirankQTensorLatticeModel(int lx, int ly, int lz, bool _xHalo, bool _yHalo, bool _zHalo, bool _useGPU = false, bool _neverGPU=false);

        //! N is the number of sites controlled, totalSites includes halo sites
        int totalSites;
        bool xHalo;
        bool yHalo;
        bool zHalo;

        int myRank;

        //When a simulation sets the configuration, the position of the local origin relative to the global coordinate system is shared with the model
        int3 latticeMinPosition;

        //! list of start/stop elements in the transfer arrays for the halo sites
        vector<int2> transferStartStopIndexes;


        //! randomly set Q tensors to correspond to a field of directors of some S0. If globallyAligned = false, each lattice site is set separately, if globallyAligned = true all will point in the same (random) direction.
        void setRandomDirectors(noiseSource &noise, scalar s0, bool globallyAligned = false);
        //!Set every lattice site to a Q tensor corresponding to the same target director and s0 value
        void setUniformDirectors(scalar3 targetDirector, scalar s0);

        //!pass a function which sets the director (first three components) and the local s0 value (fourth component) as a function of x, y, z....
        /*!
        The format should be something like
        scalar4 func(scalar x, scalar y, scalar z) {
                    scalar4 ans;
                    ans.x = nx; ans.y=ny; ans.z=nz; ans.w = s0;
                    return ans;};
        setDirectorFromFunction(&func);
        */
        void setDirectorFromFunction(std::function<scalar4(scalar,scalar,scalar)> func);
        

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
        virtual int positionToIndex(int3 &pos);
        virtual int positionToIndex(int px, int py, int pz)
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
