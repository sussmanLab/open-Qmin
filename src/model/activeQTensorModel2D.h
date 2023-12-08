#ifndef activeQTensorModel2D_H
#define activeQTensorModel2D_H

#include "qTensorLatticeModel2D.h"
/*
no GPU code written yet... when it is, include this file:
#include "activeQTensorModel2D.cuh"
*/

/*! \file 2DTensorLatticeModel.h */

//! Each site on the underlying lattice gets a local Q-tensor
/*!

*/

class activeQTensorModel2D : public qTensorLatticeModel2D 
    {
    public:
        //! construct an underlying cubic lattice
        activeQTensorModel2D(int l,bool _useGPU = false, bool _neverGPU=false);
        activeQTensorModel2D(int lx,int ly,bool _useGPU = false, bool _neverGPU=false);


        //Add data structures relevant to an active simulation
        //! advective derivatives and pressure boundary conditions need second-nearest neighbors; create an alternate structure to hold indices of those neighbors
        Index2D alternateNeighborIndex;
        //!List of neighboring lattice sites
        GPUArray<int> alternateNeighboringSites;

        //!Definitely need the pressure as a separate data structure
        GPUArray<scalar> pressure;
        //!For now we'll carry around \Pi^S and \Pi^A...might eliminate them in favor of more arithmetic later, though.
        GPUArray<dVec> symmetricStress;
        GPUArray<dVec> antisymmetricStress;

    protected:
        void initializeDataStructures();
    };
#endif
