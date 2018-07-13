#ifndef liquidCrystalElasticity_H
#define liquidCrystalElasticity_H

#include "baseLatticeForce.h"
#include "qTensorLatticeModel.h"
/*! \file liquidCrystalElasticity */

//!phenomenological Landau-de Gennes Q-tensor stuff
class liquidCrystalElasticity : public baseLatticeForce
    {
    public:
        liquidCrystalElasticity();

        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        virtual void computeEnergyCPU(){printf("code not written\n");baseLatticeForce::computeEnergyCPU();};
        virtual void computeEnergyGPU(){printf("gpu energy calculation of lattice model being done on the cpu");energy = 0.0;baseLatticeForce::computeEnergyGPU();};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<qTensorLatticeModel> _model){qLattice = _model;lattice=_model;model = _model;};

        //!Set all of the elastic constants
        void setElasticConstants(scalar _L1, scalar _L2, scalar _L3){};

    protected:
        shared_ptr<qTensorLatticeModel> qLattice;

        //elastic constants:
    };
#endif
