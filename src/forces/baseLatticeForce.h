#ifndef baseLatticeForce_H
#define baseLatticeForce_H

#include "baseForce.h"
#include "cubicLattice.h"
/*! \file baseLatticeForce.h */

//!A lattice-based force specialized to lattices (which support getNiehgbor function)
class baseLatticeForce : public force
    {
    public:
        baseLatticeForce();
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true)
            {
            if(useGPU)
                computeForceGPU(forces,zeroOutForce);
            else
                computeForceCPU(forces,zeroOutForce);
            };
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        void setJ(scalar _j){J=_j;};

        virtual scalar computeEnergy()
            {
            if(useGPU)
                computeEnergyGPU();
            else
                computeEnergyCPU();
            return energy;
            };
        virtual void computeEnergyCPU();
        virtual void computeEnergyGPU(){UNWRITTENCODE("gpu energy calculation of lattice model");};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<cubicLattice> _model){lattice=_model;model = _model;};
    protected:
        shared_ptr<cubicLattice> lattice;
        //!if all lattice interactions are uniform
        scalar J;
    };

#endif
