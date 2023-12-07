#ifndef squareLatticeForce_H
#define squareLatticeForce_H

#include "baseForce.h"
#include "squareLattice.h"
/*! \file squareLatticeForce.h */

//!A lattice-based force specialized to lattices (which support getNeighbor function)
class squareLatticeForce : public force
    {
    public:
        squareLatticeForce();
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true, int type = 0)
            {
            if(useGPU)
                computeForceGPU(forces,zeroOutForce);
            else
                computeForceCPU(forces,zeroOutForce);
            };
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        void setJ(scalar _j){J=_j;};

        virtual scalar computeEnergy(bool verbose = false)
            {
            if(useGPU)
                computeEnergyGPU(verbose);
            else
                computeEnergyCPU(verbose);
            return energy;
            };
        virtual void computeEnergyCPU(bool verbose = false);
        virtual void computeEnergyGPU(bool verbose = false){printf("gpu energy calculation of lattice model being done on the cpu");energy = 0.0;};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<squareLattice> _model){lattice=_model;model = _model;};
        //!kernelTuner object
        shared_ptr<kernelTuner> forceTuner;

        virtual scalar getClassSize()
            {
            return  0.000000001*(sizeof(scalar)+sizeof(kernelTuner)) + force::getClassSize();
            };

    protected:
        shared_ptr<squareLattice> lattice;
        //!if all lattice interactions are uniform
        scalar J;
    };

#endif
