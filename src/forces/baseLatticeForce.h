#ifndef baseLatticeForce_H
#define baseLatticeForce_H

#include "baseForce.h"
#include "cubicLattice.h"
/*! \file baseLatticeForce.h */

//!A lattice-based force specialized to lattices (which support getNeighbor function)
class baseLatticeForce : public force
    {
    public:
        baseLatticeForce();
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true, int type = 0)
            {
#ifdef ENABLE_CUDA
            if(useGPU)
                computeForceGPU(forces,zeroOutForce);
            else
#endif
                computeForceCPU(forces,zeroOutForce);
            };
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        void setJ(scalar _j){J=_j;};

        virtual scalar computeEnergy(bool verbose = false)
            {
#ifdef ENABLE_CUDA
            if(useGPU)
                computeEnergyGPU(verbose);
            else
#endif
                computeEnergyCPU(verbose);
            return energy;
            };
        virtual void computeEnergyCPU(bool verbose = false);

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<cubicLattice> _model){lattice=_model;model = _model;};
        //!kernelTuner object
        shared_ptr<kernelTuner> forceTuner;

        virtual scalar getClassSize()
            {
            return  0.000000001*(sizeof(scalar)+sizeof(kernelTuner)) + force::getClassSize();
            };
#ifdef ENABLE_CUDA
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeEnergyGPU(bool verbose = false){printf("gpu energy calculation of lattice model being done on the cpu");energy = 0.0;};
#endif

    protected:
        shared_ptr<cubicLattice> lattice;
        //!if all lattice interactions are uniform
        scalar J;
    };

#endif
