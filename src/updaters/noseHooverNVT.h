#ifndef noseHooverNVT_H
#define noseHooverNVT_H

#include "equationOfMotion.h"
/*! \file noseHooverNVT.h */

//! Implements NVT dynamics according to the Nose-Hoover equations of motion with a chain of thermostats
/*!
 *This allows one to do standard NVT simulations. A chain (whose length can be specified by the user)
 of thermostats is used to maintain the target temperature. We closely follow the Frenkel & Smit
 update scheme, which is itself based on:
 Martyna, Tuckerman, Tobias, and Klein
 Mol. Phys. 87, 1117 (1996)
*/
class noseHooverNVT : public equationOfMotion
    {
    public:
        noseHooverNVT(shared_ptr<simpleModel> system,scalar _Temperature, int _nChain = 4);
        virtual void integrateEOMGPU();
        virtual void integrateEOMCPU();

        void setChainLength(int _m);
        virtual void initializeFromModel();
        GPUArray<scalar> keArray;
        GPUArray<scalar> kineticEnergyScaleFactor;
        int Nchain;

        GPUArray<scalar4> bathVariables;
        void setT(scalar _t);
        scalar temperature;

    protected:
        //!partially update bath (chain) positions and velocities... call twice per timestep
        void propagateChain();
        //!update the positions and velocities of particles
        void propagatePositionsVelocites();
        //!A structure for performing partial reductions on the gpu
        GPUArray<scalar> keIntermediateReduction;
    };
#endif

