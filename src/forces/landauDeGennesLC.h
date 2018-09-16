#ifndef landauDeGennesLC_H
#define landauDeGennesLC_H

#include "baseLatticeForce.h"
/*! \file landauDeGennesLC.h */

enum class distortionEnergyType {oneConstant,twoConstant,threeConstant};

//!A landau-de gennes  q-tensor framework force computer...currently working with the one-constant approximation for the distortion term
class landauDeGennesLC : public baseLatticeForce
    {
    public:

        landauDeGennesLC(double _A, double _B, double _C, double _L);

        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        virtual void computeEnergyCPU();//NOT DONE YET
        virtual void computeEnergyGPU(){};//NOT DONE YET;

    protected:
        //!constants, etc.
        scalar A;
        scalar B;
        scalar C;
        scalar L1;
        scalar L2;
        scalar L3;

        distortionEnergyType numberOfConstants;
    };

#endif
