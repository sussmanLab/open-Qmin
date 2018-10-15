#ifndef landauDeGennesLC_H
#define landauDeGennesLC_H

#include "baseLatticeForce.h"
/*! \file landauDeGennesLC.h */

enum class distortionEnergyType {oneConstant,twoConstant,threeConstant};

//!A landau-de gennes  q-tensor framework force computer...currently working with the one-constant approximation for the distortion term
class landauDeGennesLC : public baseLatticeForce
    {
    public:

        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1);
        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2);
        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2, scalar _L3);

        //select the force routing based on the number of elastic constants
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true)
            {
            switch (numberOfConstants)
                {
                case distortionEnergyType::oneConstant :
                    computeForceOneConstantGPU(forces,zeroOutForce);
                    break;
                case distortionEnergyType::twoConstant :
                    computeForceOneConstantGPU(forces,zeroOutForce);
                    break;
                case distortionEnergyType::threeConstant :
                    computeForceOneConstantGPU(forces,zeroOutForce);
                    break;
                };
            if(lattice->boundaries.size() >0)
                {
                computeBoundaryForcesGPU(forces,false);
                };
            };
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true)
            {
            switch (numberOfConstants)
                {
                case distortionEnergyType::oneConstant :
                    computeForceOneConstantCPU(forces,zeroOutForce);
                    break;
                case distortionEnergyType::twoConstant :
                    computeForceOneConstantCPU(forces,zeroOutForce);
                    break;
                case distortionEnergyType::threeConstant :
                    computeForceOneConstantCPU(forces,zeroOutForce);
                    break;
                };
            if(lattice->boundaries.size() >0)
                {
                computeBoundaryForcesCPU(forces,false);
                };
            };

        virtual void computeBoundaryForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce);
        virtual void computeBoundaryForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce);

        virtual void computeForceOneConstantCPU(GPUArray<dVec> &forces,bool zeroOutForce);
        virtual void computeForceOneConstantGPU(GPUArray<dVec> &forces,bool zeroOutForce);

        virtual void computeForceTwoConstantCPU(GPUArray<dVec> &forces,bool zeroOutForce){};
        virtual void computeForceTwoConstantGPU(GPUArray<dVec> &forces,bool zeroOutForce){};

        virtual void computeForceThreeConstantCPU(GPUArray<dVec> &forces,bool zeroOutForce){};
        virtual void computeForceThreeConstantGPU(GPUArray<dVec> &forces,bool zeroOutForce){};

        virtual void computeEnergyCPU();
        virtual void computeEnergyGPU(){computeEnergyCPU();};//NOT DONE YET;

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
