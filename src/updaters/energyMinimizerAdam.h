#ifndef energyMinimizerAdam_H
#define energyMinimizerAdam_H

#include "equationOfMotion.h"
/*! \file energyMinimizerAdam.h */

class energyMinimizerAdam : public equationOfMotion
    {
    public:
        virtual void initializeFromModel();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();
        //!The "intergate equatios of motion just calls minimize
        virtual void performUpdate(){minimize();};
        void setAdamParameters(scalar b1 = 0.9, scalar b2 = 0.99, scalar eps = 0.00000001,scalar _dt = 0.0001,scalar fc = 1e-12)
            {
            beta1 = b1;
            beta1t=b1;
            beta2=b2;
            beta2t=b2;
            epsilon=eps;
            deltaT=_dt;
            setForceCutoff(fc);
            };

        //!Return the maximum force
        scalar getMaxForce(){return forceMax;};
        //!Set the maximum number of iterations before terminating (or set to -1 to ignore)
        void setMaximumIterations(int maxIt){maxIterations = maxIt;};
        //!Set the force cutoff
        void setForceCutoff(scalar fc){forceCutoff = fc;};

    protected:
        void adamStepCPU();
        void adamStepGPU();
        scalar beta1;
        scalar beta2;
        scalar beta1t;
        scalar beta2t;
        scalar epsilon;
        scalar forceMax;
        //!The number of iterations performed
        int iterations;
        //!The maximum number of iterations allowed
        int maxIterations;
        //!The cutoff value of the maximum force
        scalar forceCutoff;
        scalar alpha;
        GPUArray<dVec> biasedMomentumEstimate;
        GPUArray<dVec> biasedMomentumSquaredEstimate;
        GPUArray<dVec> correctedMomentumEstimate;
        GPUArray<dVec> correctedMomentumSquaredEstimate;
    };
#endif
