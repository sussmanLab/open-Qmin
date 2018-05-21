#ifndef adamMinimizer_H
#define adamMinimizer_H

#include "equationOfMotion.h"
/*! \file adamMinimizer.h */

class adamMinimizer : public equationOfMotion
    {
    public:
        virtual void integrateEOMGPU();
        virtual void integrateEOMCPU();

        virtual void initializeFromModel();

        void setAdamParameters(scalar b1 = 0.9, scalar b2 = 0.999, scalar eps = 0.00000001,scalar a = 0.001)
            {
            beta1 = b1;
            beta1t=b1;
            beta2=b2;
            beta2t=b2;
            epsilon=eps;
            alpha=a;
            };

        //!Return the maximum force
        scalar getMaxForce(){return forceMax;};

    protected:
        scalar beta1;
        scalar beta2;
        scalar beta1t;
        scalar beta2t;
        scalar epsilon;
        scalar forceMax;
        scalar alpha;
        GPUArray<dVec> biasedMomentumEstimate;
        GPUArray<dVec> biasedMomentumSquaredEstimate;
        GPUArray<dVec> correctedMomentumEstimate;
        GPUArray<dVec> correctedMomentumSquaredEstimate;
    };
#endif
