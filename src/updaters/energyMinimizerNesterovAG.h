#ifndef energyMinimizerNesterovAG_H
#define energyMinimizerNesterovAG_H

#include "equationOfMotion.h"
#include "kernelTuner.h"
/*! \file energyMinimizerNesterovAG.h */

class energyMinimizerNesterovAG : public equationOfMotion
    {
    public:
        //!The basic constructor
        energyMinimizerNesterovAG(){initializeParameters();};
        //!The basic constructor that feeds in a target system to minimize
        energyMinimizerNesterovAG(shared_ptr<simpleModel> system);

        //!Sets a bunch of default parameters that do not depend on the number of degrees of freedom
        virtual void initializeParameters();
        virtual void initializeFromModel();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();
        //!The "intergate equatios of motion just calls minimize
        virtual void performUpdate(){minimize();};
        void setNesterovAGParameters(scalar _dt = 0.0001,scalar _mu = 0.01, scalar fc = 1e-12)
            {
            mu = _mu;
            deltaT=_dt;
            setForceCutoff(fc);
            };

        //!Return the maximum force
        scalar getMaxForce(){return forceMax;};
        //!Set the force cutoff
        void setForceCutoff(scalar fc){forceCutoff = fc;};

    protected:
        void nesterovStepCPU();
        void nesterovStepGPU();
        scalar mu;
        scalar forceMax;
        //!The cutoff value of the maximum force
        scalar forceCutoff;

        GPUArray<dVec> alternateSequence;
        //!Utility array for simple reductions
        GPUArray<scalar> sumReductionIntermediate;
        //!Utility array for simple reductions
        GPUArray<scalar> sumReductions;

        //!kernel tuner for performance
        shared_ptr<kernelTuner> dotProductTuner;
        shared_ptr<kernelTuner> minimizationTuner;
    };
#endif
