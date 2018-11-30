#ifndef energyMinimizerLoLBFGS_H
#define energyMinimizerLoLBFGS_H

#include "equationOfMotion.h"
#include "kernelTuner.h"
/*! \file energyMinimizerLoLBFGS.h */
//!Implement energy minimization via a laughable version of LBFGS
/*!
the "online" formalism of LBFGS ("A Stochastic Quasi-Newton Method for Online Convex Optimization",
Nicol N. Schraudolph, Jin Yu, Simon Gunter, 2007) so that there is no line search, but without the "online"
part related to stochastic gradient estimates
*/
class energyMinimizerLoLBFGS : public equationOfMotion
    {
    public:
        //!The basic constructor
        energyMinimizerLoLBFGS(){initializeParameters();};
        //!The basic constructor that feeds in a target system to minimize
        energyMinimizerLoLBFGS(shared_ptr<simpleModel> system);

        //!Sets a bunch of default parameters that do not depend on the number of degrees of freedom
        virtual void initializeParameters();
        virtual void initializeFromModel();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();
        //!The "intergate equatios of motion just calls minimize
        virtual void performUpdate(){minimize();};
        void setLoLBFGSParameters(int _m=5, scalar _dt = 0.0001,scalar _c = 1.00, scalar fc = 1e-12,scalar _tau=10)
            {
            c = _c;
            deltaT=_dt;
            eta=deltaT;
            tau = _tau;
            setForceCutoff(fc);
            if(_m > m)
                {
                m=_m;
                alpha.resize(m);
                sDotY.resize(m);
                gradientDifference.resize(m);
                secantEquation.resize(m);
                initializeFromModel();
                }
            };

        //!Return the maximum force
        scalar getMaxForce(){return forceMax;};
        //!Set the force cutoff
        void setForceCutoff(scalar fc){forceCutoff = fc;};

        bool scheduledMomentum = false;
    protected:
        void LoLBFGSStepCPU();
        void LoLBFGSStepGPU();

        //!the number of past gradients, etc., to save
        int m=0;
        //!a scalaing factor
        scalar c;
        //!the maximum value of the force
        scalar forceMax;
        //!The cutoff value of the maximum force
        scalar forceCutoff;

        //! gain parameter
        scalar eta;
        scalar tau;
        //! hmm...
        int currentIterationInMLoop;

        //!vector of alpha values
        GPUArray<scalar> alpha;
        //!vector of s\cdot y terms
        GPUArray<scalar> sDotY;

        //!the unscaled version of the step size
        GPUArray<dVec> unscaledStep;
        //!vector of GPUArray of gradient differences
        vector<GPUArray<dVec> > gradientDifference;
        //!vector of GPUArray of steps in trajectory space
        vector<GPUArray<dVec> > secantEquation;

        //!Utility array for simple reductions
        GPUArray<scalar> sumReductionIntermediate;
        GPUArray<scalar> sumReductionIntermediate2;
        //!Utility array for simple (sum or dot product) reductions
        GPUArray<scalar> reductions;

        //!kernel tuner for performance
        shared_ptr<kernelTuner> dotProductTuner;
        shared_ptr<kernelTuner> minimizationTuner;
    };
#endif
