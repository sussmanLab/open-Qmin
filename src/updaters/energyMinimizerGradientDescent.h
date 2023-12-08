#ifndef ENERGYMINIMIZERGD_H
#define ENERGYMINIMIZERGD_H

#include "functions.h"
#include "gpuarray.h"
#include "simpleModel.h"
#include "kernelTuner.h"
#include "equationOfMotion.h"

/*! \file energyMinimizerGradientDescent.h */
//!Implement energy minimization via the FIRE algorithm
/*!
This class uses a simple gradient descent algorithm
The class is in the same framework as the simpleEquationOfMotion class, so it is called by calling
performTimestep on a Simulation object that has been given the minimizer and the configuration
to minimize. Each timestep, though, is a complete minimization (i.e. will run for the maximum number
of iterations, or until a target tolerance has been acheived, or whatever stopping condition the user
sets.
*/
class energyMinimizerGradientDescent : public equationOfMotion
    {
    public:
        //!The basic constructor
        energyMinimizerGradientDescent(){initializeParameters();};
        //!The basic constructor that feeds in a target system to minimize
        energyMinimizerGradientDescent(shared_ptr<simpleModel> system);
        //!Sets a bunch of default parameters that do not depend on the number of degrees of freedom
        virtual void initializeParameters();
        //!Set a bunch of default initialization parameters (if the State is available to determine the size of vectors)
        void initializeFromModel();

        //!Set a lot of parameters!
        void setGradientDescentParameters(scalar deltaT, scalar forceCutoff, int maxSteps);

        //!Set the force cutoff
        void setForceCutoff(scalar fc){forceCutoff = fc;};

        //!an interface to call either the CPU or GPU FIRE algorithm
        void gradientDescentStep();
        //!Perform a velocity Verlet step on the CPU
        void gradientDescentCPU();
        //!Perform a velocity Verlet step on the GPU
        void gradientDescentGPU();

        //!Minimize to either the force tolerance or the maximum number of iterations
        void minimize();
        //!The "intergate equatios of motion just calls minimize
        virtual void performUpdate(){minimize();};

        //!Return the maximum force
        virtual scalar getMaxForce(){return forceMax;};

        virtual scalar getClassSize()
            {
            scalar thisClassSize = 0.000000001*(sizeof(scalar)*(2+sumReductionIntermediate.getNumElements()+sumReductionIntermediate2.getNumElements())
                +2*sizeof(int)+sizeof(kernelTuner));
            return thisClassSize+equationOfMotion::getClassSize();
            }

    protected:
        //!sqrt(force.force) / N_{dof}
        scalar forceMax;
        //!The cutoff value of the maximum force
        scalar forceCutoff;

        //!kernel tuner for performance
        shared_ptr<kernelTuner> dotProductTuner;
        //!Utility array for simple reductions
        GPUArray<scalar> sumReductionIntermediate;
        GPUArray<scalar> sumReductionIntermediate2;

    };
#endif
