#include "energyMinimizerGradientDescent.h"
#include "utilities.cuh"

/*! \file energyMinimizerGradientDescent.cpp
 */

/*!
Initialize the minimizer with a reference to a target system, set a bunch of default parameters.
Of note, the current default is CPU operation
*/
energyMinimizerGradientDescent::energyMinimizerGradientDescent(shared_ptr<simpleModel> system)
    {
    setModel(system);
    initializeParameters();
    initializeFromModel();
    };

/*!
Initialize the minimizer with some default parameters. that do not depend on Ndof
*/
void energyMinimizerGradientDescent::initializeParameters()
    {
    dotProductTuner = make_shared<kernelTuner>(1024,1024,32,5,200000);
    iterations = 0;
    setMaximumIterations(1000);
    setForceCutoff(1e-7);
    setDeltaT(0.01);
    setGPU(false);
    updaterData.resize(1);
    };


/*!
Initialize the minimizer with some default parameters.
\pre requires a Simple2DModel (to set N correctly) to be already known
*/
void energyMinimizerGradientDescent::initializeFromModel()
    {
    Ndof = model->getNumberOfParticles();
    displacement.resize(Ndof);
    sumReductionIntermediate.resize(Ndof);
    sumReductionIntermediate2.resize(Ndof);
    };

/*!
 * Call the correct FIRE step routine
 */
void energyMinimizerGradientDescent::gradientDescentStep()
    {
    if (useGPU)
        gradientDescentGPU();
    else
        gradientDescentCPU();
    };

/*!
 * Perform a GD minimization step on the GPU
 */
void energyMinimizerGradientDescent::gradientDescentGPU()
    {
    sim->moveParticles(model->returnForces(),deltaT);
    sim->computeForces();
    scalar forceNorm = gpu_gpuarray_dVec_dot_products(model->returnForces(),model->returnForces(),
                                                sumReductionIntermediate,sumReductionIntermediate2,Ndof);
    updaterData[0] = forceNorm;
    sim->sumUpdaterData(updaterData);
    forceNorm = updaterData[0];
    forceMax = sqrt(forceNorm) / ((scalar)Ndof * sim->nRanks);
    };

/*!
 * Perform a GD minimization step on the CPU
 */
void energyMinimizerGradientDescent::gradientDescentCPU()
    {
    sim->moveParticles(model->returnForces(),deltaT);
    sim->computeForces();
    scalar forceNorm = 0.0;
    {//scope for array handles
    ArrayHandle<dVec> h_f(model->returnForces());
    //ArrayHandle<scalar> h_m(model->returnMasses());
    for (int i = 0; i < Ndof; ++i)
        {
        //update displacement
        forceNorm +=  dot(h_f.data[i],h_f.data[i]);
        };
    };//handle scope
    //move particles
    updaterData[0] = forceNorm;
    sim->sumUpdaterData(updaterData);
    forceNorm = updaterData[0];
    forceMax = sqrt(forceNorm) / ((scalar)Ndof * sim->nRanks);
    };

/*!
 * Perform a GD minimization sequence...attempts to get attain:
 * (1/N)\sum_i|f_i|^2 < forceCutoff
 */
void energyMinimizerGradientDescent::minimize()
    {
    //cout << "attempting a minimization" << endl;
    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();
    //initialize the forces?
    sim->computeForces();
    int curIterations = iterations;
    //always iterate at least once
    while((iterations < maxIterations && forceMax > forceCutoff) || iterations == curIterations)
        {
        iterations +=1;

        gradientDescentStep();
        if(iterations%1000 == 999)
            printf("step %i max force:%.3g \n",iterations,forceMax);cout.flush();
        };
        printf("gradient descent finished: step %i max force:%.3g \n",iterations,forceMax);cout.flush();
    };


void energyMinimizerGradientDescent::setGradientDescentParameters(scalar deltaT, scalar forceCutoff)
    {
    setDeltaT(deltaT);
    setForceCutoff(forceCutoff);
    };
