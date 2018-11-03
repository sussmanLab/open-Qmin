#include "energyMinimizerFIRE.h"
#include "energyMinimizerFIRE.cuh"
#include "utilities.cuh"

/*! \file energyMinimizerFIRE.cpp
 */

/*!
Initialize the minimizer with a reference to a target system, set a bunch of default parameters.
Of note, the current default is CPU operation
*/
energyMinimizerFIRE::energyMinimizerFIRE(shared_ptr<simpleModel> system)
    {
    setModel(system);
    initializeParameters();
    initializeFromModel();
    };

/*!
Initialize the minimizer with some default parameters. that do not depend on Ndof
*/
void energyMinimizerFIRE::initializeParameters()
    {
    dotProductTuner = make_shared<kernelTuner>(64,512,32,5,200000);
    sumReductions.resize(3);
    iterations = 0;
    Power = 0;
    NSinceNegativePower = 0;
    forceMax = 100.;
    setMaximumIterations(1000);
    setForceCutoff(1e-7);
    setAlphaStart(0.99);
    setDeltaT(0.01);
    setDeltaTMax(.1);
    setDeltaTInc(1.05);
    setDeltaTDec(0.95);
    setAlphaDec(.9);
    setNMin(5);
    setGPU(false);
    alphaMin = 0.0;
    };


/*!
Initialize the minimizer with some default parameters.
\pre requires a Simple2DModel (to set N correctly) to be already known
*/
void energyMinimizerFIRE::initializeFromModel()
    {
    Ndof = model->getNumberOfParticles();
    displacement.resize(Ndof);
    sumReductionIntermediate.resize(Ndof);
    };

/*!
 * Call the correct FIRE step routine
 */
void energyMinimizerFIRE::fireStep()
    {
    if (useGPU)
        fireStepGPU();
    else
        fireStepCPU();
    };

/*!
 * Perform a FIRE minimization step on the GPU
 */
void energyMinimizerFIRE::fireStepGPU()
    {
    Power = 0.0;
    forceMax = 0.0;
    {//array handle scope
    ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<scalar> d_intermediate(sumReductionIntermediate,access_location::device,access_mode::overwrite);
    {//scope for reduction / assist array
    ArrayHandle<scalar> d_assist(sumReductions,access_location::device,access_mode::overwrite);
    dotProductTuner->begin();
    int maxBlockSize = dotProductTuner->getParameter();
    gpu_dVec_dot_products(d_f.data,d_f.data,d_intermediate.data,d_assist.data,0,Ndof,maxBlockSize);
    gpu_dVec_dot_products(d_f.data,d_v.data,d_intermediate.data,d_assist.data,1,Ndof,maxBlockSize);
    gpu_dVec_dot_products(d_v.data,d_v.data,d_intermediate.data,d_assist.data,2,Ndof,maxBlockSize);
    dotProductTuner->end();
    }//scope for reduction array
    ArrayHandle<scalar> h_assist(sumReductions,access_location::host,access_mode::read);
    scalar forceNorm = h_assist.data[0];
    Power = h_assist.data[1];
    scalar velocityNorm = h_assist.data[2];
    forceMax = sqrt(forceNorm) / (scalar)Ndof;
    scalar scaling = 0.0;
    if(forceNorm > 0.)
        scaling = sqrt(velocityNorm/forceNorm);
    gpu_update_velocity_FIRE(d_v.data,d_f.data,alpha,scaling,Ndof);
    };//end handle scope

    //check how the power is doing
    if (Power > 0)
        {
        if (NSinceNegativePower > NMin)
            {
            deltaT = min(deltaT*deltaTInc,deltaTMax);
            alpha = alpha * alphaDec;
            alpha = max(alpha, alphaMin);
            };
        NSinceNegativePower += 1;
        }
    else
        {
        NSinceNegativePower = 0;
        deltaT = deltaT*deltaTDec;
        deltaT = max (deltaT,deltaTMin);
        alpha = alphaStart;
        ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::overwrite);
        gpu_zero_array(d_v.data,Ndof);
        };
    };

/*!
 * Perform a FIRE minimization step on the CPU
 */
void energyMinimizerFIRE::fireStepCPU()
    {
    Power = 0.0;
    forceMax = 0.0;
    {//scope for array handles
    //calculate the power, and precompute norms of vectors
    ArrayHandle<dVec> h_f(model->returnForces());
    ArrayHandle<dVec> h_v(model->returnVelocities());
    scalar forceNorm = 0.0;
    scalar velocityNorm = 0.0;
    for (int i = 0; i < Ndof; ++i)
        {
        Power += dot(h_f.data[i],h_v.data[i]);
        scalar fdot = dot(h_f.data[i],h_f.data[i]);
//        if (fdot > forceMax) forceMax = fdot;
        forceNorm += fdot;
        velocityNorm += dot(h_v.data[i],h_v.data[i]);
        };
    forceMax = sqrt(forceNorm) / (scalar)Ndof;
    scalar scaling = 0.0;
    if(forceNorm > 0.)
        scaling = sqrt(velocityNorm/forceNorm);
    //adjust the velocity according to the FIRE algorithm
    #include "ompParallelLoopDirective.h"
    for (int i = 0; i < Ndof; ++i)
        {
        h_v.data[i] = (1.0-alpha)*h_v.data[i] + alpha*scaling*h_f.data[i];
        };
    };

    if (Power > 0)
        {
        if (NSinceNegativePower > NMin)
            {
            deltaT = min(deltaT*deltaTInc,deltaTMax);
            alpha = alpha * alphaDec;
            alpha = max(alpha, alphaMin);
            };
        NSinceNegativePower += 1;
        }
    else
        {
        NSinceNegativePower = 0;
        deltaT = deltaT*deltaTDec;
        deltaT = max (deltaT,deltaTMin);
        alpha = alphaStart;
        ArrayHandle<dVec> h_v(model->returnVelocities());
        for (int i = 0; i < Ndof; ++i)
            {
            h_v.data[i] = make_dVec(0.0);
            };
        };
    };

/*!
 * Perform a FIRE minimization step on the CPU...attempts to get attain:
 * (1/N)\sum_i|f_i|^2 < forceCutoff
 */
void energyMinimizerFIRE::minimize()
    {
    cout << "attempting a minimization" << endl;
    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();
    //initialize the forces?
    sim->computeForces();
    forceMax = 110.0;
    while( (iterations < maxIterations) && (forceMax > forceCutoff) )
        {
        iterations +=1;
        integrateEquationOfMotion();
        fireStep();
        if(iterations%1000 == 999)
            printf("step %i max force:%.3g \tpower: %.3g\t alpha %.3g\t dt %g \n",iterations,forceMax,Power,alpha,deltaT);
        };
        printf("fire finished: step %i max force:%.3g \tpower: %.3g\t alpha %.3g\t dt %g \n",iterations,forceMax,Power,alpha,deltaT);
    };


void energyMinimizerFIRE::setFIREParameters(scalar deltaT, scalar alphaStart, scalar deltaTMax, scalar deltaTInc, scalar deltaTDec, scalar alphaDec, int nMin, scalar forceCutoff, scalar _alphaMin)
    {
    setDeltaT(deltaT);
    setAlphaStart(alphaStart);
    setDeltaTMax(deltaTMax);
    setDeltaTInc(deltaTInc);
    setDeltaTDec(deltaTDec);
    setAlphaDec(alphaDec);
    setNMin(nMin);
    setForceCutoff(forceCutoff);
    alphaMin = _alphaMin;
    };
