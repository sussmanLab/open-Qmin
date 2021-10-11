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
    dotProductTuner = make_shared<kernelTuner>(1024,1024,32,5,200000);
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
    updaterData.resize(3);
    nTotal = Ndof;
    };


/*!
Initialize the minimizer with some default parameters.
\pre requires a Simple2DModel (to set N correctly) to be already known
*/
void energyMinimizerFIRE::initializeFromModel()
    {
    //model->freeGPUArrays(false,false,false);
    Ndof = model->getNumberOfParticles();
    neverGPU = model->neverGPU;
    if(neverGPU)
        {
        displacement.noGPU = true;
        sumReductions.noGPU=true;
        sumReductionIntermediate.noGPU=true;
        sumReductionIntermediate2.noGPU=true;
        }
    //printf("FIRE dof = %i\n",Ndof);
    sumReductions.resize(3);
    displacement.resize(Ndof);
    sumReductionIntermediate.resize(Ndof);
    sumReductionIntermediate2.resize(Ndof);
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
    //
    //The forces are really ``co-forces'' as defined in the non-orthonormal basis of Qxx,Qxy,Qyy,Qxz,Qyz
    //As a result, we take the vector norm of all three quantities
    //
    scalar forceNorm = gpu_gpuarray_QT_vector_dot_product(model->returnForces(),
                                            sumReductionIntermediate,sumReductionIntermediate2,Ndof);
    scalar velocityNorm = gpu_gpuarray_QT_vector_dot_product(model->returnVelocities(),
                                                sumReductionIntermediate,sumReductionIntermediate2,Ndof);
    Power = gpu_gpuarray_QT_vector_dot_product(model->returnForces(),model->returnVelocities(),
                                                sumReductionIntermediate,sumReductionIntermediate2,Ndof);

    updaterData[0] = forceNorm;
    updaterData[1] = Power;
    updaterData[2] = velocityNorm;
    sim->sumUpdaterData(updaterData);
    forceNorm = updaterData[0];
    Power = updaterData[1];
    velocityNorm = updaterData[2];

    forceMax = sqrt(forceNorm) / ((scalar)nTotal);
    scaling = 0.0;
    if(forceNorm > 0.)
        scaling = sqrt(velocityNorm/forceNorm);
    {
    ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    gpu_update_velocity_FIRE(d_v.data,d_f.data,alpha,scaling,Ndof);
    }

    //check how the power is doing
    if (Power > 0 && iterations % 500 != 0)
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
        dVec zero(0.0);
        gpu_set_array(d_v.data,zero,Ndof,512);
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
    ArrayHandle<int> h_t(model->returnTypes(),access_location::host,access_mode::read);
    ArrayHandle<dVec> h_f(model->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(model->returnVelocities());
    scalar forceNorm = 0.0;
    scalar velocityNorm = 0.0;
    for (int i = 0; i < Ndof; ++i)
        {
        //
        //The forces are really ``co-forces'' as defined in the non-orthonormal basis of Qxx,Qxy,Qyy,Qxz,Qyz
        //As a result, we take the vector norm of all three quantities
        //
        scalar fdot  = dotVec(h_f.data[i],h_f.data[i]);
        scalar vdot  = dotVec(h_v.data[i],h_v.data[i]);
        scalar pdot  = dotVec(h_f.data[i],h_v.data[i]);
        forceNorm    += fdot ;
        velocityNorm += vdot;
        Power        += pdot;
        };

    updaterData[0] = forceNorm;
    updaterData[1] = Power;
    updaterData[2] = velocityNorm;
    sim->sumUpdaterData(updaterData);
    forceNorm = updaterData[0];
    Power = updaterData[1];
    velocityNorm = updaterData[2];

    forceMax = sqrt(forceNorm) / ((scalar)nTotal);
    //printf("fnorm = %g\t velocity norm = %g\n",forceNorm,velocityNorm);
    scaling = 0.0;
    if(forceNorm > 0.)
        scaling = sqrt(velocityNorm/forceNorm);
    //adjust the velocity according to the FIRE algorithm
    for (int i = 0; i < Ndof; ++i)
        {
        for (int dd = 0; dd < DIMENSION; ++dd)
            h_v.data[i][dd] = (1.0-alpha)*h_v.data[i][dd] + alpha*scaling*h_f.data[i][dd];
        };
    };

    if (Power > 0 && iterations % 500 != 0)
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
        }
    };

/*!
 * Perform a FIRE minimization step on the CPU...attempts to get attain:
 * (1/N)\sum_i|f_i|^2 < forceCutoff
 */
void energyMinimizerFIRE::minimize()
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
        integrateEquationOfMotion();

        fireStep();
        if(iterations%1000 == 999)
            printf("step %i max force:%.3g \tpower: %.3g\t alpha %.3g\t dt %g \t scaling %.3g \n",iterations,forceMax,Power,alpha,deltaT,scaling);cout.flush();
        };
        printf("fire finished: step %i max force:%.3g \tpower: %.3g\t alpha %.3g\t dt %g \tscaling %.3g \n",iterations,forceMax,Power,alpha,deltaT,scaling);cout.flush();
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
