#define ENABLE_CUDA

#include "energyMinimizerFIRE.h"
//#include "energyMinimizerFIRE.cuh"
//#include "utilities.cuh"

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
Initialize the minimizer with some default parameters. that do not depend on N
*/
void energyMinimizerFIRE::initializeParameters()
    {
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
    };


/*!
Initialize the minimizer with some default parameters.
\pre requires a Simple2DModel (to set N correctly) to be already known
*/
void energyMinimizerFIRE::initializeFromModel()
    {
    N = model->getNumberOfParticles();
    forceDotForce.resize(N);
    forceDotVelocity.resize(N);
    velocityDotVelocity.resize(N);
    displacement.resize(N);
    sumReductionIntermediate.resize(N);
    };

/*!
 * Call the correct velocity Verlet routine
 */
void energyMinimizerFIRE::velocityVerlet()
    {
    if (useGPU)
        velocityVerletGPU();
    else
        velocityVerletCPU();
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
 * Perform a velocity verlet integration step on the GPU
 */
void energyMinimizerFIRE::velocityVerletGPU()
    {
    UNWRITTENCODE("gpu velocity verlet");
    /*
    //calculated displacements and update velocities
    if (true) //scope for array handles
        {
        ArrayHandle<dVec> d_f(force,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_v(velocity,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> d_d(displacement,access_location::device,access_mode::overwrite);
        gpu_displacement_velocity_verlet(d_d.data,d_v.data,d_f.data,deltaT,N);
        gpu_update_velocity(d_v.data,d_f.data,deltaT,N);
        };
    //move particles and update forces
    model->moveDegreesOfFreedom(displacement);
    model->enforceTopology();
    model->computeForces();
    model->getForces(force);

    //update velocities again
    ArrayHandle<dVec> d_f(force,access_location::device,access_mode::read);
    ArrayHandle<dVec> d_v(velocity,access_location::device,access_mode::readwrite);
    gpu_update_velocity(d_v.data,d_f.data,deltaT,N);
    */
    };


/*!
 * Perform a velocity verlet integration step on the CPU
 */
void energyMinimizerFIRE::velocityVerletCPU()
    {
    {//scope for array handles
    ArrayHandle<dVec> h_f(model->returnForces());
    ArrayHandle<dVec> h_v(model->returnVelocities());
    ArrayHandle<dVec> h_d(displacement);
    for (int i = 0; i < N; ++i)
        {
        //update displacement
        h_d.data[i] = deltaT*h_v.data[i] + (0.5*deltaT*deltaT)*h_f.data[i];
        //do first half of velocity update
        h_v.data[i] += 0.5*deltaT*h_f.data[i];
        };
    };//handle scope
    //move particles, then update the forces
    model->moveParticles(displacement);
    sim->computeForces();

    {//array handle scope
    //update second half of velocity vector based on new forces
    ArrayHandle<dVec> h_f(model->returnForces());
    ArrayHandle<dVec> h_v(model->returnVelocities());
    for (int i = 0; i < N; ++i)
        {
        h_v.data[i] += 0.5*deltaT*h_f.data[i];
        };
    };//handle scope
    };

/*!
 * Perform a FIRE minimization step on the GPU
 */
void energyMinimizerFIRE::fireStepGPU()
    {
    UNWRITTENCODE("fire step GPU");
    /*
    Power = 0.0;
    forceMax = 0.0;
    if(true)//scope for array handles
        {
        //ArrayHandle<dVec> d_f(force,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<dVec> d_v(velocity,access_location::device,access_mode::readwrite);
        ArrayHandle<scalar> d_ff(forceDotForce,access_location::device,access_mode::readwrite);
        ArrayHandle<scalar> d_fv(forceDotVelocity,access_location::device,access_mode::readwrite);
        ArrayHandle<scalar> d_vv(velocityDotVelocity,access_location::device,access_mode::readwrite);
        gpu_dot_dVec_vectors(d_f.data,d_f.data,d_ff.data,N);
        gpu_dot_dVec_vectors(d_f.data,d_v.data,d_fv.data,N);
        gpu_dot_dVec_vectors(d_v.data,d_v.data,d_vv.data,N);
        //parallel reduction
        if (true)//scope for reduction arrays
            {
            ArrayHandle<scalar> d_intermediate(sumReductionIntermediate,access_location::device,access_mode::overwrite);
            ArrayHandle<scalar> d_assist(sumReductions,access_location::device,access_mode::overwrite);
            gpu_parallel_reduction(d_ff.data,d_intermediate.data,d_assist.data,0,N);
            gpu_parallel_reduction(d_fv.data,d_intermediate.data,d_assist.data,1,N);
            gpu_parallel_reduction(d_vv.data,d_intermediate.data,d_assist.data,2,N);
            };
        ArrayHandle<scalar> h_assist(sumReductions,access_location::host,access_mode::read);
        scalar forceNorm = h_assist.data[0];
        Power = h_assist.data[1];
        scalar velocityNorm = h_assist.data[2];
        forceMax = forceNorm / (scalar)N;
        scalar scaling = 0.0;
        if(forceNorm > 0.)
            scaling = sqrt(velocityNorm/forceNorm);
        gpu_update_velocity_FIRE(d_v.data,d_f.data,alpha,scaling,N);
        };

    if (Power > 0)
        {
        if (NSinceNegativePower > NMin)
            {
            deltaT = min(deltaT*deltaTInc,deltaTMax);
            alpha = alpha * alphaDec;
            };
        NSinceNegativePower += 1;
        }
    else
        {
        deltaT = deltaT*deltaTDec;
        alpha = alphaStart;
        ArrayHandle<dVec> d_v(velocity,access_location::device,access_mode::overwrite);
        gpu_zero_velocity(d_v.data,N);
        };
    */
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
    for (int i = 0; i < N; ++i)
        {
        Power += dot(h_f.data[i],h_v.data[i]);
        scalar fdot = dot(h_f.data[i],h_f.data[i]);
        if (fdot > forceMax) forceMax = fdot;
        forceNorm += fdot;
        velocityNorm += dot(h_v.data[i],h_v.data[i]);
        };
    scalar scaling = 0.0;
    if(forceNorm > 0.)
        scaling = sqrt(velocityNorm/forceNorm);
    //adjust the velocity according to the FIRE algorithm
    for (int i = 0; i < N; ++i)
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
            //alpha = max(alpha, 0.75);
            };
        NSinceNegativePower += 1;
        }
    else
        {
        deltaT = deltaT*deltaTDec;
        deltaT = max (deltaT,deltaTMin);
        alpha = alphaStart;
        ArrayHandle<dVec> h_v(model->returnVelocities());
        for (int i = 0; i < N; ++i)
            {
            h_v.data[i] = make_dVec(0.0);
            };
        };
    };

/*!
 * Perform a FIRE minimization step on the CPU
 */
void energyMinimizerFIRE::minimize()
    {
    cout << "attempting a minimization" << endl;
    if (N != model->getNumberOfParticles())
        initializeFromModel();
    //initialize the forces?
    sim->computeForces();
    forceMax = 110.0;
    while( (iterations < maxIterations) && (sqrt(forceMax) > forceCutoff) )
        {
        iterations +=1;
        velocityVerlet();
        fireStep();
        };
        printf("step %i max force:%.3g \tpower: %.3g\t alpha %.3g\t dt %g \n",iterations,sqrt(forceMax),Power,alpha,deltaT);
    };
