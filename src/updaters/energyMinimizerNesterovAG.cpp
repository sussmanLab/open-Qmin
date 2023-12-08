#include"energyMinimizerNesterovAG.h"
#include"energyMinimizerNesterovAG.cuh"
#include "utilities.cuh"

/*! \file energyMinimizerNesterovAG.cpp */

energyMinimizerNesterovAG::energyMinimizerNesterovAG(shared_ptr<simpleModel> system)
    {
    setModel(system);
    initializeParameters();
    initializeFromModel();
    };

/*!
Initialize the minimizer with some default parameters. that do not depend on Ndof
*/
void energyMinimizerNesterovAG::initializeParameters()
    {
    dotProductTuner = make_shared<kernelTuner>(64,512,32,5,200000);
    minimizationTuner= make_shared<kernelTuner>(64,512,32,5,200000);
    sumReductions.resize(3);
    iterations = 0;
    forceMax = 100.;
    setNesterovAGParameters();
    setMaximumIterations(1000);
    setGPU(false);
    lambda  = 1.;
    scheduledMomentum = false;
    };

void energyMinimizerNesterovAG::initializeFromModel()
    {
    Ndof = model->getNumberOfParticles();
    sumReductionIntermediate.resize(Ndof);
    sumReductionIntermediate2.resize(Ndof);
    alternateSequence = model->returnPositions();
    };

void energyMinimizerNesterovAG::nesterovStepGPU()
    {
    sim->computeForces();
    {
    ArrayHandle<dVec> negativeGrad(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<dVec> positions(model->returnPositions(),access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> altPos(alternateSequence,access_location::device,access_mode::readwrite);

    minimizationTuner->begin();
    int blockSize = minimizationTuner->getParameter();
    gpu_nesterovAG_step(negativeGrad.data,
                  positions.data,
                  altPos.data,
                  deltaT,
                  mu,
                  Ndof,
                  blockSize);
    minimizationTuner->end();

    ArrayHandle<scalar> d_intermediate(sumReductionIntermediate,access_location::device,access_mode::overwrite);
    ArrayHandle<scalar> d_intermediate2(sumReductionIntermediate2,access_location::device,access_mode::overwrite);
    ArrayHandle<scalar> d_assist(sumReductions,access_location::device,access_mode::overwrite);
    dotProductTuner->begin();
    int maxBlockSize  = dotProductTuner->getParameter();
    gpu_dVec_dot_products(negativeGrad.data,negativeGrad.data,d_intermediate.data,d_intermediate2.data,d_assist.data,0,Ndof,maxBlockSize);
    dotProductTuner->end();
    }
    ArrayHandle<scalar> h_assist(sumReductions,access_location::host,access_mode::read);
    scalar forceNorm = h_assist.data[0];
    forceMax = sqrt(forceNorm) / (scalar)Ndof;
    }

void energyMinimizerNesterovAG::nesterovStepCPU()
    {
    sim->computeForces();
    forceMax = 0.0;
    scalar forceNorm = 0.0;
    ArrayHandle<dVec> negativeGrad(model->returnForces());
    ArrayHandle<dVec> positions(model->returnPositions());
    ArrayHandle<dVec> altPos(alternateSequence);
    dVec oldAltPos;
    for (int nn = 0; nn < Ndof;++nn)
        {
        forceNorm += dot(negativeGrad.data[nn],negativeGrad.data[nn]);
        oldAltPos = altPos.data[nn];
        altPos.data[nn] = positions.data[nn] + deltaT*negativeGrad.data[nn];
        positions.data[nn] = altPos.data[nn] +mu*(altPos.data[nn] - oldAltPos);
        }
    forceMax = sqrt(forceNorm)/Ndof;
    };

void energyMinimizerNesterovAG::minimize()
    {
    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();
    forceMax = 110.0;
    while( (iterations < maxIterations) && (forceMax > forceCutoff) )
        {
        scalar oldLambda = lambda;
        lambda = 0.5*(1.0+sqrt(1.0+4.0*oldLambda*oldLambda));
        gamma = (1 - oldLambda)/lambda;
        if(scheduledMomentum)
            {
            mu = 1.+gamma;
            //printf("momentum at %f\n",mu);
            };
        iterations +=1;
        if(useGPU)
            nesterovStepGPU();
        else
            nesterovStepCPU();
        if(iterations%1000 == 999)
            printf("nesterov step %i max force:%.3g\n",iterations,forceMax);
        };
            printf("nesterov finished: step %i max force:%.3g\n",iterations,forceMax);
    }
