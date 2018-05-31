#include"energyMinimizerAdam.h"
#include"energyMinimizerAdam.cuh"
#include "utilities.cuh"

/*! \file energyMinimizerAdam.cpp */

void energyMinimizerAdam::initializeFromModel()
    {
    iterations = 0;
    Ndof = model->getNumberOfParticles();
    displacement.resize(Ndof);
    biasedMomentumEstimate.resize(Ndof);
    biasedMomentumSquaredEstimate.resize(Ndof);
    correctedMomentumEstimate.resize(Ndof);
    correctedMomentumSquaredEstimate.resize(Ndof);
    vector<dVec> zeroes(Ndof,make_dVec(0.0));
    fillGPUArrayWithVector(zeroes,biasedMomentumEstimate);
    fillGPUArrayWithVector(zeroes,biasedMomentumSquaredEstimate);
    fillGPUArrayWithVector(zeroes,correctedMomentumEstimate);
    fillGPUArrayWithVector(zeroes,correctedMomentumSquaredEstimate);
    };

void energyMinimizerAdam::adamStepGPU()
    {
    sim->computeForces();
    {//array handle
    ArrayHandle<dVec> negativeGrad(model->returnForces(),access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> m(biasedMomentumEstimate,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> v(biasedMomentumSquaredEstimate,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> mc(correctedMomentumEstimate,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> vc(correctedMomentumSquaredEstimate,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> disp(displacement,access_location::device,access_mode::readwrite);
    int blockSize = 128;
    gpu_adam_step(negativeGrad.data,
                  m.data,
                  v.data,
                  mc.data,
                  vc.data,
                  disp.data,
                  deltaT,
                  beta1,
                  beta2,
                  beta1t,
                  beta2t,
                  Ndof,
                  blockSize);

    }//handle scope end
    model->moveParticles(displacement);
    //forceMax = sqrt(forceNorm)/Ndof;
    beta1t *= beta1;
    beta2t *= beta2;
    }

void energyMinimizerAdam::adamStepCPU()
    {
    sim->computeForces();
    ArrayHandle<dVec> negativeGrad(model->returnForces());
    ArrayHandle<dVec> m(biasedMomentumEstimate);
    ArrayHandle<dVec> v(biasedMomentumSquaredEstimate);
    ArrayHandle<dVec> mc(correctedMomentumEstimate);
    ArrayHandle<dVec> vc(correctedMomentumSquaredEstimate);
    ArrayHandle<dVec> disp(displacement);
    forceMax = 0.0;
    scalar forceNorm = 0.0;
    for (int nn = 0; nn < Ndof;++nn)
        {
        forceNorm += dot(negativeGrad.data[nn],negativeGrad.data[nn]);
        m.data[nn] = beta1*m.data[nn] + (beta1-1)*negativeGrad.data[nn];
        v.data[nn] = beta2*v.data[nn] + (1-beta2)*(negativeGrad.data[nn]*negativeGrad.data[nn]);
        mc.data[nn] = m.data[nn] *(1.0/(1.0 - beta1t));
        vc.data[nn] = v.data[nn]*(1.0/(1.0 - beta2t));
        for (int dd = 0; dd < DIMENSION; ++dd)
            {
            scalar rootvc = sqrt(vc.data[nn].x[dd]);
            if (rootvc == 0) rootvc = epsilon;
            disp.data[nn].x[dd] = -deltaT*mc.data[nn].x[dd]/(rootvc);
            }
        }
    model->moveParticles(displacement);
    forceMax = sqrt(forceNorm)/Ndof;
    beta1t *= beta1;
    beta2t *= beta2;
    };

void energyMinimizerAdam::minimize()
    {
    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();
    forceMax = 110.0;
    cout << "attempting minimization " <<iterations <<" out of " << maxIterations << " maximum attempts" << endl;
    while( (iterations < maxIterations) && (forceMax > forceCutoff) )
        {
        iterations +=1;
        if(useGPU)
            adamStepGPU();
        else
            adamStepCPU();
        if(iterations%1000 == 999)
            printf("step %i max force:%.3g\t energy %.3g\n",iterations,forceMax,sim->computePotentialEnergy());
        };
            printf("adam finished: step %i max force:%.3g\t energy %.3g\n",iterations,forceMax,sim->computePotentialEnergy());
    }
