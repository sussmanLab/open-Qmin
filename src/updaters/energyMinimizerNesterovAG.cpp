#include"energyMinimizerNesterovAG.h"
#include"energyMinimizerNesterovAG.cuh"
#include "utilities.cuh"

/*! \file energyMinimizerNesterovAG.cpp */

void energyMinimizerNesterovAG::initializeFromModel()
    {
    iterations = 0;
    Ndof = model->getNumberOfParticles();
    alternateSequence = model->returnPositions();
    dotProductTuner = make_shared<kernelTuner>(64,512,32,5,200000);
    minimizationTuner= make_shared<kernelTuner>(64,512,32,5,200000);
    sumReductions.resize(3);
    sumReductionIntermediate.resize(Ndof);
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
    ArrayHandle<scalar> d_assist(sumReductions,access_location::device,access_mode::overwrite);
    dotProductTuner->begin();
    int maxBlockSize  = dotProductTuner->getParameter();
    gpu_dVec_dot_products(negativeGrad.data,negativeGrad.data,d_intermediate.data,d_assist.data,0,Ndof,maxBlockSize);
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
        positions.data[nn] = altPos.data[nn] + mu*(altPos.data[nn]-oldAltPos);
        }
    forceMax = sqrt(forceNorm)/Ndof;
    };

void energyMinimizerNesterovAG::minimize()
    {
    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();
    forceMax = 110.0;
    cout << "attempting minimization " <<iterations <<" out of " << maxIterations << " maximum attempts" << endl;
    while( (iterations < maxIterations) && (forceMax > forceCutoff) )
        {
        iterations +=1;
        if(useGPU)
            nesterovStepGPU();
        else
            nesterovStepCPU();
        if(iterations%1000 == 999)
            printf("nesterov step %i max force:%.3g\t energy %.3g\n",iterations,forceMax,sim->computePotentialEnergy());
        };
            printf("nesterov finished: step %i max force:%.3g\t energy %.3g\n",iterations,forceMax,sim->computePotentialEnergy());
    }
