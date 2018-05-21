#include"adamMinimizer.h"
//#include"adamMinimizer.cuh"
//#include "utilities.cuh"

/*! \file adamMinimizer.cpp */

void adamMinimizer::initializeFromModel()
    {
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

void adamMinimizer::integrateEOMCPU()
    {
    sim->computeForces();
    {
    beta1t *= beta1;
    beta2t *= beta2;
    ArrayHandle<dVec> negativeGrad(model->returnForces());
    ArrayHandle<dVec> m(biasedMomentumEstimate);
    ArrayHandle<dVec> v(biasedMomentumSquaredEstimate);
    ArrayHandle<dVec> mc(correctedMomentumEstimate);
    ArrayHandle<dVec> vc(correctedMomentumSquaredEstimate);
    ArrayHandle<dVec> disp(displacement);
    forceMax = 0.0;
    for (int nn = 0; nn < Ndof;++nn)
        {
        if(norm(negativeGrad.data[nn])>forceMax)
            forceMax = norm(negativeGrad.data[nn]);
        m.data[nn] = beta1*m.data[nn] + (beta1-1)*negativeGrad.data[nn];
        v.data[nn] = beta2*v.data[nn] + (1-beta2)*(negativeGrad.data[nn]*negativeGrad.data[nn]);
        mc.data[nn] = m.data[nn] *(1.0/(1.0 - beta1t));
        vc.data[nn] = v.data[nn]*(1.0/(1.0 - beta2t));
        for (int dd = 0; dd < DIMENSION; ++dd)
            {
            disp.data[nn].x[dd] = -alpha*mc.data[nn].x[dd]/(sqrt(vc.data[nn].x[dd]) + epsilon);
            }
        }
    }
    model->moveParticles(displacement);
    }

void adamMinimizer::integrateEOMGPU()
    {
    UNWRITTENCODE("adam gpu stuff");
    }
