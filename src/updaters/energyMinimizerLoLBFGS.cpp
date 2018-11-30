#include"energyMinimizerLoLBFGS.h"
//#include"energyMinimizerNesterovAG.cuh"
#include "utilities.cuh"

/*! \file energyMinimizerLoLBFGS.cpp */

energyMinimizerLoLBFGS::energyMinimizerLoLBFGS(shared_ptr<simpleModel> system)
    {
    setModel(system);
    initializeParameters();
    initializeFromModel();
    };

/*!
Initialize the minimizer with some default parameters. that do not depend on Ndof
*/
void energyMinimizerLoLBFGS::initializeParameters()
    {
    currentIterationInMLoop=0;
    setForceCutoff(0.000000000001);
    reductions.resize(3);
    setMaximumIterations(1000);
    setLoLBFGSParameters();
    iterations = 0;
    forceMax = 100.;
    setGPU(false);
    };

void energyMinimizerLoLBFGS::initializeFromModel()
    {
    Ndof = model->getNumberOfParticles();
    unscaledStep.resize(Ndof);
    sumReductionIntermediate.resize(Ndof);
    sumReductionIntermediate2.resize(Ndof);
    for (int mm = 0; mm < m; ++mm)
        {
        gradientDifference[mm].resize(Ndof);
        secantEquation[mm].resize(Ndof);
        dVec zero(0.0);
        {
        ArrayHandle<dVec> y(gradientDifference[mm]);
        ArrayHandle<dVec> s(secantEquation[mm]);
        for (int ii = 0; ii < Ndof; ++ii)
            {
            y.data[ii] =zero;
            s.data[ii] =zero;
            }
        }
        }
    };

void energyMinimizerLoLBFGS::LoLBFGSStepGPU()
    {
        /*
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
    */
    }

void energyMinimizerLoLBFGS::LoLBFGSStepCPU()
    {
    int lastM = (currentIterationInMLoop - 1+m) %m;
    //step 1
    {
    if(iterations == 0)
        {
        sim->computeForces();
        unscaledStep = model->returnForces();
        }
    }

    //step 2
    {
    ArrayHandle<dVec> p(unscaledStep);
    ArrayHandle<scalar> sy(sDotY);
    ArrayHandle<scalar> a(alpha);
    for(int ii = 0; ii < m; ++ii)
        {
        int tMinusI = (currentIterationInMLoop - ii - 1 + m) % m;
        ArrayHandle<dVec> s(secantEquation[tMinusI],access_location::host,access_mode::read);
        ArrayHandle<dVec> y(gradientDifference[tMinusI],access_location::host,access_mode::read);
        sy.data[ii]=host_dVec_dot_products(s.data,y.data,Ndof);
        a.data[ii] = 0;
        if(sy.data[ii] != 0)
            a.data[ii] = host_dVec_dot_products(s.data,p.data,Ndof)/sy.data[ii];
        host_dVec_plusEqual_dVec(p.data,y.data,-a.data[ii],Ndof);
        }
    }

    //step 3
    {
    ArrayHandle<scalar> sy(sDotY);
    ArrayHandle<dVec> y(gradientDifference[lastM],access_location::host,access_mode::read);
    scalar val1 = sy.data[0];
    scalar val2 = host_dVec_dot_products(y.data,y.data,Ndof);
    if(val2!=0)
        {
        ArrayHandle<dVec> p(unscaledStep);
        host_dVec_plusEqual_dVec(p.data,y.data,val1/val2,Ndof);
        };
    }

    //step 4
    {
    ArrayHandle<dVec> p(unscaledStep);
    ArrayHandle<scalar> sy(sDotY);
    ArrayHandle<scalar> a(alpha);
    for(int ii = m-1; ii >= 0; --ii)
        {
        int tMinusI = (currentIterationInMLoop - ii - 1 + m) % m;
        ArrayHandle<dVec> s(secantEquation[tMinusI],access_location::host,access_mode::read);
        ArrayHandle<dVec> y(gradientDifference[tMinusI],access_location::host,access_mode::read);
        scalar beta =0;
        if(sy.data[ii] != 0)
            {
            beta = host_dVec_dot_products(y.data,p.data,Ndof)/sy.data[ii];
            host_dVec_plusEqual_dVec(p.data,s.data,a.data[ii]-beta,Ndof);
            }
        }
    }

    //update step
    {
    ArrayHandle<dVec> p(unscaledStep,access_location::host,access_mode::read);
    ArrayHandle<dVec> s(secantEquation[currentIterationInMLoop],access_location::host,access_mode::readwrite);
    host_dVec_times_scalar(p.data,eta/c,s.data,Ndof);
    }
    //temporarily store the old forces here in the gradient difference term
    gradientDifference[currentIterationInMLoop] = model->returnForces();
    model->moveParticles(secantEquation[currentIterationInMLoop]);
    sim->computeForces();
    unscaledStep = model->returnForces();
    {
    ArrayHandle<dVec> y(gradientDifference[currentIterationInMLoop],access_location::host,access_mode::readwrite);
    ArrayHandle<dVec> p(unscaledStep,access_location::host,access_mode::read);
    host_dVec_plusEqual_dVec(y.data,p.data,-1.0,Ndof);
    }

    //get force norm
    {
    ArrayHandle<dVec> p(unscaledStep,access_location::host,access_mode::read);
    forceMax=sqrt(host_dVec_dot_products(p.data,p.data,Ndof))/Ndof;
    }

    };

void energyMinimizerLoLBFGS::minimize()
    {

    if (Ndof != model->getNumberOfParticles())
        initializeFromModel();

    int curIterations = iterations;
    //always iterate at least once
    while( ((iterations < maxIterations) && (forceMax > forceCutoff)) || iterations == curIterations )
        {
        if(useGPU)
            LoLBFGSStepGPU();
        else
            LoLBFGSStepCPU();
        iterations +=1;
        currentIterationInMLoop= (currentIterationInMLoop+1)%m;
        if(iterations%1000 == 999)
            printf("step %i max force:%.3g  \n",iterations,forceMax);cout.flush();
        eta = tau/(tau + iterations)*deltaT;
        };
    printf("LoLBFGS finished: step %i max force:%.3g  \n",iterations,forceMax);cout.flush();

    }
