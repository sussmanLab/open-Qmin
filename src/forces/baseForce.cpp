#include "baseForce.h"
/*! \file baseForce.cpp */

force::force()
    {
    useGPU = false;
    };

void force::setForceParameters(vector<scalar> &params)
    {
    };

void force::computeForces(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    cout << "in the base force computer... that's odd..." << endl;


    };
