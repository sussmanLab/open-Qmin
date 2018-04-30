#include "simple2DModel.h"
/*! \file simple2DModel.cpp" */

/*!
 * Set the size of basic data structures...
*/
simple2DModel::simple2DModel(int n, bool _useGPU) :
    N(n), useGPU(_useGPU)
    {
    cout << "initializing a model with "<< N << " particles" << endl;
    initializeSimple2DModel(n);
    };

/*!
 * actually set the array sizes. positions, velocities, forces are zero
 * masses are set to unity
*/
void simple2DModel::initializeSimple2DModel(int n)
    {
    positions.resize(n);
    velocities.resize(n);
    forces.resize(n);
    masses.resize(n);
    vector<dVec> zeroes(N,make_dVec(0.0));
    vector<scalar> ones(N,1.0);
    fillGPUArrayWithVector(zeroes,positions);
    fillGPUArrayWithVector(zeroes,velocities);
    fillGPUArrayWithVector(zeroes,forces);
    fillGPUArrayWithVector(ones,masses);
    };

