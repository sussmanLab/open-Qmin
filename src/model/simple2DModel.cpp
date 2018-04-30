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

/*!
 * move particles on either CPU or gpu
*/
void simple2DModel::moveParticles(GPUArray<dVec> &displacement, scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacement, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(positions);
        for(int pp = 0; pp < N; ++pp)
            {
            h_pos.data[pp] += scale*h_disp.data[pp];
            Box->putInBoxReal(h_pos.data[pp]);
            }
        }
    else
        {//gpu branch
        };
    };
