#include "simpleModel.h"
/*! \file simpleModel.cpp" */

/*!
 * Set the size of basic data structures...
*/
simpleModel::simpleModel(int n, bool _useGPU) :
    N(n), useGPU(_useGPU)
    {
    cout << "initializing a model with "<< N << " particles" << endl;
    initializeSimpleModel(n);
    Box = make_shared<periodicBoundaryConditions>(pow(N,1.0/DIMENSION));
    };

/*!
 * actually set the array sizes. positions, velocities, forces are zero
 * masses are set to unity
*/
void simpleModel::initializeSimpleModel(int n)
    {
    selfForceCompute = false;
    positions.resize(n);
    velocities.resize(n);
    forces.resize(n);
    masses.resize(n);
    radii.resize(n);
    vector<dVec> zeroes(N,make_dVec(0.0));
    vector<scalar> ones(N,1.0);
    vector<scalar> halves(N,.5);
    fillGPUArrayWithVector(zeroes,positions);
    fillGPUArrayWithVector(zeroes,velocities);
    fillGPUArrayWithVector(zeroes,forces);
    fillGPUArrayWithVector(ones,masses);
    fillGPUArrayWithVector(halves,radii);
    };

void simpleModel::setParticlePositionsRandomly(noiseSource &noise)
    {
    dVec bDims;
    Box->getBoxDims(bDims);
    ArrayHandle<dVec> pos(positions);
    for(int pp = 0; pp < N; ++pp)
        for (int dd = 0; dd <DIMENSION; ++dd)
            pos.data[pp].x[dd] = noise.getRealUniform(0.0,bDims.x[dd]);
    };

/*!
 * move particles on either CPU or gpu
*/
void simpleModel::moveParticles(GPUArray<dVec> &displacement, scalar scale)
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
            UNWRITTENCODE("moveParticles on the GPU");
        };
    };

/*!
 *
*/
void simpleModel::computeForces(bool zeroOutForces)
    {
    if(zeroOutForces)
        {
        if(!useGPU)
            {//cpu branch
            ArrayHandle<dVec> h_f(forces);
            dVec dArrayZero(0.0);
            for(int pp = 0; pp <N;++pp)
                h_f.data[pp] = dArrayZero;
            }
        else
            {
                UNWRITTENCODE("zero out forces on GPU");
            };
        };
    };
