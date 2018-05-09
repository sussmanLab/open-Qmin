#include "simpleModel.h"
#include "utilities.cuh"
#include "simpleModel.cuh"
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
    N=n;
    selfForceCompute = false;
    positions.resize(n);
    velocities.resize(n);
    forces.resize(n);
    masses.resize(n);
    radii.resize(n);
    types.resize(n);
    vector<dVec> zeroes(N,make_dVec(0.0));
    vector<scalar> ones(N,1.0);
    vector<scalar> halves(N,.5);
    vector<int> units(N,0);
    fillGPUArrayWithVector(units,types);
    fillGPUArrayWithVector(zeroes,positions);
    fillGPUArrayWithVector(zeroes,velocities);
    fillGPUArrayWithVector(zeroes,forces);
    fillGPUArrayWithVector(ones,masses);
    fillGPUArrayWithVector(halves,radii);
    };

scalar simpleModel::computeKineticEnergy()
    {
    ArrayHandle<scalar> h_m(masses,access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(velocities);
    scalar en = 0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        en += 0.0*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    return en;
    };

scalar simpleModel::computeInstantaneousTemperature(bool fixedMomentum)
    {
    ArrayHandle<scalar> h_m(masses,access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(velocities);
    scalar en = 0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        en += 1.0*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    if(fixedMomentum)
        return en /((N-DIMENSION)*DIMENSION);
    else
        return en /(N*DIMENSION);
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

scalar simpleModel::setVelocitiesMaxwellBoltzmann(scalar T,noiseSource &noise)
    {
    ArrayHandle<scalar> h_m(masses,access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(velocities);
    scalar KE = 0.0;
    dVec P(0.0);
    for (int ii = 0; ii < N; ++ii)
        {
        for (int dd = 0; dd <DIMENSION; ++dd)
            h_v.data[ii].x[dd] = noise.getRealNormal(0.0,sqrt(T/h_m.data[ii]));
        P += h_m.data[ii]*h_v.data[ii];
        KE += 0.5*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        }
    //remove excess momentum, calculate the ke
    KE = 0.0;
    for (int ii = 0; ii < N; ++ii)
        {
        h_v.data[ii] += (-1.0/(N*h_m.data[ii]))*P;
        KE += 0.5*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        };
    return KE;
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
        ArrayHandle<dVec> d_disp(displacement,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        gpu_move_particles(d_pos.data,d_disp.data,*(Box),scale,N);
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
                ArrayHandle<dVec> d_f(forces);
                gpu_zero_array(d_f.data,N);
            };
        };
    };
