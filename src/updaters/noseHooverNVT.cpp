#include "noseHooverNVT.h"
#include "noseHooverNVT.cuh"
#include "utilities.cuh"
#include "velocityVerlet.cuh"
/*! \file noseHooverNVT.cpp */

noseHooverNVT::noseHooverNVT(shared_ptr<simpleModel> system,scalar _Temperature, int _nChain)
    {
    setChainLength(_nChain);
    setModel(system);
    temperature = _Temperature;
    initializeFromModel();//this also calls setT()
    };

void noseHooverNVT::initializeFromModel()
    {
    Ndof = model->getNumberOfParticles();
    displacement.resize(Ndof);
    keArray.resize(Ndof);
    keIntermediateReduction.resize(Ndof);
    kineticEnergyScaleFactor.resize(2);
    setT(temperature); //the bath mass depends on the number of degrees of freedom
    };

void noseHooverNVT::setChainLength(int _m)
    {
    Nchain = _m;
    bathVariables.resize(Nchain+1);
    ArrayHandle<scalar4> h_bv(bathVariables);
    for (int ii = 0; ii < Nchain+1; ++ii)
        {
        h_bv.data[ii].x = 0.0;
        h_bv.data[ii].y = 0.0;
        h_bv.data[ii].z = 0.0;
        h_bv.data[ii].w = 0.0;
        };
    };

void noseHooverNVT::setT(scalar _t)
    {
    temperature = _t;
    ArrayHandle<scalar4> h_bv(bathVariables);
    h_bv.data[0].w = DIMENSION*(Ndof-DIMENSION)*temperature;
    for(int ii = 1; ii < Nchain+1; ++ii)
        h_bv.data[ii].w = temperature;

    ArrayHandle<scalar> kes(kineticEnergyScaleFactor);
    kes.data[0] = h_bv.data[0].w;
    kes.data[1] = 1.0;
    };

/*!
The implementation here closely follows algorithms 30 - 32 in Frenkel & Smit, generalized to the
case where the chain length is not necessarily always 2
*/
void noseHooverNVT::integrateEOMCPU()
    {
    //first, propagate chain and scale velocities
    {//array handle
    propagateChain();
    ArrayHandle<scalar> h_kes(kineticEnergyScaleFactor,access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(model->returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] = h_kes.data[1]*h_v.data[ii];
    }//end array handle

    propagatePositionsVelocites();

    //repeat chain propagation and velocity scaling
    {//array handle
    propagateChain();
    ArrayHandle<scalar> h_kes(kineticEnergyScaleFactor,access_location::host,access_mode::read);
    ArrayHandle<dVec> h_v(model->returnVelocities());
    for (int ii = 0; ii < Ndof; ++ii)
        h_v.data[ii] = h_kes.data[1]*h_v.data[ii];
    }//end array handle
    };

/*!
The simple part of the algorithm actually updates the positions and velocities of the partices.
This is the step in which a force calculation is required.
 */
void noseHooverNVT::propagatePositionsVelocites()
    {
    ArrayHandle<scalar> h_kes(kineticEnergyScaleFactor);
    h_kes.data[0] = 0.0;
    scalar deltaT2 = 0.5*deltaT;
    //first half of time step
    {//array handle
    ArrayHandle<dVec> h_disp(displacement, access_location::host,access_mode::overwrite);
    ArrayHandle<dVec> h_v(model->returnVelocities(),access_location::host,access_mode::read);
    #include "ompParallelLoopDirective.h"
    for (int ii = 0; ii < Ndof; ++ii)
        h_disp.data[ii] = deltaT2*h_v.data[ii];
    }//end array handle

    model->moveParticles(displacement);
    sim->computeForces();
    //second half of time step
    {//array handle
    ArrayHandle<dVec> h_disp(displacement, access_location::host,access_mode::overwrite);
    ArrayHandle<dVec> h_v(model->returnVelocities(),access_location::host,access_mode::readwrite);
    ArrayHandle<dVec> h_f(model->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<scalar> h_m(model->returnMasses(),access_location::host,access_mode::read);
    #include "ompParallelLoopDirective.h"
    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_v.data[ii] = h_v.data[ii] + (deltaT/h_m.data[ii])*h_f.data[ii];
        h_disp.data[ii] = deltaT2*h_v.data[ii];
        h_kes.data[0] += 0.5*h_m.data[ii]*dot(h_v.data[ii],h_v.data[ii]);
        };
    }//handle end
    model->moveParticles(displacement);
    };

void noseHooverNVT::propagateChainGPU()
    {
    ArrayHandle<scalar> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<scalar4> d_bath(bathVariables,access_location::device,access_mode::readwrite);
    gpu_propagate_noseHoover_chain(d_kes.data,
                                   d_bath.data,
                                   deltaT,
                                   temperature,
                                   Nchain,
                                   Ndof);
    };

void noseHooverNVT::propagateChain()
    {
    ArrayHandle<scalar> h_kes(kineticEnergyScaleFactor);
    scalar dt8 = 0.125*deltaT;
    scalar dt4 = 0.25*deltaT;
    scalar dt2 = 0.5*deltaT;

    //first quarter time step
    //partially update bath velocities and accelerations (quarter-timestep), from Nchain to 0
    ArrayHandle<scalar4> bath(bathVariables);
    for (int ii = Nchain-1; ii > 0; --ii)
        {
        //update the acceleration: G = (Q_{i-1}*v_{i-1}^2 - T)/Q_i
        bath.data[ii].z = (bath.data[ii-1].w*bath.data[ii-1].y*bath.data[ii-1].y-temperature)/bath.data[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        scalar ef = exp(-dt8*bath.data[ii+1].y);
        bath.data[ii].y *= ef;
        bath.data[ii].y += bath.data[ii].z*dt4;
        bath.data[ii].y *= ef;
        };

    bath.data[0].z = (2.0*h_kes.data[0] - DIMENSION*(Ndof-DIMENSION)*temperature)/bath.data[0].w;
    scalar ef = exp(-dt8*bath.data[1].y);
    bath.data[0].y *= ef;
    bath.data[0].y += bath.data[0].z*dt4;
    bath.data[0].y *= ef;

    //update bath positions (half timestep)
    for (int ii = 0; ii < Nchain; ++ii)
        bath.data[ii].x += dt2*bath.data[ii].y;

    //get the factor that will scale particle velocities...
    h_kes.data[1] = exp(-dt2*bath.data[0].y);
    //...and pre-emptively update the kinetic energy
    h_kes.data[0] = h_kes.data[1]*h_kes.data[1]*h_kes.data[0];

    //finally, do the other quarter-timestep of the velocities and accelerations, from 0 to Nchain
    bath.data[0].z = (2.0*h_kes.data[0] - DIMENSION*(Ndof-DIMENSION)*temperature)/bath.data[0].w;
    ef = exp(-dt8*bath.data[1].y);
    bath.data[0].y *= ef;
    bath.data[0].y += bath.data[0].z*dt4;
    bath.data[0].y *= ef;
    for (int ii = 1; ii < Nchain; ++ii)
        {
        bath.data[ii].z = (bath.data[ii-1].w*bath.data[ii-1].y*bath.data[ii-1].y-temperature)/bath.data[ii].w;
        //the exponential factor is exp(-dt*v_{i+1}/2)
        scalar ef = exp(-dt8*bath.data[ii+1].y);
        bath.data[ii].y *= ef;
        bath.data[ii].y += bath.data[ii].z*dt4;
        bath.data[ii].y *= ef;
        };
    };

void noseHooverNVT::integrateEOMGPU()
    {
    //The kernel calling scheme. To avoid ridiculous numbers of brackets for array handle scoping,
    //we'll define helper functions

    //for now, let's update the chain variables on the CPU... profile later
//    propagateChain(); // use data structure that holds [KE,s], update both.
NVTXPUSH("chain prop 1");
    propagateChainGPU();
NVTXPOP();

NVTXPUSH("rescale vel 1");
    rescaleVelocitiesGPU(); //use the velocity vector and the [KE,s] data structure. Note that KE is already scaled by s^2 in the above step
NVTXPOP();

NVTXPUSH("pos Vel prop");
    propagatePositionsVelocitiesGPU();
NVTXPOP();

NVTXPUSH("KE calc");
    calculateKineticEnergyGPU(); //get the kinetic energy into the [KE,s] data structure
NVTXPOP();

NVTXPUSH("chain prop 2");
    propagateChainGPU();
NVTXPOP();

    //propagateChain();
NVTXPUSH("rescale vel 2");
    rescaleVelocitiesGPU();
NVTXPOP();

    };

void noseHooverNVT::rescaleVelocitiesGPU()
    {
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<scalar> h_kes(kineticEnergyScaleFactor,access_location::host,access_mode::read);
    gpu_dVec_times_scalar(d_v.data,h_kes.data[1],Ndof);
    };

void noseHooverNVT::propagatePositionsVelocitiesGPU()
    {
    scalar deltaT2 = 0.5*deltaT;
    //first, move particles according to velocities
NVTXPUSH("move particles1");
    model->moveParticles(model->returnVelocities(),deltaT2);
NVTXPOP();
NVTXPUSH("computeForces");
    sim->computeForces();
NVTXPOP();

    //the second half of the time step first updates the velocites then moves particles again
NVTXPUSH("update velocity");
    {
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<scalar> d_m(model->returnMasses(),access_location::device,access_mode::read);
    //the NH velocity update is the same as the velocity verlet but with a different effective timestep
    gpu_update_velocity(d_v.data,d_f.data,d_m.data,deltaT*2.,Ndof);
    };
NVTXPOP();
    //move particles according to velocities again
NVTXPUSH("move particles2");
    model->moveParticles(model->returnVelocities(),deltaT2);
NVTXPOP();
    };

void noseHooverNVT::calculateKineticEnergyGPU()
    {
    {//array handle scope for keArray prep
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::read);
    ArrayHandle<scalar> d_m(model->returnMasses(),access_location::device,access_mode::read);
    ArrayHandle<scalar> d_keArray(keArray,access_location::device,access_mode::overwrite);
    gpu_scalar_times_dVec_squared(d_v.data,d_m.data,0.5,d_keArray.data,Ndof);
    };//end scope

    {//array handle scope for parallel reduction
    ArrayHandle<scalar> d_keArray(keArray,access_location::device,access_mode::read);
    ArrayHandle<scalar> d_kes(kineticEnergyScaleFactor,access_location::device,access_mode::readwrite);
    ArrayHandle<scalar> d_keIntermediate(keIntermediateReduction,access_location::device,access_mode::overwrite);
    int blockSize = 256;
    gpu_parallel_reduction(d_keArray.data,d_keIntermediate.data,d_kes.data,0,Ndof,blockSize);
    }
    };
