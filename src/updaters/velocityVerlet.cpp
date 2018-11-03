#include "velocityVerlet.h"
#include "velocityVerlet.cuh"
/*! \file velocityVerlet.cpp */


void velocityVerlet::integrateEOMCPU()
    {
    {//scope for array handles
    ArrayHandle<dVec> h_f(model->returnForces());
    ArrayHandle<dVec> h_v(model->returnVelocities());
    ArrayHandle<scalar> h_m(model->returnMasses());
    ArrayHandle<dVec> h_d(displacement);
    #include "ompParallelLoopDirective.h"
    for (int i = 0; i < Ndof; ++i)
        {
        //update displacement
        h_d.data[i] = deltaT*h_v.data[i] + (0.5*deltaT*deltaT)*h_f.data[i];
        //do first half of velocity update
        h_v.data[i] += (0.5/h_m.data[i])*deltaT*h_f.data[i];
        };
    };//handle scope
    //move particles, then update the forces
    model->moveParticles(displacement);
    sim->computeForces();

    {//array handle scope
    //update second half of velocity vector based on new forces
    ArrayHandle<dVec> h_f(model->returnForces());
    ArrayHandle<dVec> h_v(model->returnVelocities());
    ArrayHandle<scalar> h_m(model->returnMasses());
    #include "ompParallelLoopDirective.h"
    for (int i = 0; i < Ndof; ++i)
        {
        h_v.data[i] += (0.5/h_m.data[i])*deltaT*h_f.data[i];
        };
    };//handle scope

    };

void velocityVerlet::integrateEOMGPU()
    {
    ArrayHandle<scalar> d_m(model->returnMasses(),access_location::device,access_mode::read);
    //first half step
    {//array handle scope
    ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_d(displacement,access_location::device,access_mode::overwrite);
    //this call sets the displacement and also does the first half of the velocity update
    gpu_displacement_velocity_verlet(d_d.data,d_v.data,d_f.data,d_m.data,deltaT,Ndof);
    }
    //move particles and recompute forces
    model->moveParticles(displacement);
    sim->computeForces();

    //update velocities again
    ArrayHandle<dVec> d_f(model->returnForces(),access_location::device,access_mode::read);
    ArrayHandle<dVec> d_v(model->returnVelocities(),access_location::device,access_mode::readwrite);
    gpu_update_velocity(d_v.data,d_f.data,d_m.data,deltaT,Ndof);
    };
