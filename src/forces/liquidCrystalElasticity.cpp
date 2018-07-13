#include "liquidCrystalElasticity.h"
/*! \file liquidCrystalElasticity.cpp */

liquidCrystalElasticity::liquidCrystalElasticity() 
    : baseLatticeForce()
    {
    
    };

void liquidCrystalElasticity::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    baseLatticeForce::computeForceCPU(forces,zeroOutForce);
    };
void liquidCrystalElasticity::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    baseLatticeForce::computeForceGPU(forces,zeroOutForce);
    };
