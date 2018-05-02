#include "harmonicBond.h"
/*! \file harmonicBond.cpp */

void harmonicBond::computeForces(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    energy = 0.0;
    if(useGPU)
        computeForceGPU(forces,zeroOutForce);
    else
        computeForceCPU(forces,zeroOutForce);

    };

void harmonicBond::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    UNWRITTENCODE("gpu calculation of harmonic bonds");
    };

void harmonicBond::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    ArrayHandle<dVec> pos(model->returnPositions());
    ArrayHandle<dVec> f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < model->getNumberOfParticles(); ++pp)
            f.data[pp] = make_dVec(0.0);

    //loop over bonds
    for (int bb = 0; bb < bondList.size(); ++bb)
        {
        simpleBond B = bondList[bb];
        dVec disp;
        model->Box->minDist(pos.data[B.i],pos.data[B.j],disp);
        scalar sep = norm(disp);
        energy += 0.5*B.k*(sep-B.r0)*(sep-B.r0);
        dVec Fab = B.k*(B.r0-sep)*disp;
        f.data[B.i] += Fab;
        f.data[B.j] -= Fab;
        };
    };

