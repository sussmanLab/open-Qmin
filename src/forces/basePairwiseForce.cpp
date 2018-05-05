#include "basePairwiseForce.h"
/*! \file basePairwiseForce.cpp */

void basePairwiseForce::computeForces(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    energy = 0.0;
    if(useGPU)
        computeForceGPU(forces,zeroOutForce);
    else
        computeForceCPU(forces,zeroOutForce);

    };

void basePairwiseForce::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    UNWRITTENCODE("gpu calculation of pairwise forces");
    };

void basePairwiseForce::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    if(!useNeighborList)
        UNWRITTENCODE("pairwise forces without neighbor list not written yet");
    
    ArrayHandle<dVec> h_force(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < model->getNumberOfParticles(); ++pp)
            h_force.data[pp] = make_dVec(0.0);

    neighbors->computeNeighborLists(model->returnPositions());
    ArrayHandle<unsigned int> h_npp(neighbors->neighborsPerParticle);
    ArrayHandle<int> h_n(neighbors->particleIndices);
    ArrayHandle<dVec> h_nv(neighbors->neighborVectors);
    dVec relativeDistance;
    dVec f;
    for (int p1 = 0; p1 < model->getNumberOfParticles();++p1)
        {
        int neigh = h_npp.data[p1];
        //loop over all neighbors
        for (int nn = 0 ; nn < neigh; ++nn)
            {
            int nidx = neighbors->neighborIndexer(nn,p1);
            int p2 = h_n.data[nidx];
            if(p2 < p1) continue;
            relativeDistance = h_nv.data[nidx];

            getParametersForParticlePair(p1,p2,params);
            computePairwiseForce(relativeDistance,params,f);
            h_force.data[p1] += f;
            h_force.data[p2] -= f;
            };
        };

    };

