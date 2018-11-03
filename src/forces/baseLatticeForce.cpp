#include "baseLatticeForce.h"
#include "baseLatticeForce.cuh"
/*! \file baseLatticeForce.cpp */

baseLatticeForce::baseLatticeForce()
    {
    J=1.0;
    useNeighborList = false;
    forceTuner = make_shared<kernelTuner>(16,1024,16,5,200000);
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void baseLatticeForce::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> spins(lattice->returnPositions());

    #include "ompParallelLoopDirective.h"
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int si,sj;
        dVec spinI, spinJ;
        si = lattice->getNeighbors(i,neighbors,neighNum);
        spinI = spins.data[si];
        for (int j = 0; j < neighNum; ++j)
            {
            int sj = neighbors[j];
            if(sj < si) continue;
            spinJ = spins.data[sj];
            h_f.data[si] += (J*spinJ);
            h_f.data[sj] += (J*spinI);
            }
        };
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void baseLatticeForce::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);

    forceTuner->begin();
    gpu_lattice_spin_force_nn(d_force.data,
                              d_spins.data,
                              lattice->latticeIndex,
                              J,
                              N,
                              zeroOutForce,
                              forceTuner->getParameter()
                              );
    forceTuner->end();
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void baseLatticeForce::computeEnergyCPU()
    {
    energy=0.0;
    ArrayHandle<dVec> spins(lattice->returnPositions());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int si,sj;
    dVec spinI, spinJ;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        si = lattice->getNeighbors(i,neighbors,neighNum);
        spinI = spins.data[si];
        for (int j = 0; j < neighNum; ++j)
            {
            int sj = neighbors[j];
            if(sj < si) continue;
            spinJ = spins.data[sj];
            energy += -J*dot(spinI,spinJ);
            }
        };
    };
