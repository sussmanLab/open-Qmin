#include "nematicInteraction.h"
#include "nematicInteraction.cuh"
#include "qTensorFunctions.h"
/*! \file nematicInteraction.cpp */

nematicInteraction::nematicInteraction(double _A, double _B, double _C, double _L)
    {
    A=_A;
    B=_B;
    C=_C;
    L=_L;
    useNeighborList = false;
    forceTuner = make_shared<kernelTuner>(16,1024,16,5,200000);
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void nematicInteraction::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    energy=0.0;
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int currentIndex;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        qCurrent = Qtensors.data[currentIndex];
        xDown = Qtensors.data[neighbors[0]];
        xUp = Qtensors.data[neighbors[1]];
        yDown = Qtensors.data[neighbors[2]];
        yUp = Qtensors.data[neighbors[3]];
        zDown = Qtensors.data[neighbors[4]];
        zUp = Qtensors.data[neighbors[5]];

        //compute the elastic terms depending only on the current site
        h_f.data[currentIndex] -= 0.5*A*derivativeTrQ2(qCurrent);
        h_f.data[currentIndex] -= B/3.0*derivativeTrQ3(qCurrent);
        h_f.data[currentIndex] -= 0.25*C*derivativeTrQ2Squared(qCurrent);

        //use the neighbors to compute the laplacian term
        dVec spatialTerm = 2.0*L*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
        scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
        spatialTerm[0] += AxxAyy;
        spatialTerm[1] *= 2.0;
        spatialTerm[2] *= 2.0;
        spatialTerm[3] += AxxAyy;
        spatialTerm[4] *= 2.0;
        h_f.data[currentIndex] -= spatialTerm;
        };
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void nematicInteraction::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);

    forceTuner->begin();
    gpu_qTensor_oneConstantForce(d_force.data,
                              d_spins.data,
                              lattice->latticeIndex,
                              A,B,C,L,
                              N,
                              zeroOutForce,
                              forceTuner->getParameter()
                              );
    forceTuner->end();
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void nematicInteraction::computeEnergyCPU()
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
