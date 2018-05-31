#include "lennardJones6_12.h"
#include "lennardJones6_12.cuh"
#include "utilities.cuh"
/*! \file lennardJones6_12.cpp */


/*!
\pre the vector has 2*n^2 elements, where n is the number of types, and the values of type in the system
must be of the form 0, 1, 2, ...n. The input vector should be laid out as:
params[0] = epsilon_{0,0}
params[1] = epsilon_{0,1}
params[n] = epsilon_{0,n}
params[n+1] = epsilon_{1,0} (physically, this better be the same as epsilon_{0,1})
params[n+2] = epsilon_{1,1}
...
params[n^2-1] = epsilon_{n,n}

The second set of n^2 should be the sigma values, in the same order
*/
void lennardJones6_12::setForceParameters(vector<scalar> &params)
    {
    int nTypes = sqrt(params.size()/2);
    int nTypes2=nTypes*nTypes;
    epsilonParameters.resize(nTypes2);
    sigmaParameters.resize(nTypes2);
    particleTypeIndexer = Index2D(nTypes);
    ArrayHandle<scalar> h_s(epsilonParameters);
    ArrayHandle<scalar> h_e(sigmaParameters);
    for(int ii = 0; ii < nTypes2; ++ii)
        {
        int typeI = ii/nTypes;
        int typeJ = ii - typeI*nTypes;
        h_e.data[particleTypeIndexer(typeJ,typeI)] = params[ii];
        h_s.data[particleTypeIndexer(typeJ,typeI)] = params[nTypes2+ii];
        };
    };

/*
Need to get the type-type epsilon and the sum of radii
*/
void lennardJones6_12::getParametersForParticlePair(int index1, int index2, vector<scalar> &params)
    {
    ArrayHandle<int> particleType(model->returnTypes());
    ArrayHandle<scalar> h_e(epsilonParameters);
    ArrayHandle<scalar> h_s(sigmaParameters);
    params[0] = h_e.data[particleTypeIndexer(particleType.data[index2],particleType.data[index1])];
    params[1] = h_s.data[particleTypeIndexer(particleType.data[index2],particleType.data[index1])];
    };

void lennardJones6_12::computePairwiseForce(dVec &relativeDistance, scalar dnorm,vector<scalar> &params, dVec &f)
    {
    scalar rinv = params[1]/dnorm;
    scalar rinv2 = rinv*rinv;
    scalar rinv3 = rinv2*rinv;
    scalar rinv6 = rinv3*rinv3;
    scalar rinv12 = rinv6*rinv6;
    energy += 4.0*params[0]*(rinv12-rinv6);
    if(shiftAndCut)
        UNWRITTENCODE("shift and cut LJ potential");
    else
        {
        //F = -(\vec{r})/(norm(r)) * (dU/dr)|r
        if(dnorm <= rc)
            {
            scalar negativedUdr = 4.*params[0]*(12.0*rinv12/dnorm -6.*rinv6/dnorm);
            f= (negativedUdr/dnorm)*relativeDistance;
            }
        else
            f = make_dVec(0.0);
        };
    };

void lennardJones6_12::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    neighbors->computeNeighborLists(model->returnPositions());

    if(shiftAndCut)
        UNWRITTENCODE("gpu shift and cut LJ potential");

    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    int N = model->getNumberOfParticles();

    ArrayHandle<unsigned int> d_npp(neighbors->neighborsPerParticle,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(neighbors->particleIndices,access_location::device,access_mode::read);
    ArrayHandle<dVec> d_nv(neighbors->neighborVectors,access_location::device,access_mode::read);

    ArrayHandle<int> particleType(model->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<scalar> d_epsilon(epsilonParameters,access_location::device,access_mode::read);
    ArrayHandle<scalar> d_sigma(sigmaParameters,access_location::device,access_mode::read);

        gpu_lennardJones6_12_calculation(d_force.data,
                                       d_npp.data,
                                       d_n.data,
                                       d_nv.data,
                                       particleType.data,
                                       d_epsilon.data,
                                       d_sigma.data,
                                       neighbors->neighborIndexer,
                                       particleTypeIndexer,
                                       rc,
                                       N,
                                       zeroOutForce);
    }

scalar lennardJones6_12::computeEnergyGPU()
    {
    if(shiftAndCut)
        UNWRITTENCODE("gpu shift and cut LJ potential");

    neighbors->computeNeighborLists(model->returnPositions());
    int N = model->getNumberOfParticles();
    if(energyPerParticle.getNumElements() != N)
        {
        energyPerParticle.resize(N);
        energyIntermediateReduction.resize(N);
        energyReduction.resize(1);
        }

    {//scope for "energy per particle" calculation 
    ArrayHandle<scalar> d_energy(energyPerParticle,access_location::device,access_mode::overwrite);
    ArrayHandle<unsigned int> d_npp(neighbors->neighborsPerParticle,access_location::device,access_mode::read);
    ArrayHandle<int> d_n(neighbors->particleIndices,access_location::device,access_mode::read);
    ArrayHandle<dVec> d_nv(neighbors->neighborVectors,access_location::device,access_mode::read);
    ArrayHandle<int> particleType(model->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<scalar> d_epsilon(epsilonParameters,access_location::device,access_mode::read);
    ArrayHandle<scalar> d_sigma(sigmaParameters,access_location::device,access_mode::read);

        gpu_lennardJones6_12_energy(d_energy.data,
                                       d_npp.data,
                                       d_n.data,
                                       d_nv.data,
                                       particleType.data,
                                       d_epsilon.data,
                                       d_sigma.data,
                                       neighbors->neighborIndexer,
                                       particleTypeIndexer,
                                       rc,
                                       N);
    };//scope of energy per particle
    {//scope for parallel reduction
    ArrayHandle<scalar> d_energy(energyPerParticle,access_location::device,access_mode::read);
    ArrayHandle<scalar> d_intermediate(energyIntermediateReduction,access_location::device,access_mode::overwrite);
    ArrayHandle<scalar> d_ans(energyReduction,access_location::device,access_mode::overwrite);
    int blockSize = 256;
    gpu_parallel_reduction(d_energy.data,d_intermediate.data,d_ans.data,0,N,blockSize);
    };
    ArrayHandle<scalar> h_ans(energyReduction);
    return h_ans.data[0];
    }

