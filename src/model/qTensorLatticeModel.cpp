#include "qTensorLatticeModel.h"
#include "cubicLattice.cuh"
/*! \file qTensorLatticeModel.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */
qTensorLatticeModel::qTensorLatticeModel(int l, bool _useGPU)
    : cubicLattice(l,false,_useGPU)
    {
    normalizeSpins = false;
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

void qTensorLatticeModel::setNematicQTensorRandomly(noiseSource &noise,scalar S0)
    {
    scalar amplitude =  3./2.*S0;
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar theta = acos(2.0*noise.getRealUniform()-1);
            scalar phi = 2.0*PI*noise.getRealUniform();
            pos.data[pp][0] = amplitude*(sin(theta)*sin(theta)*cos(phi)*cos(phi)-1.0/3.0);
            pos.data[pp][1] = amplitude*sin(theta)*sin(theta)*cos(phi)*sin(phi);
            pos.data[pp][2] = amplitude*sin(theta)*cos(theta)*cos(phi);
            pos.data[pp][3] = amplitude*(sin(theta)*sin(theta)*sin(phi)*sin(phi)-1.0/3.0);
            pos.data[pp][4] = amplitude*sin(theta)*cos(theta)*sin(phi);
            };
        }
    else
        {
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::overwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(N);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_nematic_qTensors(pos.data,d_curandRNGs.data, amplitude, blockSize,nBlocks,N);
        }
    };

void qTensorLatticeModel::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
    cubicLattice::moveParticles(displacements,scale);
    };
