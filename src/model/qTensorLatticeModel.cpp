#include "qTensorLatticeModel.h"
/*! \file qTensorLatticeModel.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */ 
qTensorLatticeModel::qTensorLatticeModel(int l, bool _useGPU)
    : cubicLattice(l,false,_useGPU)
    {
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

void qTensorLatticeModel::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
    cubicLattice::moveParticles(displacements,scale);
    };
