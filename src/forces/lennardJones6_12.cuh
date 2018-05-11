#ifndef harmonicRepulsion_CUH
#define harmonicRepulsion_CUH

#include "std_include.h"
#include "indexer.h"
#include "periodicBoundaryConditions.h"
/*! \file harmonicRepulsion.cuh */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

//!calculate LJ forces repulsive forces
bool gpu_lennardJones6_12_calculation(dVec *d_force,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rCut,
                                   int N,
                                   bool zeroForce);

//!calculate LJ forces energy per particle
bool gpu_lennardJones6_12_energy(scalar *d_energy,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_epsilon,
                                   scalar *d_sigma,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   scalar rCut,
                                   int N);

/** @} */ //end of group declaration
#endif
