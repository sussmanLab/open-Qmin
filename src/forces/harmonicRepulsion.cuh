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

//!calculate harmonic repulsive forces by looping over all pairs...
bool gpu_harmonic_repulsion_allPairs(dVec *d_force,
                                   dVec *d_pos,
                                   int *particleType,
                                   scalar *d_radii,
                                   scalar *d_params,
                                   Index2D particleTypeIndexer,
                                   periodicBoundaryConditions &Box,
                                   int N,
                                   bool zeroForce);

//!calculate harmonic repulsive forces
bool gpu_harmonic_repulsion_calculation(dVec *d_force,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   int *particleType,
                                   scalar *d_radii,
                                   scalar *d_params,
                                   Index2D neighborIndexer,
                                   Index2D particleTypeIndexer,
                                   int N,
                                   bool zeroForce);
//!calculate harmonic repulsive forces assuming monodisperse systems
bool gpu_harmonic_repulsion_monodisperse_calculation(dVec *d_force,
                                   unsigned int *d_neighborsPerParticle,
                                   int *d_neighbors,
                                   dVec *d_neighborVectors,
                                   Index2D neighborIndexer,
                                   int N,
                                   bool zeroForce);

/** @} */ //end of group declaration
#endif
