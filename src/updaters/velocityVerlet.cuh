#ifndef velocityVerlet_CUH
#define velocityVerlet_CUH

#include "std_include.h"
#include <cuda_runtime.h>
/*! \file velocityVerlet.cuh */

/** @addtogroup updaterKernels updater Kernels
 * @{
 * \brief CUDA kernels and callers 
 */

//!velocity = velocity +0.5*deltaT*force
bool gpu_update_velocity(dVec *d_velocity,
                      dVec *d_force,
                      scalar *d_mass,
                      scalar deltaT,
                      int N
                      );

//!displacement = dt*velocity + 0.5*dt^2*force
bool gpu_displacement_velocity_verlet(dVec *d_displacement,
                      dVec *d_velocity,
                      dVec *d_force,
                      scalar *d_mass,
                      scalar deltaT,
                      int N
                      );

/** @} */ //end of group declaration
#endif
