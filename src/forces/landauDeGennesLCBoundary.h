#ifndef landauDeGennesLCBoundary_H
#define landauDeGennesLCBoundary_H

#include "latticeBoundaries.h"
#include "qTensorFunctions.h"
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif
/*! \file landauDeGennesLCBoundary.h */

//!compute the force on lattice site exerted by the boundary site
HOSTDEVICE void computeBoundaryForce(const dVec &latticeSite, const dVec &boundarySite,
                                     const boundaryObject &bObj, dVec &force)
    {
    //scalar p1 = bObj.P1;
    //scalar p2 = bObj.P2;
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            dVec difference = 2.0*(latticeSite-boundarySite);
            dVec answer = 2.0*difference;
            answer[0] += difference[3];
            answer[3] += difference[0];
            force = -bObj.P1*answer;
            break;
            }
        //case boundaryType::other:
            //{
            //do stuff;
            //break;
            //}
        default:
            UNWRITTENCODE("non-defined boundary type is attempting a force computation");
        }
    //make sure "force" has been modified
    };

//!compute the force on lattice site exerted by the boundary site
HOSTDEVICE scalar computeBoundaryEnergy(const dVec &latticeSite, const dVec &boundarySite,
                                     const boundaryObject &bObj)
    {
    //scalar p1 = bObj.P1;
    //scalar p2 = bObj.P2;
    scalar energy = 0.0;
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            dVec difference = latticeSite-boundarySite;
            energy = bObj.P1*TrQ2(difference);
            break;
            }
        //case boundaryType::other:
            //{
            //do stuff;
            //break;
            //}
        default:
            UNWRITTENCODE("non-defined boundary type is attempting an energy computation");
        }
    return energy;
    };

#undef HOSTDEVICE
#endif
