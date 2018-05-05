#ifndef harmonicBond_H
#define harmonicBond_H

#include "baseForce.h"
/*! \file harmonicBond.h */
//! calculate the forces due to a set of harmonic springs
/*!
 * assumes that the energy associated with each bond is
 * e = 0.5*k(|r_i - r_j|-r0)^2
 *
*/
class harmonicBond : public force
    {
    public:
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true);

        //!set the bond list by passing in a vector of simpleBonds (defined in structures.h)
        void setBondList(vector<simpleBond> &blist){bondList = blist;};

        vector<simpleBond> bondList;

        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
    };
#endif
