#ifndef harmonicAngle_H
#define harmonicAngle_H

#include "baseForce.h"
/*! \file harmonicAngle.h */
//! calculate the forces due to a set of angular springs
/*!
 * assumes that the energy associated with each bond is
 * e = 0.5*k(theta-theta_0)^2,
 *where theta is the angle formed by particles i, j, k (with j as the central particle
*/
class harmonicAngle : public force
    {
    public:
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true);

        //!set the bond list by passing in a vector of simpleAngles (defined in structures.h)
        void setAngleList(vector<simpleAngle> &alist){angleList = alist;};

        vector<simpleAngle> angleList;

        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
    };

#endif
