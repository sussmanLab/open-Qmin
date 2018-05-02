#include "harmonicAngle.h"
/*! \file harmonicAngle.cpp */

void harmonicAngle::computeForces(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    energy = 0.0;
    if(useGPU)
        computeForceGPU(forces,zeroOutForce);
    else
        computeForceCPU(forces,zeroOutForce);

    };

void harmonicAngle::computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    UNWRITTENCODE("gpu calculation of harmonic Angles");
    };

/*!
 below, we have
 v1 = ri - rj
 v2 = rk - rj
 */
void harmonicAngle::computeForceCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    ArrayHandle<dVec> pos(model->returnPositions());
    ArrayHandle<dVec> f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < model->getNumberOfParticles(); ++pp)
            f.data[pp] = make_dVec(0.0);

    //loop over Angles
    for (int aa = 0; aa < angleList.size(); ++aa)
        {
        simpleAngle angle = angleList[aa];
        dVec v12,v32;
        //define useful vectors and scalars for the force computation
        model->Box->minDist(pos.data[angle.i],pos.data[angle.j],v12);
        model->Box->minDist(pos.data[angle.k],pos.data[angle.j],v32);
        scalar normV12 = norm(v12);
        scalar normV32 = norm(v32);
        scalar Cabbc = dot(v12,v32)/(normV12*normV32);
        scalar Sabbc = sqrt(1-Cabbc*Cabbc);
        scalar theta = acos(Cabbc);
        scalar dtheta = (theta - angle.t0);
        scalar thetaK = angle.kt * dtheta;

        energy += 0.5*thetaK*dtheta;

        scalar A = -thetaK*Sabbc;
        scalar A11 = A*Cabbc/(normV12*normV12);
        scalar A22 = A*Cabbc/(normV32*normV32);
        scalar A12 = -A/(normV12*normV32);
        dVec Fab = A11*v12 + A12*v32;
        dVec Fcb = A12*v12 + A22*v32;

        f.data[angle.i] += Fab;
        f.data[angle.j] -= (Fab+Fcb);
        f.data[angle.k] += Fcb;
        };
    };
