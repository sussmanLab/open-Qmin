#ifndef QTENSORFUNCTIONS2D_H
#define QTENSORFUNCTIONS2D_H

#include "std_include.h"

/*!
The 2Q-tensor has five independent components, which will get passed around in dVec structures...
a dVec of q[0,1] corresponds to the symmetric traceless tensor laid out as
    (q[0]    q[1]  )       q[2]    )
Q = (q[1]    -q[0] )

This file implements handy manipulations and functions of the Q-tensor as laid out this way
 */

#pragma hd_warning_disable
#pragma diag_suppress 2739

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file qTensorFunctions2d.h */

/** @defgroup Functions functions
 * @{
 \brief Utility functions that can be called from host or device
 */

//! return a qtensor given a director and a value of S0
HOSTDEVICE void qTensorFromDirector2D(scalar2 n, scalar S0, dVec &q)
    {
    //first, ensure that n is a unit vector
    scalar scale = sqrt(n.x*n.x+n.y*n.y);
    if(scale == 0) // this shouldn't be called. ifit is, by default make \hat{n} the x-direction
        {
        q[0] = S0; q[1] = 0.;
        return;
        }
    else if (scale != 1)
        {
        n.x /=scale;n.y /=scale;
        }
    q[0] = 2.0*S0*(n.x*n.x-0.5);
    q[1] = 2.0*S0*n.x*n.y;
    };

//!Tr(Q^2) = Q_{kl}Q_{lk}
HOSTDEVICE scalar Tr2DQ2(dVec &q)
    {
    return 2.0*(q[0]*q[0] + q[1]*q[1]);
    };

//!Tr(Q^3)
HOSTDEVICE scalar Tr2DQ3(dVec &q)
    {
    return .0;
    };

//!(Tr(Q^2))^2
HOSTDEVICE scalar Tr2DQ2Squared(dVec &q)
    {
    scalar Q2 = Tr2DQ2(q);
    return Q2*Q2;
    };

//!derivative of Tr(Q^2) w/r/t q[0],q[1]
HOSTDEVICE dVec derivativeTr2DQ2(dVec &q)
    {
    dVec ans;
    ans[0] = 4.*q[0];
    ans[1] = 4.*q[1];
    return ans;
    };

//!derivative of Tr(Q^3) w/r/t q[0], q[1]
HOSTDEVICE dVec derivativeTr2DQ3(dVec &q)
    {
    dVec ans;
    ans[0] = 0.;
    ans[1] = 0.;
    return ans;
    };

//!derivative of (Tr(Q^2))^2 w/r/t q[0], q[1]
HOSTDEVICE dVec derivativeTr2DQ2Squared(dVec &q)
    {
    scalar squares = q[0]*q[0] + q[1]*q[1];
    dVec ans;
    ans[0] = 16.0*q[0]*squares;
    ans[1] = 16.0*q[1]*squares;
    return ans;
    };

//!Phase components combined into one for computational efficiency
HOSTDEVICE dVec allPhaseComponentForces2D(dVec &q, scalar &a, scalar &b, scalar &c)
    {
    UNWRITTENCODE("allPhaseComponentForces2D");
    scalar squares = q[0]*q[0]+q[1]*q[1];
    
    dVec ans;
    ans[0] = 0.0;
    ans[1] = 0.0;
    return ans;
    }

//!Q_{jk}Q_{ki}
HOSTDEVICE dVec QjkQki2d(dVec &q)
    {
    dVec ans;
    ans[0] = q[0]*q[0] + q[1]*q[1];
    ans[1] = 0;
    return ans;
    };

//! determinant of a qt matrix
HOSTDEVICE scalar determinantOf2DQ(dVec &q)
    {
    return -q[0]*q[0] - q[1]*q[1];
    };

//!get the eigensystem associated with a Q tensor
HOSTDEVICE void eigensystemOfQ2D(dVec &q, vector<scalar> &eVals,
                                vector<scalar> &eVec1, vector<scalar> &eVec2)
    {

    if(eVals.size()!=2) eVals.resize(2);
    if(eVec1.size()!=2) eVec1.resize(2);
    if(eVec2.size()!=2) eVec2.resize(2);
    scalar squares = sqrt(q[0]*q[0] + q[1]*q[1]);
    eVals[0] = -squares;
    eVals[1] = squares;

    eVec1[0] = q[0] - squares;
    eVec1[1] = q[1];
    eVec2[0] = q[0] + squares;
    eVec2[1] = q[1];
    }

//!Get the eigenvalues of a real symmetric traceless 2x2 matrix
HOSTDEVICE void eigenvaluesOfQ2D(dVec &q,scalar &a,scalar &b)
    {
    scalar squares = sqrt(q[0]*q[0] + q[1]*q[1]);
    a = -squares;
    b=squares;
    }

/** @} */ //end of group declaration
#undef HOSTDEVICE
#endif
