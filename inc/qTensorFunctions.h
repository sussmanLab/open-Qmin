#ifndef QTENSORFUNCTIONS_H
#define QTENSORFUNCTIONS_H

#include "std_include.h"
#include "symmetric3x3Eigensolver.h"

/*!
The Q-tensor has five independent components, which will get passed around in dVec structures...
a dVec of q[0,1,2,3,4] corresponds to the symmetric traceless tensor laid out as
    (q[0]    q[1]        q[2]    )
Q = (q[1]    q[3]        q[4]    )
    (q[2]    q[4]   -(q[0]+q[3]) )

This file implements handy manipulations and functions of the Q-tensor as laid out this way
 */

#pragma hd_warning_disable
#pragma diag_suppress 2739

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file qTensorFunctions.h */

/** @defgroup Functions functions
 * @{
 \brief Utility functions that can be called from host or device
 */

//! return a qtensor given a director and a value of S0
HOSTDEVICE void qTensorFromDirector(scalar3 n, scalar S0, dVec &q)
    {
    //first, ensure that n is a unit vector
    scalar scale = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if(scale == 0) // this shouldn't be called. ifit is, by default make \hat{[}n} the z-direction
        {
        q[0] = -0.5*S0; q[3] = -0.5*S0;
        }
    else if (scale != 1)
        {
        n.x /=scale;n.y /=scale;n.z /=scale;
        }
    q[0] = 1.5*S0*(n.x*n.x-1.0/3.0);
    q[1] = 1.5*S0*n.x*n.y;
    q[2] = 1.5*S0*n.x*n.z;
    q[3] = 1.5*S0*(n.y*n.y-1.0/3.0);
    q[4] = 1.5*S0*n.y*n.z;
    };

//!Tr(Q^2) = Q_{kl}Q_{lk}
HOSTDEVICE scalar TrQ2(dVec &q)
    {
    return 2.0*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3]);
    };

//!Tr(Q^3)
HOSTDEVICE scalar TrQ3(dVec &q)
    {
    return -3.0*((-(q[1]*q[1]) + q[3]*q[3] + q[4]*q[4])*q[0] + q[0]*q[0]*q[3] - q[1]*q[1]*q[3] + q[2]*q[2]*q[3] - 2*q[1]*q[2]*q[4]);
    };

//!(Tr(Q^2))^2
HOSTDEVICE scalar TrQ2Squared(dVec &q)
    {
    scalar Q2 = TrQ2(q);
    return Q2*Q2;
    };

//!derivative of Tr(Q^2) w/r/t q[0] .. q[4]
HOSTDEVICE dVec derivativeTrQ2(dVec &q)
    {
    dVec ans;
    ans[0] = 2.*(2*q[0] + q[3]);
    ans[1] = 4.*q[1];
    ans[2] = 4.*q[2];
    ans[3] = 2.*(q[0] + 2*q[3]);
    ans[4] = 4.*q[4];
    return ans;
    };

//!derivative of Tr(Q^3) w/r/t q[0] .. q[4]
HOSTDEVICE dVec derivativeTrQ3(dVec &q)
    {
    dVec ans;
    ans[0] = -3.0*(-q[1]*q[1] + q[3]*q[3] + q[4]*q[4] + 2*q[0]*q[3]);
    ans[1] = 6*(q[0]*q[1] + q[1]*q[3] + q[2]*q[4]);
    ans[2] = -6*q[2]*q[3] + 6*q[1]*q[4];
    ans[3] = -3*(q[0]*q[0] - q[1]*q[1] + q[2]*q[2] + 2*q[0]*q[3]);
    ans[4] = 6*q[1]*q[2] - 6*q[0]*q[4];
    return ans;
    };

//!derivative of (Tr(Q^2))^2 w/r/t q[0] .. q[4]
HOSTDEVICE dVec derivativeTrQ2Squared(dVec &q)
    {
    scalar squares = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]+q[4]*q[4]+q[0]*q[3];
    dVec ans;
    ans[0] = 8.0*(2.0*q[0]+q[3])*squares;
    ans[1] = 16*q[1]*squares;
    ans[2] = 16*q[2]*squares;
    ans[3] = 8.0*(q[0]+2.0*q[3])*squares;
    ans[4] = 16*q[4]*squares;
    return ans;
    };

//!Phase components combined into one for computational efficiency
HOSTDEVICE dVec allPhaseComponentForces(dVec &q, scalar &a, scalar &b, scalar &c)
    {
    scalar squares = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]+q[4]*q[4]+q[0]*q[3];
    
    dVec ans;
    ans[0] = -a*2.*(2. *q[0] + q[3]) +b*3.0*(-q[1]*q[1] + q[3]*q[3] + q[4]*q[4] + 2*q[0]*q[3]) - c*8.0*(2.0*q[0]+q[3])*squares;
    ans[1] = -a*4.*q[1]             -b*6.*(q[0]*q[1] + q[1]*q[3] + q[2]*q[4]) - c*16.*q[1]*squares;
    ans[2] = -a*4.*q[2]             +b*6.*q[2]*q[3] + 6.*q[1]*q[4] - c*16.*q[2]*squares;
    ans[3] = -a*2.*(q[0] + 2.*q[3]) +b*3.*(q[0]*q[0] - q[1]*q[1] + q[2]*q[2] + 2.*q[0]*q[3]) - c*8.0*(q[0]+2.0*q[3])*squares;
    ans[4] = -a*4.*q[4]             -b*6.*q[1]*q[2] - 6.*q[0]*q[4] - c*16.*q[4]*squares;
    return ans;
    }

//!Q_{jk}Q_{ki}
HOSTDEVICE dVec QjkQki(dVec &q)
    {
    dVec ans;
    ans[0] = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    ans[1] = q[0]*q[1] + q[1]*q[3] + q[2]*q[4];
    ans[2] = q[0]*q[2] - q[2]*(q[0]+q[3]) + q[1]*q[4];
    ans[3] = q[1]*q[1] + q[3]*q[3] + q[4]*q[4];
    ans[4] = q[1]*q[2] - q[4]*(q[0]+q[3]) + q[3]*q[4];
    return ans;
    };

//! determinant of a qt matrix
HOSTDEVICE scalar determinantOfQ(dVec &q)
    {
    return q[3]*(q[1]*q[1] - q[0]*q[0] - q[2]*q[2]) + 2*q[1]*q[2]*q[4] +
           q[0]*(q[1]*q[1] - q[3]*q[3] - q[4]*q[4]);
    };

//!eVec is Q*e, ev0,ev1,ev2 are components ofe
HOSTDEVICE scalar eigFromVecs(vector<scalar> &eVec, scalar ev0, scalar ev1, scalar ev2)
    {
    scalar ans = 0.0;
    if(eVec[0]!=0)
        ans = eVec[0]/ev0;
    else if (eVec[1]!=0)
        ans = eVec[1]/ev1;
    else if (eVec[2]!=0)
        ans = eVec[2]/ev2;
    return ans;
    }

//!get the eigensystem associated with a Q tensor
HOSTDEVICE void eigensystemOfQ(dVec &q, vector<scalar> &eVals,
                                vector<scalar> &eVec1, vector<scalar> &eVec2, vector<scalar> &eVec3)
    {

    std::array<scalar, 3> evals;
    std::array<std::array<scalar, 3>, 3> evecs;

    NISymmetricEigensolver3x3 eigenSolver;
    eigenSolver(q[0],q[1],q[2],q[3],q[4],-q[0]-q[3],evals,evecs);

    eVals[0]=evals[0];eVals[1]=evals[1];eVals[2]=evals[2];
    eVec1[0]=evecs[0][0];eVec1[1]=evecs[0][1];eVec1[2]=evecs[0][2];
    eVec2[0]=evecs[1][0];eVec2[1]=evecs[1][1];eVec2[2]=evecs[1][2];
    eVec3[0]=evecs[2][0];eVec3[1]=evecs[2][1];eVec3[2]=evecs[2][2];
/*

    eVec1[0] = q[0]*evecs[0][0]+q[1]*evecs[0][1] + q[2]*evecs[0][2];
    eVec1[1] = q[1]*evecs[0][0]+q[3]*evecs[0][1] + q[4]*evecs[0][2];
    eVec1[2] = q[2]*evecs[0][0]+q[4]*evecs[0][1] + (-q[0]-q[3])*evecs[0][2];

    eVec2[0] = q[0]*evecs[1][0]+q[1]*evecs[1][1] + q[2]*evecs[1][2];
    eVec2[1] = q[1]*evecs[1][0]+q[3]*evecs[1][1] + q[4]*evecs[1][2];
    eVec2[2] = q[2]*evecs[1][0]+q[4]*evecs[1][1] + (-q[0]-q[3])*evecs[1][2];

    eVec3[0] = q[0]*evecs[2][0]+q[1]*evecs[2][1] + q[2]*evecs[2][2];
    eVec3[1] = q[1]*evecs[2][0]+q[3]*evecs[2][1] + q[4]*evecs[2][2];
    eVec3[2] = q[2]*evecs[2][0]+q[4]*evecs[2][1] + (-q[0]-q[3])*evecs[2][2];

    scalar a1 =  eigFromVecs(eVec1,evecs[0][0],evecs[0][1],evecs[0][2]);
    scalar a2 =  eigFromVecs(eVec2,evecs[1][0],evecs[1][1],evecs[1][2]);
    scalar a3 =  eigFromVecs(eVec3,evecs[2][0],evecs[2][1],evecs[2][2]);

    vector<pair<scalar,int> > eigList(3);
    eigList[0] = make_pair(a1,0);
    eigList[1] = make_pair(a2,1);
    eigList[2] = make_pair(a3,2);
    sort(eigList.begin(),eigList.end());
    eVec1[0] = evecs[eigList[0].second][0];
    eVec1[1] = evecs[eigList[0].second][1];
    eVec1[2] = evecs[eigList[0].second][2];

    eVec2[0] = evecs[eigList[1].second][0];
    eVec2[1] = evecs[eigList[1].second][1];
    eVec2[2] = evecs[eigList[1].second][2];

    eVec3[0] = evecs[eigList[2].second][0];
    eVec3[1] = evecs[eigList[2].second][1];
    eVec3[2] = evecs[eigList[2].second][2];


    printf("%f %f %f\n",eVals[0],eVals[1],eVals[2]);
    printf("%f %f %f\t\n",eVec1[0],eVec1[1],eVec1[2]);
    printf("%f %f %f\t\n",eVec2[0],eVec2[1],eVec2[2]);
    printf("%f %f %f\t\n",eVec3[0],eVec3[1],eVec3[2]);
    */

    }

//!Get the eigenvalues of a real symmetric traceless 3x3 matrix
HOSTDEVICE void eigenvaluesOfQ(dVec &q,scalar &a,scalar &b,scalar &c)
    {
    scalar p1 = q[1]*q[1] + q[2]*q[2]+ q[4]*q[4];
    if(p1 ==0) // diagonal matrix case
        {
        a=q[0];
        b=q[3];
        c=-q[0]-q[3];
        return;
        }
    //since q is traceless, some of these expressions are simpler than expected
    scalar p2 = 2*(q[0]*q[0]+q[3]*q[3] + q[0]*q[3]) + 2*p1;
    scalar p = sqrt(p2/6.0);
    dVec B = (1.0/p) * q;
    scalar r = 0.5*determinantOfQ(B);
    scalar phi;
    if(r <= -1)
        phi = PI/3.0;
    else if (r >= 1)
        phi = 0.0;
    else
        phi = acos(r)/3.0;

    a= 2.0*p*cos(phi);
    c= 2.0*p*cos(phi+2.0*PI/3.0);
    b=-a-c;
    }

/** @} */ //end of group declaration
#undef HOSTDEVICE
#endif
