#ifndef QTENSORFUNCTIONSGartland_H
#define QTENSORFUNCTIONSGartland_H

#include "std_include.h"
#include "symmetric3x3Eigensolver.h"

/*!
The Q-tensor has five independent components, which will get passed around in dVec structures... here we are using the Gartland basis, in which the space of symmetric traceless tensors is spanned by

    ((-3+sqrt(3))/6          0              0     )
E0= (     0            (3+sqrt(3)/6         0     )
    (     0                  0          -1/sqrt(3))

    ((3+sqrt(3))/6           0              0     )
E1= (     0            (-3+sqrt(3)/6        0     )
    (     0                  0          -1/sqrt(3))

    (     0        1/sqrt(2)      0   )
E2= ( 1/sqrt(2)       0           0   )
    (     0           0           0   )

    (     0           0     1/sqrt(2) )
E3= (     0           0         0     )
    ( 1/sqrt(2)       0         0     )

    (     0           0         0      )
E4= (     0           0     1/sqrt(2)  )
    (     0       1/sqrt(2)     0      )


A dVec of q[0,1,2,3,4] corresponds to the symmetric traceless tensor given by

Q = q[0]*E0 + q[1]*E1 + q[2]*E2 + q[3]*E3 + q[4]*E4

This file implements handy manipulations and functions of the Q-tensor as laid out this way
 */

#pragma hd_warning_disable
#pragma diag_suppress 2739

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file qTensorFunctionsGartlandBasis.h */

//! return a Gartland basis components for a Q tensor corresponding to a director and a value of S0
HOSTDEVICE void qTensorFromDirector(scalar3 n, scalar S0, dVec &q)
    {
    //first, ensure that n is a unit vector
    scalar scale = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if(scale == 0) // this shouldn't be called. ifit is, by default make \hat{[}n} the z-direction
        {
        q[0] = -sqrt3*S0*.5; q[1] = -sqrt3*S0*.5; q[2] = 0.0; q[3] = 0.0; q[4] = 0.0; 
        }
    else if (scale != 1)
        {
        n.x /=scale;n.y /=scale;n.z /=scale;
        }
    q[0] = 0.25*S0*((sqrt3-3.)*nx*nx + (sqrt3+3.)*ny*ny - 2.*sqrt3*nz*nz);
    q[1] = 0.25*S0*((sqrt3+3.)*nx*nx + (sqrt3-3.)*ny*ny - 2.*sqrt3*nz*nz);
    q[2] = (3.0/sqrt2)*S0*nx*ny;
    q[3] = (3.0/sqrt2)*S0*nx*nz;
    q[4] = (3.0/sqrt2)*S0*ny*nz;
    };

//!Tr(Q^2) = Q_{kl}Q_{lk}
HOSTDEVICE scalar TrQ2(dVec &q)
    {
    return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4];
    };

//!Tr(Q^3)
HOSTDEVICE scalar TrQ3(dVec &q)
    {
    return (2*sqrt3*(q[0]*q[0]*q[0]) + 2*sqrt3*(q[1]*q[1]*q[1]) - 3*(2*sqrt3*(q[1]*q[1]) - 2*sqrt3*(q[2]*q[2]) + (3 + sqrt3)*(q[3]*q[3]) + (-3 + sqrt3)*(q[4]*q[4]))*q[0] - 6*sqrt3*(q[0]*q[0])*q[1] + (6*sqrt3*(q[2]*q[2]) - 3*(-3 + sqrt3)*(q[3]*q[3]) - 3*(3 + sqrt3)*(q[4]*q[4]))*q[1] + 18*sqrt2*q[2]*q[3]*q[4])/12.;
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
    ans[0] = 2.*q[0];
    ans[1] = 2.*q[1];
    ans[2] = 2.*q[2];
    ans[3] = 2.*q[3];
    ans[4] = 2.*q[4];
    return ans;
    };

//!derivative of Tr(Q^3) w/r/t q[0] .. q[4]
HOSTDEVICE dVec derivativeTrQ3(dVec &q)
    {
    dVec ans;
    ans[0] = (6*sqrt3*(q[0]*q[0]) - 3*(2*sqrt3*(q[1]*q[1]) - 2*sqrt3*(q[2]*q[2]) + (3 + sqrt3)*(q[3]*q[3]) + (-3 + sqrt3)*(q[4]*q[4])) - 12*sqrt3*q[0]*q[1])/12.;
    ans[1] = (-6*sqrt3*(q[0]*q[0]) + 6*sqrt3*(q[1]*q[1]) + 6*sqrt3*(q[2]*q[2]) - 3*(-3 + sqrt3)*(q[3]*q[3]) - 3*(3 + sqrt3)*(q[4]*q[4]) - 12*sqrt3*q[0]*q[1])/12.;
    ans[2] = (12*sqrt3*q[0]*q[2] + 12*sqrt3*q[1]*q[2] + 18*sqrt2*q[3]*q[4])/12.;
    ans[3] = (-6*(3 + sqrt3)*q[0]*q[3] - 6*(-3 + sqrt3)*q[1]*q[3] + 18*sqrt2*q[2]*q[4])/12.;
    ans[4] = (18*sqrt2*q[2]*q[3] - 6*(-3 + sqrt3)*q[0]*q[4] - 6*(3 + sqrt3)*q[1]*q[4])/12.;
    return ans;
    };

//!derivative of (Tr(Q^2))^2 w/r/t q[0] .. q[4]
HOSTDEVICE dVec derivativeTrQ2Squared(dVec &q)
    {
    scalar squares = 4.*(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]+q[4]*q[4]);
    dVec ans;
    ans[0] = q[0]*squares;
    ans[1] = q[1]*squares;
    ans[2] = q[2]*squares;
    ans[3] = q[3]*squares;
    ans[4] = q[4]*squares;
    return ans;
    };

//!Phase components combined into one for computational efficiency
HOSTDEVICE dVec allPhaseComponentForces(dVec &q, scalar &a, scalar &b, scalar &c)
    {
    scalar q02=q[0]*q[0];
    scalar q12=q[1]*q[1];
    scalar q22=q[2]*q[2];
    scalar q32=q[3]*q[3];
    scalar q42=q[4]*q[4];
    scalar squares = q02+q12+q22+q32+q42;
    
    dVec ans;
    ans[0]= 2*a*q[0] + 4*c*(squares)*q[0] + (b*(2*sqrt3*(q22) - (3 + sqrt3)*(q32) - (-3 + sqrt3)*(q42) + 2*sqrt3*(q02 - q12 - 2*q[0]*q[1])))/4.;
    ans[1]= 2*(a + 2*c*(squares))*q[1] - (b*(-2*sqrt3*(q22) + (-3 + sqrt3)*(q32) + (3 + sqrt3)*(q42) + 2*sqrt3*(q02 - q12 + 2*q[0]*q[1])))/4.;
    ans[2]= 2*a*q[2] + 4*c*(squares)*q[2] + sqrt3*b*(q[0] + q[1])*q[2] + (3*b*q[3]*q[4])/sqrt2;
    ans[3]= 2*a*q[3] + 4*c*(squares)*q[3] - (b*((3 + sqrt3)*q[0] + (-3 + sqrt3)*q[1])*q[3])/2. + (3*b*q[2]*q[4])/sqrt2;
    ans[4]= 2*(a + 2*c*(squares))*q[4] + (b*(3*sqrt2*q[2]*q[3] - ((-3 + sqrt3)*q[0] + (3 + sqrt3)*q[1])*q[4]))/2.;
    return ans;
    }

//! determinant of a qt matrix
HOSTDEVICE scalar determinantOfQ(dVec &q)
    {
    return (2*sqrt3*(q[0]*q[0]*q[0]) + 2*sqrt3*(q[1]*q[1]*q[1]) - 3*(2*sqrt3*(q[1]*q[1]) - 2*sqrt3*(q[2]*q[2]) + (3 + sqrt3)*(q[3]*q[3]) + (-3 + sqrt3)*(q[4]*q[4]))*q[0] - 6*sqrt3*(q[0]*q[0])*q[1] + (6*sqrt3*(q[2]*q[2]) - 3*(-3 + sqrt3)*(q[3]*q[3]) - 3*(3 + sqrt3)*(q[4]*q[4]))*q[1] + 18*sqrt2*q[2]*q[3]*q[4])/36.;
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
    eigenSolver(((-3 + sqrt3)*q[0] + (3 + sqrt3)*q[1])/6.,
                q[2]/sqrt2,
                q[3]/sqrt2,
                ((3 + sqrt3)*q[0] + (-3 + sqrt3)*q[1])/6.,
                q[4]/sqrt2,
                -((q[0] + q[1])/sqrt3),
                evals,evecs);

    eVals[0]=evals[0];eVals[1]=evals[1];eVals[2]=evals[2];
    eVec1[0]=evecs[0][0];eVec1[1]=evecs[0][1];eVec1[2]=evecs[0][2];
    eVec2[0]=evecs[1][0];eVec2[1]=evecs[1][1];eVec2[2]=evecs[1][2];
    eVec3[0]=evecs[2][0];eVec3[1]=evecs[2][1];eVec3[2]=evecs[2][2];
    }

//!Get the eigenvalues of a real symmetric traceless 3x3 matrix
HOSTDEVICE void eigenvaluesOfQ(dVec &q,scalar &a,scalar &b,scalar &c)
    {
    scalar Qxx =((-3 + sqrt3)*q[0] + (3 + sqrt3)*q[1])/6.;
    scalar Qxy = q[2]/sqrt2;
    scalar Qxz = q[3]/sqrt2;
    scalar Qyy = ((3 + sqrt3)*q[0] + (-3 + sqrt3)*q[1])/6.;
    scalar Qyz = q[4]/sqrt2;
    scalar p1 = Qxy*Qxy + Qxz*Qxz + Qyz*Qyz;
    if(p1 ==0) // diagonal matrix case
        {
        a=Qxx;
        b=Qyy;
        c=-Qxx-Qyy;
        return;
        }
    //since q is traceless, some of these expressions are simpler than expected
    scalar p2 = 2*(Qxx*Qxx+Qyy*Qyy + Qxx*Qyy) + 2*p1;
    scalar p = sqrt(p2/6.0);
    dVec B; B[0]=Qxx; B[1]=Qxy; B[2]=Qxz; B[3] = Qyy; B[4]=Qyz;
    B = (1.0/p) * B;
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


#endif
