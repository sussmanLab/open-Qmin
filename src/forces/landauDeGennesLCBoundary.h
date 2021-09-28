#ifndef landauDeGennesLCBoundary_H
#define landauDeGennesLCBoundary_H

#include "latticeBoundaries.h"
#include "qTensorFunctions.h"
#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif
/*! \file landauDeGennesLCBoundary.h */

//!compute the force on lattice site exerted by the boundary site
/*!
these boundary force and energy functions use the information at the boundary site to pass
additional information. For instance, given homeotropic anchoring the boundary site should
have a q tensor which is Q^B = 3 S_0/2*(\nu^s \nu^s - \delta_{ab}/3), where \nu^s is the
locally preferred director.
For degenerate planar anchoring the boundary site should be,
Q^B[0] = \hat{nu}_x
Q^B[1] = \hat{nu}_y
Q^B[2] = \hat{nu}_z
where \nu^s = {Cos[\[Phi]] Sin[\[theta]], Sin[\[Phi]] Sin[\[theta]], Cos[\[theta]]}
 is the direction to which the LC should try to be orthogonal
*/
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
        case boundaryType::degeneratePlanar:
            {
            scalar W1 = bObj.P1;
            scalar W2 = bObj.P1;
            scalar S0 = bObj.P2;
            scalar nuX = boundarySite.x[0];
            scalar nuY = boundarySite.x[1];
            scalar nuZ = boundarySite.x[2];
            scalar nuX2 = nuX*nuX;
            scalar nuX3 = nuX2*nuX;
            scalar nuY2 = nuY*nuY;
            scalar nuY3 = nuY2*nuY;
            scalar nuZ2 = nuZ*nuZ;
            scalar nuZ3 = nuZ2*nuZ;
            scalar nuXY = nuX*nuY;
            scalar nuXZ = nuX*nuZ;
            scalar nuYZ = nuY*nuZ;
            dVec q = latticeSite;

            force[0]=2*W2*(2*q[0] + q[3])*(3*(S0*S0) - 4*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3])) - W1*((2 + nuX2 - nuZ2)*(nuZ3)*(nuZ3*(S0 - 2*(q[0] + q[3])) + nuZ*(S0*(-2 + nuX2 + nuY2) + 2*((2 + nuX2)*q[0] + 2*nuXY*q[1] + (2 + nuY2)*q[3])) - 4*(nuX*q[2] + nuY*q[4]) + 4*(nuZ2)*(nuX*q[2] + nuY*q[4])) + nuY3*(nuX2 - nuZ2)*(nuY3*(S0 + 2*q[3]) + nuY*(S0*(-2 + nuX2 + nuZ2) + 2*(nuX2)*q[0] + 4*nuXZ*q[2] - 2*(2*q[3] + nuZ2*(q[0] + q[3]))) - 4*(nuX*q[1] + nuZ*q[4]) + 4*(nuY2)*(nuX*q[1] + nuZ*q[4])) + 2*nuXY*(-1 + nuX2 - nuZ2)*(nuY*(nuX3)*(S0 + 2*q[0]) - 2*nuY*(nuY*q[1] + nuZ*q[2]) + nuX2*((-2 + 4*(nuY2))*q[1] + 4*nuYZ*q[2]) + nuX*(nuY3*(S0 + 2*q[3]) + nuY*(S0*(-2 + nuZ2) - 2*(1 + nuZ2)*(q[0] + q[3])) - 2*nuZ*q[4] + 4*nuZ*(nuY2)*q[4])) + 2*nuXZ*(nuX2 - nuZ2)*(nuZ*(nuX3)*(S0 + 2*q[0]) - 2*nuZ*(nuY*q[1] + nuZ*q[2]) + nuX2*(4*nuYZ*q[1] + 2*(-1 + 2*(nuZ2))*q[2]) + nuX*(nuZ*(S0*(-2 + nuY2) + 2*(1 + nuY2)*q[3]) + nuZ3*(S0 - 2*(q[0] + q[3])) - 2*nuY*q[4] + 4*nuY*(nuZ2)*q[4])) + 2*nuYZ*(1 + nuX2 - nuZ2)*(nuZ*(nuY3)*(S0 + 2*q[3]) + nuY*(nuZ*(S0*(-2 + nuX2) + 2*(1 + nuX2)*q[0]) - 2*nuX*q[2] + 4*nuX*(nuZ2)*q[2] + nuZ3*(S0 - 2*(q[0] + q[3]))) - 2*nuZ*(nuX*q[1] + nuZ*q[4]) + nuY2*(4*nuXZ*q[1] + 2*(-1 + 2*(nuZ2))*q[4])) + nuX3*(-2 + nuX2 - nuZ2)*(nuX3*(S0 + 2*q[0]) - 4*(nuY*q[1] + nuZ*q[2]) + 4*(nuX2)*(nuY*q[1] + nuZ*q[2]) + nuX*(S0*(-2 + nuY2 + nuZ2) - 2*((2 + nuZ2)*q[0] - nuY2*q[3] + nuZ2*q[3] - 2*nuYZ*q[4]))));
            force[1]=4*W2*q[1]*(3*(S0*S0) - 4*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3])) - 2*W1*(pow(nuX,7)*nuY*(S0 + 2*q[0]) + 4*pow(nuX,6)*nuY*(nuY*q[1] + nuZ*q[2]) + 2*nuY*(nuY2 + nuZ2)*(nuY*q[1] + nuZ*q[2]) + 2*(nuX2)*(2*pow(nuY,6)*q[1] + 2*(nuY2)*(3 + pow(nuZ,4) - 4*(nuZ2))*q[1] + 4*pow(nuY,4)*(-2 + nuZ2)*q[1] + nuZ2*q[1] + 2*pow(nuY,5)*nuZ*q[2] + nuYZ*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*q[2] + 4*nuZ*(nuY3)*(-2 + nuZ2)*q[2]) + 2*pow(nuX,4)*((1 + 4*pow(nuY,4) + 4*(nuY2)*(-2 + nuZ2))*q[1] + 4*nuYZ*(-2 + nuY2 + nuZ2)*q[2]) + nuX3*(pow(nuY,5)*(3*S0 + 2*q[0] + 4*q[3]) + nuY*(S0*(4 + 3*pow(nuZ,4) - 8*(nuZ2)) - 2*(-3 + pow(nuZ,4))*q[0] + 2*(1 - 2*pow(nuZ,4) + 4*(nuZ2))*q[3]) + nuY3*(S0*(-8 + 6*(nuZ2)) - 8*(q[0] + q[3])) + 2*nuZ*q[4] + 8*pow(nuY,4)*nuZ*q[4] + 8*nuZ*(nuY2)*(-2 + nuZ2)*q[4]) + nuX*(pow(nuY,5)*(S0*(-4 + 3*(nuZ2)) - 2*(nuZ2)*(q[0] - q[3]) - 8*q[3]) + pow(nuY,7)*(S0 + 2*q[3]) + nuY3*(S0*(4 + 3*pow(nuZ,4) - 8*(nuZ2)) + (2 - 4*pow(nuZ,4) + 8*(nuZ2))*q[0] - 2*(-3 + pow(nuZ,4))*q[3]) + nuY*(nuZ2)*(S0*((-2 + nuZ2)*(-2 + nuZ2)) - 2*(1 + pow(nuZ,4) - 4*(nuZ2))*(q[0] + q[3])) + 4*pow(nuY,6)*nuZ*q[4] + 2*nuZ*(nuY2)*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*q[4] + 8*pow(nuY,4)*nuZ*(-2 + nuZ2)*q[4] + 2*(nuZ3)*q[4]) + pow(nuX,5)*nuY*(S0*(-4 + 3*(nuY2) + 3*(nuZ2)) + 2*((-4 + 2*(nuY2) + nuZ2)*q[0] + nuY2*q[3] - nuZ2*q[3] + 2*nuYZ*q[4])));
            force[2]=4*W2*q[2]*(3*(S0*S0) - 4*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3])) - 2*W1*(pow(nuX,7)*nuZ*(S0 + 2*q[0]) + 4*pow(nuX,6)*nuZ*(nuY*q[1] + nuZ*q[2]) + 2*nuZ*(nuY2 + nuZ2)*(nuY*q[1] + nuZ*q[2]) + 2*pow(nuX,4)*(4*nuZ*(nuY3)*q[1] + 4*nuYZ*(-2 + nuZ2)*q[1] + (1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[2] + 4*(nuY2)*(nuZ2)*q[2]) + 2*(nuX2)*(2*pow(nuY,5)*nuZ*q[1] + nuYZ*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*q[1] + 4*nuZ*(nuY3)*(-2 + nuZ2)*q[1] + nuY2*(1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[2] + 2*pow(nuY,4)*(nuZ2)*q[2] + 2*(3 + pow(nuZ,4) - 4*(nuZ2))*(nuZ2)*q[2]) + nuX3*(nuZ3*(S0*(-8 + 6*(nuY2)) + 8*q[3]) + nuZ*(S0*(4 + 3*pow(nuY,4) - 8*(nuY2)) + 2*(2 + pow(nuY,4) - 4*(nuY2))*q[0] + 2*(-1 + 2*pow(nuY,4) - 4*(nuY2))*q[3]) + pow(nuZ,5)*(3*S0 - 2*(q[0] + 2*q[3])) + 2*nuY*q[4] + 8*nuY*pow(nuZ,4)*q[4] + 8*nuY*(-2 + nuY2)*(nuZ2)*q[4]) + nuX*(pow(nuY,4)*nuZ*(S0*(-4 + 3*(nuZ2)) - 2*(nuZ2)*(q[0] - q[3]) - 8*q[3]) + pow(nuY,6)*nuZ*(S0 + 2*q[3]) + nuZ*(nuY2)*(S0*(4 + 3*pow(nuZ,4) - 8*(nuZ2)) + 8*(nuZ2)*q[0] + 2*q[3] - 2*pow(nuZ,4)*(2*q[0] + q[3])) + nuZ3*(S0*((-2 + nuZ2)*(-2 + nuZ2)) - 2*((2 + pow(nuZ,4) - 4*(nuZ2))*q[0] + (3 + pow(nuZ,4) - 4*(nuZ2))*q[3])) + 2*(nuY3)*(1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[4] + 4*pow(nuY,5)*(nuZ2)*q[4] + 2*nuY*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*(nuZ2)*q[4]) + pow(nuX,5)*nuZ*(S0*(-4 + 3*(nuY2) + 3*(nuZ2)) + 2*((-4 + 2*(nuY2) + nuZ2)*q[0] + nuY2*q[3] - nuZ2*q[3] + 2*nuYZ*q[4])));
            force[3]=2*W2*(q[0] + 2*q[3])*(3*(S0*S0) - 4*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3])) - (W1*(4*(2 + nuY2 - nuZ2)*(nuZ3)*(nuZ3*(S0 - 2*(q[0] + q[3])) + nuZ*(S0*(-2 + nuX2 + nuY2) + 2*((2 + nuX2)*q[0] + 2*nuXY*q[1] + (2 + nuY2)*q[3])) - 4*(nuX*q[2] + nuY*q[4]) + 4*(nuZ2)*(nuX*q[2] + nuY*q[4])) + 4*(nuY3)*(-2 + nuY2 - nuZ2)*(nuY3*(S0 + 2*q[3]) + nuY*(S0*(-2 + nuX2 + nuZ2) + 2*(nuX2)*q[0] + 4*nuXZ*q[2] - 2*(2*q[3] + nuZ2*(q[0] + q[3]))) - 4*(nuX*q[1] + nuZ*q[4]) + 4*(nuY2)*(nuX*q[1] + nuZ*q[4])) + 8*nuXY*(-1 + nuY2 - nuZ2)*(nuY*(nuX3)*(S0 + 2*q[0]) - 2*nuY*(nuY*q[1] + nuZ*q[2]) + nuX2*((-2 + 4*(nuY2))*q[1] + 4*nuYZ*q[2]) + nuX*(nuY3*(S0 + 2*q[3]) + nuY*(S0*(-2 + nuZ2) - 2*(1 + nuZ2)*(q[0] + q[3])) - 2*nuZ*q[4] + 4*nuZ*(nuY2)*q[4])) + 8*nuXZ*(1 + nuY2 - nuZ2)*(nuZ*(nuX3)*(S0 + 2*q[0]) - 2*nuZ*(nuY*q[1] + nuZ*q[2]) + nuX2*(4*nuYZ*q[1] + 2*(-1 + 2*(nuZ2))*q[2]) + nuX*(nuZ*(S0*(-2 + nuY2) + 2*(1 + nuY2)*q[3]) + nuZ3*(S0 - 2*(q[0] + q[3])) - 2*nuY*q[4] + 4*nuY*(nuZ2)*q[4])) + 8*nuYZ*(nuY2 - nuZ2)*(nuZ*(nuY3)*(S0 + 2*q[3]) + nuY*(nuZ*(S0*(-2 + nuX2) + 2*(1 + nuX2)*q[0]) - 2*nuX*q[2] + 4*nuX*(nuZ2)*q[2] + nuZ3*(S0 - 2*(q[0] + q[3]))) - 2*nuZ*(nuX*q[1] + nuZ*q[4]) + nuY2*(4*nuXZ*q[1] + 2*(-1 + 2*(nuZ2))*q[4])) - 4*(nuX3)*(-(nuY2) + nuZ2)*(nuX3*(S0 + 2*q[0]) - 4*(nuY*q[1] + nuZ*q[2]) + 4*(nuX2)*(nuY*q[1] + nuZ*q[2]) + nuX*(S0*(-2 + nuY2 + nuZ2) - 2*((2 + nuZ2)*q[0] - nuY2*q[3] + nuZ2*q[3] - 2*nuYZ*q[4])))))/4.;
            force[4]=4*W2*(3*(S0*S0) - 4*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] + q[4]*q[4] + q[0]*q[3]))*q[4] - 2*W1*(pow(nuX,6)*nuYZ*(S0 + 2*q[0]) + 4*pow(nuX,5)*nuYZ*(nuY*q[1] + nuZ*q[2]) + 2*(nuX3)*(nuZ*(1 + 4*pow(nuY,4) - 8*(nuY2))*q[1] + 4*(nuY2)*(nuZ3)*q[1] + nuY*q[2] + 4*nuY*pow(nuZ,4)*q[2] + 4*nuY*(-2 + nuY2)*(nuZ2)*q[2]) + 2*nuX*(2*pow(nuY,6)*nuZ*q[1] + nuZ*(nuY2)*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*q[1] + 4*pow(nuY,4)*nuZ*(-2 + nuZ2)*q[1] + nuZ3*q[1] + nuY3*(1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[2] + 2*pow(nuY,5)*(nuZ2)*q[2] + nuY*(5 + 2*pow(nuZ,4) - 8*(nuZ2))*(nuZ2)*q[2]) + pow(nuY,5)*nuZ*(S0*(-4 + 3*(nuZ2)) - 2*(nuZ2)*(q[0] - q[3]) - 8*q[3]) + pow(nuY,7)*nuZ*(S0 + 2*q[3]) + nuZ*(nuY3)*(S0*(4 + 3*pow(nuZ,4) - 8*(nuZ2)) - 2*((1 + 2*pow(nuZ,4) - 4*(nuZ2))*q[0] + (-2 + pow(nuZ,4))*q[3])) + nuY*(nuZ3)*(S0*((-2 + nuZ2)*(-2 + nuZ2)) - 2*((3 + pow(nuZ,4) - 4*(nuZ2))*q[0] + (2 + pow(nuZ,4) - 4*(nuZ2))*q[3])) + 2*pow(nuZ,4)*q[4] + 2*pow(nuY,4)*(1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[4] + 4*pow(nuY,6)*(nuZ2)*q[4] + 4*(nuY2)*(3 + pow(nuZ,4) - 4*(nuZ2))*(nuZ2)*q[4] + nuX2*(pow(nuY,5)*nuZ*(3*S0 + 2*q[0] + 4*q[3]) + nuYZ*(S0*(4 + 3*pow(nuZ,4) - 8*(nuZ2)) - 2*(-1 + pow(nuZ,4))*q[0] - 4*(-2 + nuZ2)*(nuZ2)*q[3]) + 2*nuZ*(nuY3)*(S0*(-4 + 3*(nuZ2)) - 4*(q[0] + q[3])) + 2*(nuY2)*(1 + 4*pow(nuZ,4) - 8*(nuZ2))*q[4] + 2*(nuZ2)*q[4] + 8*pow(nuY,4)*(nuZ2)*q[4]) + pow(nuX,4)*nuYZ*(S0*(-4 + 3*(nuY2) + 3*(nuZ2)) + 2*((-4 + 2*(nuY2) + nuZ2)*q[0] + nuY2*q[3] - nuZ2*q[3] + 2*nuYZ*q[4])));
            break;
            }
            #ifndef __NVCC__
        default:
            UNWRITTENCODE("non-defined boundary type is attempting a force computation");
            #endif
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
        case boundaryType::degeneratePlanar:
            {
            scalar W1 = bObj.P1;
            scalar W2 = bObj.P1;
            scalar S0 = bObj.P2;
            scalar nuX = boundarySite.x[0];
            scalar nuY = boundarySite.x[1];
            scalar nuZ = boundarySite.x[2];
            scalar nuX2 = nuX*nuX;
            scalar nuX3 = nuX2*nuX;
            scalar nuY2 = nuY*nuY;
            scalar nuY3 = nuY2*nuY;
            scalar nuZ2 = nuZ*nuZ;
            scalar nuZ3 = nuZ2*nuZ;
            scalar nuXY = nuX*nuY;
            scalar nuXZ = nuX*nuZ;
            scalar nuYZ = nuY*nuZ;
            dVec q = latticeSite;

            scalar energy1 = 0.25*W1*(nuZ2*((nuZ3*(S0 - 2*(q[0] + q[3])) + nuZ*((-2 + nuX2 + nuY2)*S0 + 2*((2 + nuX2)*q[0] + 2*nuXY*q[1] + (2 + nuY2)*q[3])) - 4*(nuX*q[2] + nuY*q[4]) + 4*(nuZ2)*(nuX*q[2] + nuY*q[4]))*(nuZ3*(S0 - 2*(q[0] + q[3])) + nuZ*((-2 + nuX2 + nuY2)*S0 + 2*((2 + nuX2)*q[0] + 2*nuXY*q[1] + (2 + nuY2)*q[3])) - 4*(nuX*q[2] + nuY*q[4]) + 4*(nuZ2)*(nuX*q[2] + nuY*q[4]))) + nuY2*((nuY3*(S0 + 2*q[3]) + nuY*((-2 + nuX2 + nuZ2)*S0 + 2*(nuX2)*q[0] + 4*nuXZ*q[2] - 2*(2*q[3] + nuZ2*(q[0] + q[3]))) - 4*(nuX*q[1] + nuZ*q[4]) + 4*(nuY2)*(nuX*q[1] + nuZ*q[4]))*(nuY3*(S0 + 2*q[3]) + nuY*((-2 + nuX2 + nuZ2)*S0 + 2*(nuX2)*q[0] + 4*nuXZ*q[2] - 2*(2*q[3] + nuZ2*(q[0] + q[3]))) - 4*(nuX*q[1] + nuZ*q[4]) + 4*(nuY2)*(nuX*q[1] + nuZ*q[4]))) + 2*((nuX3*nuY*(S0 + 2*q[0]) - 2*nuY*(nuY*q[1] + nuZ*q[2]) + nuX2*((-2 + 4*(nuY2))*q[1] + 4*nuYZ*q[2]) + nuX*(nuY3*(S0 + 2*q[3]) + nuY*((-2 + nuZ2)*S0 - 2*(1 + nuZ2)*(q[0] + q[3])) - 2*nuZ*q[4] + 4*(nuY2)*nuZ*q[4]))*(nuX3*nuY*(S0 + 2*q[0]) - 2*nuY*(nuY*q[1] + nuZ*q[2]) + nuX2*((-2 + 4*(nuY2))*q[1] + 4*nuYZ*q[2]) + nuX*(nuY3*(S0 + 2*q[3]) + nuY*((-2 + nuZ2)*S0 - 2*(1 + nuZ2)*(q[0] + q[3])) - 2*nuZ*q[4] + 4*(nuY2)*nuZ*q[4]))) + 2*((nuX3*nuZ*(S0 + 2*q[0]) - 2*nuZ*(nuY*q[1] + nuZ*q[2]) + nuX2*(4*nuYZ*q[1] + 2*(-1 + 2*(nuZ2))*q[2]) + nuX*(nuZ*((-2 + nuY2)*S0 + 2*(1 + nuY2)*q[3]) + nuZ3*(S0 - 2*(q[0] + q[3])) - 2*nuY*q[4] + 4*nuY*(nuZ2)*q[4]))*(nuX3*nuZ*(S0 + 2*q[0]) - 2*nuZ*(nuY*q[1] + nuZ*q[2]) + nuX2*(4*nuYZ*q[1] + 2*(-1 + 2*(nuZ2))*q[2]) + nuX*(nuZ*((-2 + nuY2)*S0 + 2*(1 + nuY2)*q[3]) + nuZ3*(S0 - 2*(q[0] + q[3])) - 2*nuY*q[4] + 4*nuY*(nuZ2)*q[4]))) + 2*((nuY3*nuZ*(S0 + 2*q[3]) + nuY*(nuZ*((-2 + nuX2)*S0 + 2*(1 + nuX2)*q[0]) - 2*nuX*q[2] + 4*nuX*(nuZ2)*q[2] + nuZ3*(S0 - 2*(q[0] + q[3]))) - 2*nuZ*(nuX*q[1] + nuZ*q[4]) + nuY2*(4*nuXZ*q[1] + 2*(-1 + 2*(nuZ2))*q[4]))*(nuY3*nuZ*(S0 + 2*q[3]) + nuY*(nuZ*((-2 + nuX2)*S0 + 2*(1 + nuX2)*q[0]) - 2*nuX*q[2] + 4*nuX*(nuZ2)*q[2] + nuZ3*(S0 - 2*(q[0] + q[3]))) - 2*nuZ*(nuX*q[1] + nuZ*q[4]) + nuY2*(4*nuXZ*q[1] + 2*(-1 + 2*(nuZ2))*q[4]))) + nuX2*((nuX3*(S0 + 2*q[0]) - 4*(nuY*q[1] + nuZ*q[2]) + 4*(nuX2)*(nuY*q[1] + nuZ*q[2]) + nuX*((-2 + nuY2 + nuZ2)*S0 - 2*((2 + nuZ2)*q[0] - nuY2*q[3] + nuZ2*q[3] - 2*nuYZ*q[4])))*(nuX3*(S0 + 2*q[0]) - 4*(nuY*q[1] + nuZ*q[2]) + 4*(nuX2)*(nuY*q[1] + nuZ*q[2]) + nuX*((-2 + nuY2 + nuZ2)*S0 - 2*((2 + nuZ2)*q[0] - nuY2*q[3] + nuZ2*q[3] - 2*nuYZ*q[4])))));


            scalar energy2 = (3.*(S0*S0) - 4.*(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[0]*q[3] + q[3]*q[3] + q[4]*q[4]));
            energy2 = 0.25*W2*energy2*energy2;
            energy = energy1+energy2;
            break;
            }
            #ifndef __NVCC__
        default:
            UNWRITTENCODE("non-defined boundary type is attempting a force computation");
            #endif
        }
    return energy;
    };

#undef HOSTDEVICE
#endif
