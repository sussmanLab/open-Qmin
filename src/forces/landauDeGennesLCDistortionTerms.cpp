#include "landauDeGennesLC.h"
#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
#include "utilities.cuh"
#include "lcForces.h"
/*! \file landauDeGennesLCDistortionTerms.cpp */

/*
This file keeps separate the functions that actually compute the bulk and boundary 
terms coming from L1, L2,... L6
Keeps files and compilation more managable.

void landauDeGennesLC::computeForceTwoConstantCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    //start by precomputing first d_derivatives
    computeFirstDerivatives();
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int currentIndex;
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] <= 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            int ixd = neighbors[0]; int ixu = neighbors[1];
            int iyd = neighbors[2]; int iyu = neighbors[3];
            int izd = neighbors[4]; int izu = neighbors[5];
            xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
            yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
            zDown = Qtensors.data[izd]; zUp = Qtensors.data[izu];
            cubicLatticeDerivativeVector qCurrentDerivative = h_derivatives.data[currentIndex];
            cubicLatticeDerivativeVector xDownDerivative = h_derivatives.data[ixd];
            cubicLatticeDerivativeVector xUpDerivative = h_derivatives.data[ixu];
            cubicLatticeDerivativeVector yDownDerivative = h_derivatives.data[iyd];
            cubicLatticeDerivativeVector yUpDerivative = h_derivatives.data[iyu];
            cubicLatticeDerivativeVector zDownDerivative = h_derivatives.data[izd];
            cubicLatticeDerivativeVector zUpDerivative = h_derivatives.data[izu];

            //compute the phase terms depending only on the current site
            h_f.data[currentIndex] -= a*derivativeTrQ2(qCurrent);
            h_f.data[currentIndex] -= b*derivativeTrQ3(qCurrent);
            h_f.data[currentIndex] -= c*derivativeTrQ2Squared(qCurrent);

            dVec xMinusTerm(0.0);
            dVec xPlusTerm(0.0);
            dVec yMinusTerm(0.0);
            dVec yPlusTerm(0.0);
            dVec zMinusTerm(0.0);
            dVec zPlusTerm(0.0);
            if(latticeTypes.data[ixd] <= 0) //xMinus
                {
                xMinusTerm[0]=-(L1*(2*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) + qCurrentDerivative[12] - 2*xDown[0] - 2*xDown[3] - 18*q0*(qCurrentDerivative[11] - 2*(qCurrentDerivative[7] + xDown[4])) + xDownDerivative[12]))/2. - (L2*(2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] - 2*xDown[0] + xDownDerivative[6] + xDownDerivative[12]))/2.;

                xMinusTerm[1]=(L1*(qCurrentDerivative[5] - 2*(qCurrent[1] + 324*(q0*q0)*qCurrent[1] - xDown[1] + 9*q0*(qCurrentDerivative[9] + qCurrentDerivative[10] - qCurrentDerivative[13] + 2*xDown[2])) + xDownDerivative[5]))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xDown[1] + xDownDerivative[8] + xDownDerivative[14]))/2.;

                xMinusTerm[2]=-(L1*(1296*(q0*q0)*qCurrent[2] - 36*q0*(2*qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xDown[1]) - 2*(-2*qCurrent[2] + qCurrentDerivative[10] + 2*xDown[2] + xDownDerivative[10])))/4. + (L2*(-2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*xDown[2] - xDownDerivative[9] + xDownDerivative[10] + xDownDerivative[13]))/2.;

                xMinusTerm[3]=-(L1*(2*qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 4*qCurrent[3] + 648*(q0*q0)*qCurrent[3] - qCurrentDerivative[6] + 18*q0*qCurrentDerivative[7] + 18*q0*qCurrentDerivative[11] + qCurrentDerivative[12] - 2*xDown[0] - 4*xDown[3] + 72*q0*xDown[4] - xDownDerivative[6] + xDownDerivative[12]))/2.;

                xMinusTerm[4]=(L1*(-4*qCurrent[4] - 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[7] + qCurrentDerivative[11] + 18*q0*(qCurrentDerivative[6] - qCurrentDerivative[12] + 2*xDown[0] + 4*xDown[3]) + 4*xDown[4] + xDownDerivative[7] + xDownDerivative[11]))/2.;
                }
            if(latticeTypes.data[ixu] <= 0) //xPlus
                {
                xPlusTerm[0]=(L2*(-2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] + 2*xUp[0] + xUpDerivative[6] + xUpDerivative[12]))/2. - (L1*(648*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) - 36*q0*(-2*qCurrentDerivative[7] + qCurrentDerivative[11] + 2*xUp[4]) - 2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[12] + 2*xUp[0] + 2*xUp[3] + xUpDerivative[12])))/4.;

                xPlusTerm[1]=-(L1*(qCurrentDerivative[5] + 2*(qCurrent[1] + 324*(q0*q0)*qCurrent[1] - xUp[1] + 9*q0*(qCurrentDerivative[9] + qCurrentDerivative[10] - qCurrentDerivative[13] - 2*xUp[2])) + xUpDerivative[5]))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xUp[1] + xUpDerivative[8] + xUpDerivative[14]))/2.;

                xPlusTerm[2]=-(L1*(2*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[10] - 18*q0*(2*qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xUp[1]) - 2*xUp[2] + xUpDerivative[10]))/2. - (L2*(2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*xUp[2] - xUpDerivative[9] + xUpDerivative[10] + xUpDerivative[13]))/2.;

                xPlusTerm[3]=-(L1*(2*qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 4*qCurrent[3] + 648*(q0*q0)*qCurrent[3] + qCurrentDerivative[6] + 18*q0*qCurrentDerivative[7] + 18*q0*qCurrentDerivative[11] - qCurrentDerivative[12] - 2*xUp[0] - 4*xUp[3] - 72*q0*xUp[4] + xUpDerivative[6] - xUpDerivative[12]))/2.;

                xPlusTerm[4]=-(L1*(4*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[7] + qCurrentDerivative[11] - 18*q0*(qCurrentDerivative[6] - qCurrentDerivative[12] - 2*xUp[0] - 4*xUp[3]) - 4*xUp[4] + xUpDerivative[7] + xUpDerivative[11]))/2.;
                }

            if(latticeTypes.data[iyd] <= 0) //yMinus
                {
                yMinusTerm[0]=-(L1*(4*qCurrent[0] + 648*(q0*q0)*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*qCurrent[3] - qCurrentDerivative[1] - 18*q0*qCurrentDerivative[4] - 18*q0*qCurrentDerivative[11] + qCurrentDerivative[14] - 4*yDown[0] - 72*q0*yDown[2] - 2*yDown[3] - yDownDerivative[1] + yDownDerivative[14]))/2.;

                yMinusTerm[1]=-(L1*(1296*(q0*q0)*qCurrent[1] + 36*q0*(-qCurrentDerivative[2] + qCurrentDerivative[10] - qCurrentDerivative[13] - 2*yDown[4]) - 2*(-2*qCurrent[1] + qCurrentDerivative[3] + 2*yDown[1] + yDownDerivative[3])))/4. - (L2*(2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] - 2*yDown[1] + yDownDerivative[0] + yDownDerivative[12]))/2.;

                yMinusTerm[2]=(L1*(-4*qCurrent[2] - 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[4] + qCurrentDerivative[11] + 4*yDown[2] - 18*q0*(qCurrentDerivative[1] - qCurrentDerivative[14] + 4*yDown[0] + 2*yDown[3]) + yDownDerivative[4] + yDownDerivative[11]))/2.;

                yMinusTerm[3]=-(L1*(2*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + qCurrentDerivative[14] - 2*yDown[0] + 18*q0*(qCurrentDerivative[11] - 2*(qCurrentDerivative[4] + yDown[2])) - 2*yDown[3] + yDownDerivative[14]))/2. - (L2*(2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] - 2*yDown[3] + yDownDerivative[1] + yDownDerivative[14]));

                yMinusTerm[4]=(L2*(-2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*yDown[4] - yDownDerivative[2] + yDownDerivative[10] + yDownDerivative[13]))/2. - (L1*(1296*(q0*q0)*qCurrent[4] + 36*q0*(qCurrentDerivative[0] + qCurrentDerivative[12] + 2*(qCurrentDerivative[3] + yDown[1])) - 2*(-2*qCurrent[4] + qCurrentDerivative[13] + 2*yDown[4] + yDownDerivative[13])))/4.;
                }

            if(latticeTypes.data[iyu] <= 0) //yPlus
                {
                yPlusTerm[0]=-(L1*(4*qCurrent[0] + 648*(q0*q0)*qCurrent[0] + 2*qCurrent[3] + 324*(q0*q0)*qCurrent[3] + qCurrentDerivative[1] - 18*q0*qCurrentDerivative[4] - 18*q0*qCurrentDerivative[11] - qCurrentDerivative[14] - 4*yUp[0] + 72*q0*yUp[2] - 2*yUp[3] + yUpDerivative[1] - yUpDerivative[14]))/2.;

                yPlusTerm[1]=-(L1*(2*qCurrent[1] + 648*(q0*q0)*qCurrent[1] + qCurrentDerivative[3] - 2*yUp[1] + 18*q0*(-qCurrentDerivative[2] + qCurrentDerivative[10] - qCurrentDerivative[13] + 2*yUp[4]) + yUpDerivative[3]))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] + 2*yUp[1] + yUpDerivative[0] + yUpDerivative[12]))/2.;

                yPlusTerm[2]=-(L1*(4*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[4] + qCurrentDerivative[11] - 4*yUp[2] + 18*q0*(qCurrentDerivative[1] - qCurrentDerivative[14] - 4*yUp[0] - 2*yUp[3]) + yUpDerivative[4] + yUpDerivative[11]))/2.;

                yPlusTerm[3]=(L2*(-2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] + 2*yUp[3] + yUpDerivative[1] + yUpDerivative[14]))/2. - (L1*(648*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + 36*q0*(-2*qCurrentDerivative[4] + qCurrentDerivative[11] + 2*yUp[2]) - 2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[14] + 2*yUp[0] + 2*yUp[3] + yUpDerivative[14])))/4.;

                yPlusTerm[4]=-(L1*(2*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[13] + 18*q0*(qCurrentDerivative[0] + 2*qCurrentDerivative[3] + qCurrentDerivative[12] - 2*yUp[1]) - 2*yUp[4] + yUpDerivative[13]))/2. - (L2*(2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*yUp[4] - yUpDerivative[2] + yUpDerivative[10] + yUpDerivative[13]))/2.;
                }

            if(latticeTypes.data[izd] <= 0) //zMinus
                {
                zMinusTerm[0]=(L1*(qCurrentDerivative[2] - 2*(qCurrent[0] + 162*(q0*q0)*(2*qCurrent[0] + qCurrent[3]) - zDown[0] + 9*q0*(-qCurrentDerivative[4] + 2*qCurrentDerivative[7] + 2*zDown[1])) + zDownDerivative[2]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

                zMinusTerm[1]=-(L1*(1296*(q0*q0)*qCurrent[1] - 36*q0*(qCurrentDerivative[2] - qCurrentDerivative[9] + 2*zDown[0] - 2*zDown[3]) - 2*(-4*qCurrent[1] + qCurrentDerivative[4] + qCurrentDerivative[7] + 4*zDown[1] + zDownDerivative[4] + zDownDerivative[7])))/4.;

                zMinusTerm[2]=-(L1*(2*qCurrent[2] + 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[0] + 18*q0*qCurrentDerivative[1] + qCurrentDerivative[3] - 36*q0*qCurrentDerivative[5] - 18*q0*qCurrentDerivative[8] - 2*zDown[2] + 36*q0*zDown[4] + zDownDerivative[0] + zDownDerivative[3]))/2. - (L2*(2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] - 2*zDown[2] + zDownDerivative[0] + zDownDerivative[6]))/2.;

                zMinusTerm[3]=(L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2. - (L1*(648*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + 36*q0*(qCurrentDerivative[7] - 2*(qCurrentDerivative[4] + zDown[1])) - 2*(-2*qCurrent[3] + qCurrentDerivative[9] + 2*zDown[3] + zDownDerivative[9])))/4.;

                zMinusTerm[4]=-(L2*(2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] - 2*zDown[4] + zDownDerivative[1] + zDownDerivative[8]))/2. - (L1*(2*qCurrent[4] + 648*(q0*q0)*qCurrent[4] + qCurrentDerivative[5] + qCurrentDerivative[8] + 18*q0*(qCurrentDerivative[0] + 2*qCurrentDerivative[3] - qCurrentDerivative[6] - 2*zDown[2]) - 2*zDown[4] + zDownDerivative[5] + zDownDerivative[8]))/2.;
                }

            if(latticeTypes.data[izu] <= 0) //zPlus
                {
                zPlusTerm[0]=-(L1*(qCurrentDerivative[2] + 2*(qCurrent[0] + 324*(q0*q0)*qCurrent[0] + 162*(q0*q0)*qCurrent[3] - 9*q0*qCurrentDerivative[4] + 18*q0*qCurrentDerivative[7] - zUp[0] - 18*q0*zUp[1]) + zUpDerivative[2]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

                zPlusTerm[1]=-(L1*(4*qCurrent[1] + 648*(q0*q0)*qCurrent[1] + qCurrentDerivative[4] + qCurrentDerivative[7] - 4*zUp[1] - 18*q0*(qCurrentDerivative[2] - qCurrentDerivative[9] - 2*zUp[0] + 2*zUp[3]) + zUpDerivative[4] + zUpDerivative[7]))/2.;

                zPlusTerm[2]=(L1*(-2*qCurrent[2] - 648*(q0*q0)*qCurrent[2] + qCurrentDerivative[0] - 18*q0*qCurrentDerivative[1] + qCurrentDerivative[3] + 36*q0*qCurrentDerivative[5] + 18*q0*qCurrentDerivative[8] + 2*zUp[2] + 36*q0*zUp[4] + zUpDerivative[0] + zUpDerivative[3]))/2. + (L2*(-2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] + 2*zUp[2] + zUpDerivative[0] + zUpDerivative[6]))/2.;

                zPlusTerm[3]=-(L1*(2*qCurrent[3] + 324*(q0*q0)*(qCurrent[0] + 2*qCurrent[3]) + qCurrentDerivative[9] + 18*q0*(-2*qCurrentDerivative[4] + qCurrentDerivative[7] + 2*zUp[1]) - 2*zUp[3] + zUpDerivative[9]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

                zPlusTerm[4]=(L2*(-2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[1] + zUpDerivative[8]))/2. - (L1*(1296*(q0*q0)*qCurrent[4] + 36*q0*(qCurrentDerivative[0] - qCurrentDerivative[6] + 2*(qCurrentDerivative[3] + zUp[2])) - 2*(-2*qCurrent[4] + qCurrentDerivative[5] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[5] + zUpDerivative[8])))/4.;
                }

            h_f.data[currentIndex] += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;
            };
        };
    };

void landauDeGennesLC::computeForceThreeConstantCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    //start by precomputing first d_derivatives
    computeFirstDerivatives();
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int currentIndex;
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] <= 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            int ixd = neighbors[0]; int ixu = neighbors[1];
            int iyd = neighbors[2]; int iyu = neighbors[3];
            int izd = neighbors[4]; int izu = neighbors[5];
            xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
            yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
            zDown = Qtensors.data[izd]; zUp = Qtensors.data[izu];
            cubicLatticeDerivativeVector qCurrentDerivative = h_derivatives.data[currentIndex];
            cubicLatticeDerivativeVector xDownDerivative = h_derivatives.data[ixd];
            cubicLatticeDerivativeVector xUpDerivative = h_derivatives.data[ixu];
            cubicLatticeDerivativeVector yDownDerivative = h_derivatives.data[iyd];
            cubicLatticeDerivativeVector yUpDerivative = h_derivatives.data[iyu];
            cubicLatticeDerivativeVector zDownDerivative = h_derivatives.data[izd];
            cubicLatticeDerivativeVector zUpDerivative = h_derivatives.data[izu];

            //compute the phase terms depending only on the current site
            h_f.data[currentIndex] -= a*derivativeTrQ2(qCurrent);
            h_f.data[currentIndex] -= b*derivativeTrQ3(qCurrent);
            h_f.data[currentIndex] -= c*derivativeTrQ2Squared(qCurrent);

            dVec xMinusTerm(0.0);
            dVec xPlusTerm(0.0);
            dVec yMinusTerm(0.0);
            dVec yPlusTerm(0.0);
            dVec zMinusTerm(0.0);
            dVec zPlusTerm(0.0);
            if(latticeTypes.data[ixd] <= 0) //xMinus
                {
                xMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*xDown[0] - 4*xDown[3]))/4. - (L2*(2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] - 2*xDown[0] + xDownDerivative[6] + xDownDerivative[12]))/2. - (L3*(3*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + qCurrent[3]*qCurrent[3] + qCurrent[4]*qCurrent[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xDown[0]*xDown[0] + xDown[1]*xDown[1] + xDown[2]*xDown[2] + xDown[3]*xDown[3] + xDown[4]*xDown[4] + 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[5] + qCurrent[1]*qCurrentDerivative[8] + qCurrent[2]*qCurrentDerivative[13] - qCurrentDerivative[10]*(-2*qCurrent[2] + qCurrentDerivative[13]) - 2*qCurrent[0]*xDown[0] - 2*qCurrent[1]*xDown[1] - 2*qCurrent[2]*xDown[2] - 2*qCurrent[0]*xDown[3] - 2*qCurrent[3]*xDown[3] - 2*qCurrent[4]*xDown[4] + 2*xDown[1]*xDownDerivative[5] + xDown[1]*xDownDerivative[8] + 2*xDown[2]*xDownDerivative[10] + xDown[2]*xDownDerivative[13]))/2.;

                xMinusTerm[1]=-2*L1*(qCurrent[1] - xDown[1]) - (L3*(qCurrentDerivative[8]*(qCurrent[0] + 2*qCurrent[3] - xDown[0] - 2*xDown[3]) + qCurrentDerivative[5]*(2*qCurrent[0] + qCurrent[3] - 2*xDown[0] - xDown[3]) + 2*(qCurrent[0]*qCurrent[1] + qCurrent[2]*qCurrentDerivative[7] + qCurrent[4]*qCurrentDerivative[9] + qCurrent[2]*qCurrentDerivative[11] + qCurrent[1]*xDown[0] - qCurrent[0]*xDown[1] - xDown[0]*xDown[1] - qCurrentDerivative[6]*(-2*qCurrent[1] + xDown[1]) - qCurrentDerivative[7]*xDown[2] - qCurrentDerivative[9]*xDown[4] + xDown[1]*xDownDerivative[6] + xDown[2]*xDownDerivative[11])))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] - 2*xDown[1] + xDownDerivative[8] + xDownDerivative[14]))/2.;

                xMinusTerm[2]=-2*L1*(qCurrent[2] - xDown[2]) - (L3*(qCurrentDerivative[13]*(qCurrent[0] + 2*qCurrent[3] - xDown[0] - 2*xDown[3]) + qCurrentDerivative[10]*(2*qCurrent[0] + qCurrent[3] - 2*xDown[0] - xDown[3]) + 2*(qCurrent[0]*qCurrent[2] + qCurrent[1]*qCurrentDerivative[7] + 2*qCurrent[2]*qCurrentDerivative[12] + qCurrent[4]*qCurrentDerivative[14] + qCurrent[2]*xDown[0] + qCurrentDerivative[11]*(qCurrent[1] - xDown[1]) - qCurrent[0]*xDown[2] - qCurrentDerivative[12]*xDown[2] - xDown[0]*xDown[2] - qCurrentDerivative[14]*xDown[4] + xDown[1]*xDownDerivative[7] + xDown[2]*xDownDerivative[12])))/2. + (L2*(-2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*xDown[2] - xDownDerivative[9] + xDownDerivative[10] + xDownDerivative[13]))/2.;

                xMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*xDown[0] - 8*xDown[3]))/4. - (L3*(qCurrent[0]*qCurrent[0] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xDown[0]*xDown[0] + 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[8] + qCurrentDerivative[5]*(qCurrent[1] + qCurrentDerivative[8]) + qCurrentDerivative[10]*(qCurrent[2] - qCurrentDerivative[13]) + 2*qCurrent[2]*qCurrentDerivative[13] + 2*qCurrent[3]*xDown[0] - 2*qCurrent[0]*xDown[3] - 2*xDown[0]*xDown[3] + xDown[1]*xDownDerivative[5] + 2*xDown[1]*xDownDerivative[8] + xDown[2]*xDownDerivative[10] + 2*xDown[2]*xDownDerivative[13]))/2.;

                xMinusTerm[4]=-2*L1*(qCurrent[4] - xDown[4]) - (L3*(2*qCurrent[1]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*qCurrentDerivative[10] + qCurrentDerivative[8]*qCurrentDerivative[10] + 2*qCurrentDerivative[6]*qCurrentDerivative[11] + 2*qCurrentDerivative[7]*qCurrentDerivative[12] + qCurrentDerivative[5]*qCurrentDerivative[13] + 2*qCurrentDerivative[8]*qCurrentDerivative[13] + 2*qCurrent[2]*qCurrentDerivative[14] + 2*qCurrentDerivative[9]*qCurrentDerivative[14] + 2*qCurrent[0]*(qCurrent[4] - xDown[4]) + 2*xDown[0]*(qCurrent[4] - xDown[4]) + 2*xDown[1]*xDownDerivative[9] + 2*xDown[2]*xDownDerivative[14]))/2.;
                }
            if(latticeTypes.data[ixu] <= 0) //xPlus
                {
                xPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*xUp[0] - xUp[3])) + (L2*(-2*qCurrent[0] + qCurrentDerivative[6] + qCurrentDerivative[12] + 2*xUp[0] + xUpDerivative[6] + xUpDerivative[12]))/2. + (L3*(-3*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - qCurrent[3]*qCurrent[3] - qCurrent[4]*qCurrent[4] + qCurrentDerivative[10]*qCurrentDerivative[10] + qCurrentDerivative[11]*qCurrentDerivative[11] + qCurrentDerivative[12]*qCurrentDerivative[12] + qCurrentDerivative[13]*qCurrentDerivative[13] + qCurrentDerivative[14]*qCurrentDerivative[14] + xUp[0]*xUp[0] - xUp[1]*xUp[1] - xUp[2]*xUp[2] - xUp[3]*xUp[3] - xUp[4]*xUp[4] - 2*qCurrent[0]*qCurrent[3] + 2*qCurrent[1]*qCurrentDerivative[5] + qCurrent[1]*qCurrentDerivative[8] + qCurrent[2]*qCurrentDerivative[13] + qCurrentDerivative[10]*(2*qCurrent[2] + qCurrentDerivative[13]) + 2*qCurrent[0]*xUp[0] + 2*qCurrent[1]*xUp[1] + 2*qCurrent[2]*xUp[2] + 2*qCurrent[0]*xUp[3] + 2*qCurrent[3]*xUp[3] + 2*qCurrent[4]*xUp[4] + 2*xUp[1]*xUpDerivative[5] + xUp[1]*xUpDerivative[8] + 2*xUp[2]*xUpDerivative[10] + xUp[2]*xUpDerivative[13]))/2.;

                xPlusTerm[1]=-2*L1*(qCurrent[1] - xUp[1]) - (L3*(qCurrentDerivative[5]*(-2*qCurrent[0] - qCurrent[3] + 2*xUp[0] + xUp[3]) + qCurrentDerivative[8]*(-qCurrent[0] - 2*qCurrent[3] + xUp[0] + 2*xUp[3]) - 2*(qCurrent[2]*qCurrentDerivative[7] + qCurrent[4]*qCurrentDerivative[9] + qCurrent[2]*qCurrentDerivative[11] - qCurrent[1]*xUp[0] + qCurrentDerivative[6]*(2*qCurrent[1] - xUp[1]) + xUp[0]*xUp[1] + qCurrent[0]*(-qCurrent[1] + xUp[1]) - qCurrentDerivative[7]*xUp[2] - qCurrentDerivative[9]*xUp[4] + xUp[1]*xUpDerivative[6] + xUp[2]*xUpDerivative[11])))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[8] + qCurrentDerivative[14] + 2*xUp[1] + xUpDerivative[8] + xUpDerivative[14]))/2.;

                xPlusTerm[2]=-2*L1*(qCurrent[2] - xUp[2]) - (L3*(qCurrentDerivative[10]*(-2*qCurrent[0] - qCurrent[3] + 2*xUp[0] + xUp[3]) + qCurrentDerivative[13]*(-qCurrent[0] - 2*qCurrent[3] + xUp[0] + 2*xUp[3]) - 2*(-(qCurrent[0]*qCurrent[2]) + qCurrent[1]*qCurrentDerivative[7] + 2*qCurrent[2]*qCurrentDerivative[12] + qCurrent[4]*qCurrentDerivative[14] - qCurrent[2]*xUp[0] + qCurrentDerivative[11]*(qCurrent[1] - xUp[1]) + qCurrent[0]*xUp[2] - qCurrentDerivative[12]*xUp[2] + xUp[0]*xUp[2] - qCurrentDerivative[14]*xUp[4] + xUp[1]*xUpDerivative[7] + xUp[2]*xUpDerivative[12])))/2. - (L2*(2*qCurrent[2] - qCurrentDerivative[9] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*xUp[2] - xUpDerivative[9] + xUpDerivative[10] + xUpDerivative[13]))/2.;

                xPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - xUp[0] - 2*xUp[3])) - (L3*(qCurrent[0]*qCurrent[0] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - xUp[0]*xUp[0] + 2*qCurrent[0]*qCurrent[3] - 2*qCurrent[1]*qCurrentDerivative[8] + qCurrentDerivative[5]*(-qCurrent[1] + qCurrentDerivative[8]) - 2*qCurrent[2]*qCurrentDerivative[13] - qCurrentDerivative[10]*(qCurrent[2] + qCurrentDerivative[13]) + 2*qCurrent[3]*xUp[0] - 2*qCurrent[0]*xUp[3] - 2*xUp[0]*xUp[3] - xUp[1]*xUpDerivative[5] - 2*xUp[1]*xUpDerivative[8] - xUp[2]*xUpDerivative[10] - 2*xUp[2]*xUpDerivative[13]))/2.;

                xPlusTerm[4]=-2*L1*(qCurrent[4] - xUp[4]) - (L3*(-2*qCurrent[1]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*qCurrentDerivative[10] + qCurrentDerivative[8]*qCurrentDerivative[10] + 2*qCurrentDerivative[6]*qCurrentDerivative[11] + 2*qCurrentDerivative[7]*qCurrentDerivative[12] + qCurrentDerivative[5]*qCurrentDerivative[13] + 2*qCurrentDerivative[8]*qCurrentDerivative[13] - 2*qCurrent[2]*qCurrentDerivative[14] + 2*qCurrentDerivative[9]*qCurrentDerivative[14] + 2*qCurrent[0]*(qCurrent[4] - xUp[4]) + 2*xUp[0]*(qCurrent[4] - xUp[4]) - 2*xUp[1]*xUpDerivative[9] - 2*xUp[2]*xUpDerivative[14]))/2.;
                }

            if(latticeTypes.data[iyd] <= 0) //yMinus
                {
                yMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*yDown[0] - 4*yDown[3]))/4. - (L3*(qCurrent[3]*qCurrent[3] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - yDown[3]*yDown[3] + 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[0]*(2*qCurrent[1] + qCurrentDerivative[3]) + qCurrent[4]*qCurrentDerivative[13] - qCurrentDerivative[10]*(-2*qCurrent[4] + qCurrentDerivative[13]) - 2*qCurrent[3]*yDown[0] + 2*qCurrent[0]*yDown[3] - 2*yDown[0]*yDown[3] + 2*yDown[1]*yDownDerivative[0] + yDown[1]*yDownDerivative[3] + 2*yDown[4]*yDownDerivative[10] + yDown[4]*yDownDerivative[13]))/2.;

                yMinusTerm[1]=-2*L1*(qCurrent[1] - yDown[1]) - (L3*(qCurrentDerivative[3]*(qCurrent[0] + 2*qCurrent[3] - yDown[0] - 2*yDown[3]) + qCurrentDerivative[0]*(2*qCurrent[0] + qCurrent[3] - 2*yDown[0] - yDown[3]) + 2*(qCurrent[1]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[2] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[11] - qCurrent[3]*yDown[1] - qCurrentDerivative[1]*(-2*qCurrent[1] + yDown[1]) - qCurrentDerivative[2]*yDown[2] + qCurrent[1]*yDown[3] - yDown[1]*yDown[3] - qCurrentDerivative[4]*yDown[4] + yDown[1]*yDownDerivative[1] + yDown[4]*yDownDerivative[11])))/2. - (L2*(2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] - 2*yDown[1] + yDownDerivative[0] + yDownDerivative[12]))/2.;

                yMinusTerm[2]=-2*L1*(qCurrent[2] - yDown[2]) - (L3*(2*qCurrent[1]*qCurrentDerivative[2] + 2*qCurrentDerivative[0]*qCurrentDerivative[10] + qCurrentDerivative[3]*qCurrentDerivative[10] + 2*qCurrentDerivative[1]*qCurrentDerivative[11] + 2*qCurrent[4]*qCurrentDerivative[12] + 2*qCurrentDerivative[2]*qCurrentDerivative[12] + qCurrentDerivative[0]*qCurrentDerivative[13] + 2*qCurrentDerivative[3]*qCurrentDerivative[13] + 2*qCurrentDerivative[4]*qCurrentDerivative[14] + 2*qCurrent[3]*(qCurrent[2] - yDown[2]) + 2*(qCurrent[2] - yDown[2])*yDown[3] + 2*yDown[1]*yDownDerivative[2] + 2*yDown[4]*yDownDerivative[12]))/2.;

                yMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*yDown[0] - 8*yDown[3]))/4. - (L3*(qCurrent[0]*qCurrent[0] + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 3*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] + yDown[0]*yDown[0] + yDown[1]*yDown[1] + yDown[2]*yDown[2] - yDown[3]*yDown[3] + yDown[4]*yDown[4] + 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[0] + 2*qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[10]*(qCurrent[4] - qCurrentDerivative[13]) + 2*qCurrent[4]*qCurrentDerivative[13] - 2*qCurrent[0]*yDown[0] - 2*qCurrent[3]*yDown[0] - 2*qCurrent[1]*yDown[1] - 2*qCurrent[2]*yDown[2] - 2*qCurrent[3]*yDown[3] - 2*qCurrent[4]*yDown[4] + yDown[1]*yDownDerivative[0] + 2*yDown[1]*yDownDerivative[3] + yDown[4]*yDownDerivative[10] + 2*yDown[4]*yDownDerivative[13]))/2. - (L2*(2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] - 2*yDown[3] + yDownDerivative[1] + yDownDerivative[14]))/2.;

                yMinusTerm[4]=-2*L1*(qCurrent[4] - yDown[4]) + (L2*(-2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] + 2*yDown[4] - yDownDerivative[2] + yDownDerivative[10] + yDownDerivative[13]))/2. - (L3*(qCurrentDerivative[13]*(qCurrent[0] + 2*qCurrent[3] - yDown[0] - 2*yDown[3]) + qCurrentDerivative[10]*(2*qCurrent[0] + qCurrent[3] - 2*yDown[0] - yDown[3]) + 2*(qCurrent[3]*qCurrent[4] + qCurrent[1]*qCurrentDerivative[4] + qCurrent[2]*qCurrentDerivative[12] + 2*qCurrent[4]*qCurrentDerivative[14] + qCurrentDerivative[11]*(qCurrent[1] - yDown[1]) - qCurrentDerivative[12]*yDown[2] + qCurrent[4]*yDown[3] - qCurrent[3]*yDown[4] - qCurrentDerivative[14]*yDown[4] - yDown[3]*yDown[4] + yDown[1]*yDownDerivative[4] + yDown[4]*yDownDerivative[14])))/2.;
                }

            if(latticeTypes.data[iyu] <= 0) //yPlus
                {
                yPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*yUp[0] - yUp[3])) - (L3*(qCurrent[3]*qCurrent[3] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] - qCurrentDerivative[10]*qCurrentDerivative[10] - qCurrentDerivative[11]*qCurrentDerivative[11] - qCurrentDerivative[12]*qCurrentDerivative[12] - qCurrentDerivative[13]*qCurrentDerivative[13] - qCurrentDerivative[14]*qCurrentDerivative[14] - yUp[3]*yUp[3] + 2*qCurrent[0]*qCurrent[3] - qCurrent[1]*qCurrentDerivative[3] + qCurrentDerivative[0]*(-2*qCurrent[1] + qCurrentDerivative[3]) - qCurrent[4]*qCurrentDerivative[13] - qCurrentDerivative[10]*(2*qCurrent[4] + qCurrentDerivative[13]) - 2*qCurrent[3]*yUp[0] + 2*qCurrent[0]*yUp[3] - 2*yUp[0]*yUp[3] - 2*yUp[1]*yUpDerivative[0] - yUp[1]*yUpDerivative[3] - 2*yUp[4]*yUpDerivative[10] - yUp[4]*yUpDerivative[13]))/2.;

                yPlusTerm[1]=-2*L1*(qCurrent[1] - yUp[1]) - (L3*(qCurrentDerivative[0]*(-2*qCurrent[0] - qCurrent[3] + 2*yUp[0] + yUp[3]) + qCurrentDerivative[3]*(-qCurrent[0] - 2*qCurrent[3] + yUp[0] + 2*yUp[3]) - 2*(-(qCurrent[1]*qCurrent[3]) + qCurrent[2]*qCurrentDerivative[2] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[11] + qCurrentDerivative[1]*(2*qCurrent[1] - yUp[1]) + qCurrent[3]*yUp[1] - qCurrentDerivative[2]*yUp[2] - qCurrent[1]*yUp[3] + yUp[1]*yUp[3] - qCurrentDerivative[4]*yUp[4] + yUp[1]*yUpDerivative[1] + yUp[4]*yUpDerivative[11])))/2. + (L2*(-2*qCurrent[1] + qCurrentDerivative[0] + qCurrentDerivative[12] + 2*yUp[1] + yUpDerivative[0] + yUpDerivative[12]))/2.;

                yPlusTerm[2]=-2*L1*(qCurrent[2] - yUp[2]) - (L3*(-2*qCurrent[1]*qCurrentDerivative[2] + 2*qCurrentDerivative[0]*qCurrentDerivative[10] + qCurrentDerivative[3]*qCurrentDerivative[10] + 2*qCurrentDerivative[1]*qCurrentDerivative[11] - 2*qCurrent[4]*qCurrentDerivative[12] + 2*qCurrentDerivative[2]*qCurrentDerivative[12] + qCurrentDerivative[0]*qCurrentDerivative[13] + 2*qCurrentDerivative[3]*qCurrentDerivative[13] + 2*qCurrentDerivative[4]*qCurrentDerivative[14] + 2*qCurrent[3]*(qCurrent[2] - yUp[2]) + 2*(qCurrent[2] - yUp[2])*yUp[3] - 2*yUp[1]*yUpDerivative[2] - 2*yUp[4]*yUpDerivative[12]))/2.;

                yPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - yUp[0] - 2*yUp[3])) + (L3*(-(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 3*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[10]*qCurrentDerivative[10] + qCurrentDerivative[11]*qCurrentDerivative[11] + qCurrentDerivative[12]*qCurrentDerivative[12] + qCurrentDerivative[13]*qCurrentDerivative[13] + qCurrentDerivative[14]*qCurrentDerivative[14] - yUp[0]*yUp[0] - yUp[1]*yUp[1] - yUp[2]*yUp[2] + yUp[3]*yUp[3] - yUp[4]*yUp[4] - 2*qCurrent[0]*qCurrent[3] + qCurrent[1]*qCurrentDerivative[0] + 2*qCurrent[1]*qCurrentDerivative[3] + 2*qCurrent[4]*qCurrentDerivative[13] + qCurrentDerivative[10]*(qCurrent[4] + qCurrentDerivative[13]) + 2*qCurrent[0]*yUp[0] + 2*qCurrent[3]*yUp[0] + 2*qCurrent[1]*yUp[1] + 2*qCurrent[2]*yUp[2] + 2*qCurrent[3]*yUp[3] + 2*qCurrent[4]*yUp[4] + yUp[1]*yUpDerivative[0] + 2*yUp[1]*yUpDerivative[3] + yUp[4]*yUpDerivative[10] + 2*yUp[4]*yUpDerivative[13]))/2. + (L2*(-2*qCurrent[3] + qCurrentDerivative[1] + qCurrentDerivative[14] + 2*yUp[3] + yUpDerivative[1] + yUpDerivative[14]))/2.;

                yPlusTerm[4]=-2*L1*(qCurrent[4] - yUp[4]) - (L2*(2*qCurrent[4] - qCurrentDerivative[2] + qCurrentDerivative[10] + qCurrentDerivative[13] - 2*yUp[4] - yUpDerivative[2] + yUpDerivative[10] + yUpDerivative[13]))/2. - (L3*(qCurrentDerivative[10]*(-2*qCurrent[0] - qCurrent[3] + 2*yUp[0] + yUp[3]) + qCurrentDerivative[13]*(-qCurrent[0] - 2*qCurrent[3] + yUp[0] + 2*yUp[3]) - 2*(-(qCurrent[3]*qCurrent[4]) + qCurrent[1]*qCurrentDerivative[4] + qCurrent[2]*qCurrentDerivative[12] + 2*qCurrent[4]*qCurrentDerivative[14] + qCurrentDerivative[11]*(qCurrent[1] - yUp[1]) - qCurrentDerivative[12]*yUp[2] - qCurrent[4]*yUp[3] + qCurrent[3]*yUp[4] - qCurrentDerivative[14]*yUp[4] + yUp[3]*yUp[4] + yUp[1]*yUpDerivative[4] + yUp[4]*yUpDerivative[14])))/2.;
                }

            if(latticeTypes.data[izd] <= 0) //zMinus
                {
                zMinusTerm[0]=-(L1*(8*qCurrent[0] + 4*qCurrent[3] - 8*zDown[0] - 4*zDown[3]))/4. - (L3*(-3*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 2*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[0]*qCurrentDerivative[0] + qCurrentDerivative[1]*qCurrentDerivative[1] + qCurrentDerivative[2]*qCurrentDerivative[2] + qCurrentDerivative[3]*qCurrentDerivative[3] + qCurrentDerivative[4]*qCurrentDerivative[4] + zDown[0]*zDown[0] - zDown[1]*zDown[1] - zDown[2]*zDown[2] - zDown[4]*zDown[4] - 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[3] + qCurrentDerivative[0]*(2*qCurrent[2] + qCurrentDerivative[3]) + 2*qCurrent[4]*qCurrentDerivative[5] + qCurrent[4]*qCurrentDerivative[8] + 2*qCurrent[0]*zDown[0] + 2*qCurrent[3]*zDown[0] + 2*qCurrent[1]*zDown[1] + 2*qCurrent[2]*zDown[2] + 2*qCurrent[3]*zDown[3] + 2*zDown[0]*zDown[3] + 2*qCurrent[4]*zDown[4] + 2*zDown[2]*zDownDerivative[0] + zDown[2]*zDownDerivative[3] + 2*zDown[4]*zDownDerivative[5] + zDown[4]*zDownDerivative[8]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

                zMinusTerm[1]=-2*L1*(qCurrent[1] - zDown[1]) - (L3*(2*qCurrent[2]*qCurrentDerivative[1] + 2*qCurrentDerivative[0]*qCurrentDerivative[5] + qCurrentDerivative[3]*qCurrentDerivative[5] + 2*qCurrent[4]*qCurrentDerivative[6] + 2*qCurrentDerivative[1]*qCurrentDerivative[6] + 2*qCurrentDerivative[2]*qCurrentDerivative[7] + qCurrentDerivative[0]*qCurrentDerivative[8] + 2*qCurrentDerivative[3]*qCurrentDerivative[8] + 2*qCurrentDerivative[4]*qCurrentDerivative[9] + 2*qCurrent[0]*(-qCurrent[1] + zDown[1]) + 2*qCurrent[3]*(-qCurrent[1] + zDown[1]) + 2*zDown[0]*(-qCurrent[1] + zDown[1]) + 2*(-qCurrent[1] + zDown[1])*zDown[3] + 2*zDown[2]*zDownDerivative[1] + 2*zDown[4]*zDownDerivative[6]))/2.;

                zMinusTerm[2]=-2*L1*(qCurrent[2] - zDown[2]) - (L2*(2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] - 2*zDown[2] + zDownDerivative[0] + zDownDerivative[6]))/2. - (L3*(2*qCurrent[2]*qCurrentDerivative[2] + 2*qCurrent[4]*qCurrentDerivative[7] + 2*qCurrentDerivative[0]*(qCurrent[0] - zDown[0]) + qCurrentDerivative[3]*(qCurrent[0] - zDown[0]) + 2*qCurrentDerivative[1]*(qCurrent[1] - zDown[1]) + 2*qCurrentDerivative[2]*(qCurrent[2] - zDown[2]) + 2*qCurrent[0]*(-qCurrent[2] + zDown[2]) + 2*qCurrent[3]*(-qCurrent[2] + zDown[2]) + 2*zDown[0]*(-qCurrent[2] + zDown[2]) + qCurrentDerivative[0]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[3]*(qCurrent[3] - zDown[3]) + 2*(-qCurrent[2] + zDown[2])*zDown[3] + 2*qCurrentDerivative[4]*(qCurrent[4] - zDown[4]) + 2*zDown[2]*zDownDerivative[2] + 2*zDown[4]*zDownDerivative[7]))/2.;

                zMinusTerm[3]=-(L1*(4*qCurrent[0] + 8*qCurrent[3] - 4*zDown[0] - 8*zDown[3]))/4. - (L3*(-2*(qCurrent[0]*qCurrent[0]) - qCurrent[1]*qCurrent[1] - qCurrent[2]*qCurrent[2] - 3*(qCurrent[3]*qCurrent[3]) - qCurrent[4]*qCurrent[4] + qCurrentDerivative[5]*qCurrentDerivative[5] + qCurrentDerivative[6]*qCurrentDerivative[6] + qCurrentDerivative[7]*qCurrentDerivative[7] + qCurrentDerivative[8]*qCurrentDerivative[8] + qCurrentDerivative[9]*qCurrentDerivative[9] - zDown[1]*zDown[1] - zDown[2]*zDown[2] + zDown[3]*zDown[3] - zDown[4]*zDown[4] - 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[0] + 2*qCurrent[2]*qCurrentDerivative[3] + 2*qCurrent[4]*qCurrentDerivative[8] + qCurrentDerivative[5]*(qCurrent[4] + qCurrentDerivative[8]) + 2*qCurrent[0]*zDown[0] + 2*qCurrent[1]*zDown[1] + 2*qCurrent[2]*zDown[2] + 2*qCurrent[0]*zDown[3] + 2*qCurrent[3]*zDown[3] + 2*zDown[0]*zDown[3] + 2*qCurrent[4]*zDown[4] + zDown[2]*zDownDerivative[0] + 2*zDown[2]*zDownDerivative[3] + zDown[4]*zDownDerivative[5] + 2*zDown[4]*zDownDerivative[8]))/2. + (L2*(-2*qCurrent[0] - 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDown[0] + 2*zDown[3] + zDownDerivative[2] + zDownDerivative[9]))/2.;

                zMinusTerm[4]=-2*L1*(qCurrent[4] - zDown[4]) - (L2*(2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] - 2*zDown[4] + zDownDerivative[1] + zDownDerivative[8]))/2. - (L3*(2*qCurrent[2]*qCurrentDerivative[4] + 2*qCurrent[4]*qCurrentDerivative[9] + 2*qCurrentDerivative[5]*(qCurrent[0] - zDown[0]) + qCurrentDerivative[8]*(qCurrent[0] - zDown[0]) + 2*qCurrentDerivative[6]*(qCurrent[1] - zDown[1]) + 2*qCurrentDerivative[7]*(qCurrent[2] - zDown[2]) + qCurrentDerivative[5]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[8]*(qCurrent[3] - zDown[3]) + 2*qCurrentDerivative[9]*(qCurrent[4] - zDown[4]) + 2*qCurrent[0]*(-qCurrent[4] + zDown[4]) + 2*qCurrent[3]*(-qCurrent[4] + zDown[4]) + 2*zDown[0]*(-qCurrent[4] + zDown[4]) + 2*zDown[3]*(-qCurrent[4] + zDown[4]) + 2*zDown[2]*zDownDerivative[4] + 2*zDown[4]*zDownDerivative[9]))/2.;
                }

            if(latticeTypes.data[izu] <= 0) //zPlus
                {
                zPlusTerm[0]=-(L1*(2*qCurrent[0] + qCurrent[3] - 2*zUp[0] - zUp[3])) + (L3*(3*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 2*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[0]*qCurrentDerivative[0] - qCurrentDerivative[1]*qCurrentDerivative[1] - qCurrentDerivative[2]*qCurrentDerivative[2] - qCurrentDerivative[3]*qCurrentDerivative[3] - qCurrentDerivative[4]*qCurrentDerivative[4] - zUp[0]*zUp[0] + zUp[1]*zUp[1] + zUp[2]*zUp[2] + zUp[4]*zUp[4] + 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[3] - qCurrentDerivative[0]*(-2*qCurrent[2] + qCurrentDerivative[3]) + 2*qCurrent[4]*qCurrentDerivative[5] + qCurrent[4]*qCurrentDerivative[8] - 2*qCurrent[0]*zUp[0] - 2*qCurrent[3]*zUp[0] - 2*qCurrent[1]*zUp[1] - 2*qCurrent[2]*zUp[2] - 2*qCurrent[3]*zUp[3] - 2*zUp[0]*zUp[3] - 2*qCurrent[4]*zUp[4] + 2*zUp[2]*zUpDerivative[0] + zUp[2]*zUpDerivative[3] + 2*zUp[4]*zUpDerivative[5] + zUp[4]*zUpDerivative[8]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

                zPlusTerm[1]=-2*L1*(qCurrent[1] - zUp[1]) - (L3*(-2*qCurrent[2]*qCurrentDerivative[1] + 2*qCurrentDerivative[0]*qCurrentDerivative[5] + qCurrentDerivative[3]*qCurrentDerivative[5] - 2*qCurrent[4]*qCurrentDerivative[6] + 2*qCurrentDerivative[1]*qCurrentDerivative[6] + 2*qCurrentDerivative[2]*qCurrentDerivative[7] + qCurrentDerivative[0]*qCurrentDerivative[8] + 2*qCurrentDerivative[3]*qCurrentDerivative[8] + 2*qCurrentDerivative[4]*qCurrentDerivative[9] + 2*qCurrent[0]*(-qCurrent[1] + zUp[1]) + 2*qCurrent[3]*(-qCurrent[1] + zUp[1]) + 2*zUp[0]*(-qCurrent[1] + zUp[1]) + 2*(-qCurrent[1] + zUp[1])*zUp[3] - 2*zUp[2]*zUpDerivative[1] - 2*zUp[4]*zUpDerivative[6]))/2.;

                zPlusTerm[2]=-2*L1*(qCurrent[2] - zUp[2]) + (L2*(-2*qCurrent[2] + qCurrentDerivative[0] + qCurrentDerivative[6] + 2*zUp[2] + zUpDerivative[0] + zUpDerivative[6]))/2. - (L3*(qCurrentDerivative[0]*(-2*qCurrent[0] - qCurrent[3] + 2*zUp[0] + zUp[3]) + qCurrentDerivative[3]*(-qCurrent[0] - 2*qCurrent[3] + zUp[0] + 2*zUp[3]) - 2*(qCurrent[0]*qCurrent[2] + qCurrent[2]*qCurrent[3] + qCurrent[4]*qCurrentDerivative[4] + qCurrent[4]*qCurrentDerivative[7] + qCurrent[2]*zUp[0] + qCurrentDerivative[1]*(qCurrent[1] - zUp[1]) + qCurrentDerivative[2]*(2*qCurrent[2] - zUp[2]) - qCurrent[0]*zUp[2] - qCurrent[3]*zUp[2] - zUp[0]*zUp[2] + qCurrent[2]*zUp[3] - zUp[2]*zUp[3] - qCurrentDerivative[4]*zUp[4] + zUp[2]*zUpDerivative[2] + zUp[4]*zUpDerivative[7])))/2.;

                zPlusTerm[3]=-(L1*(qCurrent[0] + 2*qCurrent[3] - zUp[0] - 2*zUp[3])) + (L3*(2*(qCurrent[0]*qCurrent[0]) + qCurrent[1]*qCurrent[1] + qCurrent[2]*qCurrent[2] + 3*(qCurrent[3]*qCurrent[3]) + qCurrent[4]*qCurrent[4] - qCurrentDerivative[5]*qCurrentDerivative[5] - qCurrentDerivative[6]*qCurrentDerivative[6] - qCurrentDerivative[7]*qCurrentDerivative[7] - qCurrentDerivative[8]*qCurrentDerivative[8] - qCurrentDerivative[9]*qCurrentDerivative[9] + zUp[1]*zUp[1] + zUp[2]*zUp[2] - zUp[3]*zUp[3] + zUp[4]*zUp[4] + 4*qCurrent[0]*qCurrent[3] + qCurrent[2]*qCurrentDerivative[0] + 2*qCurrent[2]*qCurrentDerivative[3] + qCurrentDerivative[5]*(qCurrent[4] - qCurrentDerivative[8]) + 2*qCurrent[4]*qCurrentDerivative[8] - 2*qCurrent[0]*zUp[0] - 2*qCurrent[1]*zUp[1] - 2*qCurrent[2]*zUp[2] - 2*qCurrent[0]*zUp[3] - 2*qCurrent[3]*zUp[3] - 2*zUp[0]*zUp[3] - 2*qCurrent[4]*zUp[4] + zUp[2]*zUpDerivative[0] + 2*zUp[2]*zUpDerivative[3] + zUp[4]*zUpDerivative[5] + 2*zUp[4]*zUpDerivative[8]))/2. - (L2*(2*qCurrent[0] + 2*qCurrent[3] + qCurrentDerivative[2] + qCurrentDerivative[9] - 2*zUp[0] - 2*zUp[3] + zUpDerivative[2] + zUpDerivative[9]))/2.;

                zPlusTerm[4]=-2*L1*(qCurrent[4] - zUp[4]) + (L2*(-2*qCurrent[4] + qCurrentDerivative[1] + qCurrentDerivative[8] + 2*zUp[4] + zUpDerivative[1] + zUpDerivative[8]))/2. - (L3*(qCurrentDerivative[5]*(-2*qCurrent[0] - qCurrent[3] + 2*zUp[0] + zUp[3]) + qCurrentDerivative[8]*(-qCurrent[0] - 2*qCurrent[3] + zUp[0] + 2*zUp[3]) - 2*(qCurrent[0]*qCurrent[4] + qCurrent[3]*qCurrent[4] + qCurrent[2]*qCurrentDerivative[4] + 2*qCurrent[4]*qCurrentDerivative[9] + qCurrent[4]*zUp[0] + qCurrentDerivative[6]*(qCurrent[1] - zUp[1]) + qCurrentDerivative[7]*(qCurrent[2] - zUp[2]) + qCurrent[4]*zUp[3] - qCurrent[0]*zUp[4] - qCurrent[3]*zUp[4] - qCurrentDerivative[9]*zUp[4] - zUp[0]*zUp[4] - zUp[3]*zUp[4] + zUp[2]*zUpDerivative[4] + zUp[4]*zUpDerivative[9])))/2.;
                }

            h_f.data[currentIndex] += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;
            };
        };
    };

void landauDeGennesLC::computeL24ForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce)
{
    //start by precomputing first d_derivatives if we're not in the 2 or 3 constant approx.
    if(numberOfConstants==distortionEnergyType::oneConstant)
        computeFirstDerivatives();
    energy=0.0;
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);

    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int currentIndex;
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] <= 0)
            {
            int ixd = neighbors[0]; int ixu = neighbors[1];
            int iyd = neighbors[2]; int iyu = neighbors[3];
            int izd = neighbors[4]; int izu = neighbors[5];
            cubicLatticeDerivativeVector qCurrentDerivative = h_derivatives.data[currentIndex];
            cubicLatticeDerivativeVector xDownDerivative = h_derivatives.data[ixd];
            cubicLatticeDerivativeVector xUpDerivative = h_derivatives.data[ixu];
            cubicLatticeDerivativeVector yDownDerivative = h_derivatives.data[iyd];
            cubicLatticeDerivativeVector yUpDerivative = h_derivatives.data[iyu];
            cubicLatticeDerivativeVector zDownDerivative = h_derivatives.data[izd];
            cubicLatticeDerivativeVector zUpDerivative = h_derivatives.data[izu];

            dVec xMinusTerm(0.0);
            dVec xPlusTerm(0.0);
            dVec yMinusTerm(0.0);
            dVec yPlusTerm(0.0);
            dVec zMinusTerm(0.0);
            dVec zPlusTerm(0.0);
            if(latticeTypes.data[ixd] <= 0) //xMinus
                {
                xMinusTerm[0]=(3*L24*(qCurrentDerivative[6] + xDownDerivative[6] + 2*(qCurrentDerivative[12] + xDownDerivative[12])))/2.;

                xMinusTerm[1]=(-3*L24*(qCurrentDerivative[5] - qCurrentDerivative[8] - qCurrentDerivative[14] + xDownDerivative[5] - xDownDerivative[8] - xDownDerivative[14]))/2.;

                xMinusTerm[2]=(-3*L24*(-qCurrentDerivative[9] + 2*qCurrentDerivative[10] + qCurrentDerivative[13] - xDownDerivative[9] + 2*xDownDerivative[10] + xDownDerivative[13]))/2.;

                xMinusTerm[3]=(-3*L24*(qCurrentDerivative[6] - qCurrentDerivative[12] + xDownDerivative[6] - xDownDerivative[12]))/2.;

                xMinusTerm[4]=(-3*L24*(qCurrentDerivative[7] + qCurrentDerivative[11] + xDownDerivative[7] + xDownDerivative[11]))/2.;
                }
            if(latticeTypes.data[ixu] <= 0) //xPlus
                {
                xPlusTerm[0]=(-3*L24*(qCurrentDerivative[6] + xUpDerivative[6] + 2*(qCurrentDerivative[12] + xUpDerivative[12])))/2.;

                xPlusTerm[1]=(-3*L24*(-qCurrentDerivative[5] + qCurrentDerivative[8] + qCurrentDerivative[14] - xUpDerivative[5] + xUpDerivative[8] + xUpDerivative[14]))/2.;

                xPlusTerm[2]=(3*L24*(-qCurrentDerivative[9] + 2*qCurrentDerivative[10] + qCurrentDerivative[13] - xUpDerivative[9] + 2*xUpDerivative[10] + xUpDerivative[13]))/2.;

                xPlusTerm[3]=(3*L24*(qCurrentDerivative[6] - qCurrentDerivative[12] + xUpDerivative[6] - xUpDerivative[12]))/2.;

                xPlusTerm[4]=(3*L24*(qCurrentDerivative[7] + qCurrentDerivative[11] + xUpDerivative[7] + xUpDerivative[11]))/2.;
                }

            if(latticeTypes.data[iyd] <= 0) //yMinus
                {
                yMinusTerm[0]=(-3*L24*(qCurrentDerivative[1] - qCurrentDerivative[14] + yDownDerivative[1] - yDownDerivative[14]))/2.;

                yMinusTerm[1]=(3*L24*(qCurrentDerivative[0] - qCurrentDerivative[3] + qCurrentDerivative[12] + yDownDerivative[0] - yDownDerivative[3] + yDownDerivative[12]))/2.;

                yMinusTerm[2]=(-3*L24*(qCurrentDerivative[4] + qCurrentDerivative[11] + yDownDerivative[4] + yDownDerivative[11]))/2.;

                yMinusTerm[3]=(3*L24*(qCurrentDerivative[1] + yDownDerivative[1] + 2*(qCurrentDerivative[14] + yDownDerivative[14])))/2.;

                yMinusTerm[4]=(-3*L24*(-qCurrentDerivative[2] + qCurrentDerivative[10] + 2*qCurrentDerivative[13] - yDownDerivative[2] + yDownDerivative[10] + 2*yDownDerivative[13]))/2.;
                }

            if(latticeTypes.data[iyu] <= 0) //yPlus
                {
                yPlusTerm[0]=(3*L24*(qCurrentDerivative[1] - qCurrentDerivative[14] + yUpDerivative[1] - yUpDerivative[14]))/2.;

                yPlusTerm[1]=(-3*L24*(qCurrentDerivative[0] - qCurrentDerivative[3] + qCurrentDerivative[12] + yUpDerivative[0] - yUpDerivative[3] + yUpDerivative[12]))/2.;

                yPlusTerm[2]=(3*L24*(qCurrentDerivative[4] + qCurrentDerivative[11] + yUpDerivative[4] + yUpDerivative[11]))/2.;

                yPlusTerm[3]=(-3*L24*(qCurrentDerivative[1] + yUpDerivative[1] + 2*(qCurrentDerivative[14] + yUpDerivative[14])))/2.;

                yPlusTerm[4]=(3*L24*(-qCurrentDerivative[2] + qCurrentDerivative[10] + 2*qCurrentDerivative[13] - yUpDerivative[2] + yUpDerivative[10] + 2*yUpDerivative[13]))/2.;
                }

            if(latticeTypes.data[izd] <= 0) //zMinus
                {
                zMinusTerm[0]=(-3*L24*(2*qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zDownDerivative[2] + zDownDerivative[9]))/2.;

                zMinusTerm[1]=(-3*L24*(qCurrentDerivative[4] + qCurrentDerivative[7] + zDownDerivative[4] + zDownDerivative[7]))/2.;

                zMinusTerm[2]=(3*L24*(2*qCurrentDerivative[0] + qCurrentDerivative[3] + qCurrentDerivative[6] + 2*zDownDerivative[0] + zDownDerivative[3] + zDownDerivative[6]))/2.;

                zMinusTerm[3]=(-3*L24*(qCurrentDerivative[2] + zDownDerivative[2] + 2*(qCurrentDerivative[9] + zDownDerivative[9])))/2.;

                zMinusTerm[4]=(3*L24*(qCurrentDerivative[1] + qCurrentDerivative[5] + 2*qCurrentDerivative[8] + zDownDerivative[1] + zDownDerivative[5] + 2*zDownDerivative[8]))/2.;
                }

            if(latticeTypes.data[izu] <= 0) //zPlus
                {
                zPlusTerm[0]=(3*L24*(2*qCurrentDerivative[2] + qCurrentDerivative[9] + 2*zUpDerivative[2] + zUpDerivative[9]))/2.;

                zPlusTerm[1]=(3*L24*(qCurrentDerivative[4] + qCurrentDerivative[7] + zUpDerivative[4] + zUpDerivative[7]))/2.;

                zPlusTerm[2]=(-3*L24*(2*qCurrentDerivative[0] + qCurrentDerivative[3] + qCurrentDerivative[6] + 2*zUpDerivative[0] + zUpDerivative[3] + zUpDerivative[6]))/2.;

                zPlusTerm[3]=(3*L24*(qCurrentDerivative[2] + zUpDerivative[2] + 2*(qCurrentDerivative[9] + zUpDerivative[9])))/2.;

                zPlusTerm[4]=(-3*L24*(qCurrentDerivative[1] + qCurrentDerivative[5] + 2*qCurrentDerivative[8] + zUpDerivative[1] + zUpDerivative[5] + 2*zUpDerivative[8]))/2.;
                }

            h_f.data[currentIndex] += xMinusTerm+xPlusTerm+yMinusTerm+yPlusTerm+zMinusTerm+zPlusTerm;
            };
        };
    };

void landauDeGennesLC::computeL24ForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<cubicLatticeDerivativeVector> d_derivatives(forceCalculationAssist,access_location::device,access_mode::read);
    l24ForceTuner->begin();
    gpu_qTensor_computeL24ForcesGPU(d_force.data,
                              d_latticeTypes.data,
                              d_derivatives.data,
                              lattice->latticeIndex,
                              N,L24,zeroOutForce,
                              boundaryForceTuner->getParameter());
    l24ForceTuner->end();
    };


 */

void landauDeGennesLC::computeL1BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);

    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        //currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        int currentIndex = i;
        dVec force(0.0);
        if(latticeTypes.data[currentIndex] == 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            //compute the phase terms depending only on the current site
            force -= a*derivativeTrQ2(qCurrent);
            force -= b*derivativeTrQ3(qCurrent);
            force -= c*derivativeTrQ2Squared(qCurrent);

            int ixd, ixu,iyd,iyu,izd,izu;
            ixd =latticeNeighbors.data[lattice->neighborIndex(0,currentIndex)];
            ixu =latticeNeighbors.data[lattice->neighborIndex(1,currentIndex)];
            iyd =latticeNeighbors.data[lattice->neighborIndex(2,currentIndex)];
            iyu =latticeNeighbors.data[lattice->neighborIndex(3,currentIndex)];
            izd =latticeNeighbors.data[lattice->neighborIndex(4,currentIndex)];
            izu =latticeNeighbors.data[lattice->neighborIndex(5,currentIndex)];
            xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
            yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
            zDown = Qtensors.data[izd]; zUp = Qtensors.data[izu];
            dVec spatialTerm(0.0);
            //use the neighbors to compute the distortion
            lcForce::bulkOneConstantForce(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
            force -= spatialTerm;
            };
        if(zeroOutForce)
            h_f.data[currentIndex] = force;
        else
            h_f.data[currentIndex] += force;
        };
    }
void landauDeGennesLC::computeL1BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);

    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        //currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        int currentIndex = i;
        dVec force(0.0);
        int siteType = latticeTypes.data[currentIndex];
        if(siteType < 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            //compute the phase terms depending only on the current site
            force -= a*derivativeTrQ2(qCurrent);
            force -= b*derivativeTrQ3(qCurrent);
            force -= c*derivativeTrQ2Squared(qCurrent);

            int ixd, ixu,iyd,iyu,izd,izu;
            ixd =latticeNeighbors.data[lattice->neighborIndex(0,currentIndex)];
            ixu =latticeNeighbors.data[lattice->neighborIndex(1,currentIndex)];
            iyd =latticeNeighbors.data[lattice->neighborIndex(2,currentIndex)];
            iyu =latticeNeighbors.data[lattice->neighborIndex(3,currentIndex)];
            izd =latticeNeighbors.data[lattice->neighborIndex(4,currentIndex)];
            izu =latticeNeighbors.data[lattice->neighborIndex(5,currentIndex)];
            xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
            yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
            zDown = Qtensors.data[izd]; zUp = Qtensors.data[izu];
            dVec spatialTerm(0.0);
            if(siteType == -2)
                    lcForce::bulkOneConstantForce(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
            else
                    lcForce::boundaryOneConstantForce(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                                    latticeTypes.data[ixd],latticeTypes.data[ixu],latticeTypes.data[iyd],
                            latticeTypes.data[iyu],latticeTypes.data[izd],latticeTypes.data[izu],
                            spatialTerm);
            force -= spatialTerm;
            };
        if(zeroOutForce)
            h_f.data[currentIndex] = force;
        else
            h_f.data[currentIndex] += force;
        };
    }

void landauDeGennesLC::computeL2BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }
void landauDeGennesLC::computeL2BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }

void landauDeGennesLC::computeL3BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }
void landauDeGennesLC::computeL3BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }

void landauDeGennesLC::computeL4BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }
void landauDeGennesLC::computeL4BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }

void landauDeGennesLC::computeL6BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }
void landauDeGennesLC::computeL6BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    }
