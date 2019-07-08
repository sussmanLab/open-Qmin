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
            lcForce::bulkL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
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
                    lcForce::bulkL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
            else
                    lcForce::boundaryL1Force(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
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

void landauDeGennesLC::computeOtherDistortionTermsBulkCPU(GPUArray<dVec> &forces, scalar L2, scalar L3, scalar L4, scalar L6)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);

    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        cubicLatticeDerivativeVector xDownDerivative, xUpDerivative,yDownDerivative,yUpDerivative,zDownDerivative,zUpDerivative;
        int currentIndex = i;
        dVec force(0.0);
        if(latticeTypes.data[currentIndex] == 0)
            {
            qCurrent = Qtensors.data[currentIndex];
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
            xDownDerivative = h_derivatives.data[ixd];
            xUpDerivative = h_derivatives.data[ixu];
            yDownDerivative = h_derivatives.data[iyd];
            yUpDerivative = h_derivatives.data[iyu];
            zDownDerivative = h_derivatives.data[izd];
            zUpDerivative = h_derivatives.data[izu];
            dVec spatialTerm(0.0);
            dVec individualTerms(0.0);

            if(L2 != 0)
                {
                lcForce::bulkL2Force(L2,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                    xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                    individualTerms);
                spatialTerm += individualTerms;
                }
            if(L3 != 0)
                {
                lcForce::bulkL3Force(L3,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                    xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                    individualTerms);
                spatialTerm += individualTerms;
                }
            if(L4 != 0)
                {
                lcForce::bulkL4Force(L4,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                    xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                    individualTerms);
                spatialTerm += individualTerms;
                }
            if(L6 != 0)
                {
                lcForce::bulkL6Force(L6,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                    xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                    individualTerms);
                spatialTerm += individualTerms;
                }
            force -= spatialTerm;
            };
        h_f.data[currentIndex] += force;
        };
    }

void landauDeGennesLC::computeOtherDistortionTermsBoundaryCPU(GPUArray<dVec> &forces,scalar L2, scalar L3, scalar L4, scalar L6)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);

    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        cubicLatticeDerivativeVector xDownDerivative, xUpDerivative,yDownDerivative,yUpDerivative,zDownDerivative,zUpDerivative;
        int currentIndex = i;
        dVec force(0.0);
        int siteType = latticeTypes.data[currentIndex];
        if(siteType < 0)
            {
            qCurrent = Qtensors.data[currentIndex];
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
            xDownDerivative = h_derivatives.data[ixd];
            xUpDerivative = h_derivatives.data[ixu];
            yDownDerivative = h_derivatives.data[iyd];
            yUpDerivative = h_derivatives.data[iyu];
            zDownDerivative = h_derivatives.data[izd];
            zUpDerivative = h_derivatives.data[izu];

            dVec spatialTerm(0.0);
            dVec individualTerms(0.0);

            if(siteType == -2)
                {
                if(L2 != 0)
                    {
                    lcForce::bulkL2Force(L2,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L3 != 0)
                    {
                    lcForce::bulkL3Force(L3,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L4 != 0)
                    {
                    lcForce::bulkL4Force(L4,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L6 != 0)
                    {
                    lcForce::bulkL6Force(L6,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                }
            else
                {
                int boundaryCase = lcForce::getBoundaryCase(latticeTypes.data[ixd],latticeTypes.data[ixu],
                                                   latticeTypes.data[iyd],latticeTypes.data[iyu],
                                                   latticeTypes.data[izd],latticeTypes.data[izu]);
                if(L2 != 0)
                    {
                    lcForce::boundaryL2Force(L2,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L3 != 0)
                    {
                    lcForce::boundaryL3Force(L3,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L4 != 0)
                    {
                    lcForce::boundaryL4Force(L4,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                if(L6 != 0)
                    {
                    lcForce::boundaryL6Force(L6,boundaryCase,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,
                        xDownDerivative, xUpDerivative,yDownDerivative, yUpDerivative,zDownDerivative, zUpDerivative,
                        individualTerms);
                    spatialTerm += individualTerms;
                    }
                };
            force -= spatialTerm;
            };
        h_f.data[currentIndex] += force;
        };
    }
