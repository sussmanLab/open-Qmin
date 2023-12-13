#include "landauDeGennesLC2D.h"
#include "qTensorFunctions2D.h"

void landauDeGennesLC2D::computeL1Bulk2DCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);

    scalar a = 0.5*A;
    scalar c = 0.25*C;
    dVec qCurrent, xDown, xUp, yDown,yUp,xDownyDown, xUpyDown, xDownyUp, xUpyUp;
    int ixd, ixu, iyd,iyu, ixdyd, ixdyu, ixuyd, ixuyu;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        int currentIndex = i;
        dVec force(0.0);
        dVec spatialTerm(0.0);
        if(latticeTypes.data[currentIndex] == 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            //compute the phase terms depending only on the current site
            force -= a*derivativeTr2DQ2(qCurrent);
            force -= c*derivativeTr2DQ2Squared(qCurrent);
            //nearest-neighbors
            ixd =latticeNeighbors.data[lattice->neighborIndex(0,currentIndex)];
            ixu =latticeNeighbors.data[lattice->neighborIndex(1,currentIndex)];
            iyd =latticeNeighbors.data[lattice->neighborIndex(2,currentIndex)];
            iyu =latticeNeighbors.data[lattice->neighborIndex(3,currentIndex)];
            xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
            yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
            //next-nearest neighbors
            ixdyd =latticeNeighbors.data[lattice->neighborIndex(4,currentIndex)];
            ixdyu =latticeNeighbors.data[lattice->neighborIndex(5,currentIndex)];
            ixuyd =latticeNeighbors.data[lattice->neighborIndex(6,currentIndex)];
            ixuyu =latticeNeighbors.data[lattice->neighborIndex(7,currentIndex)];
            xDownyDown = Qtensors.data[ixdyd]; xDownyUp= Qtensors.data[ixdyu];
            xUpyDown = Qtensors.data[ixuyd]; xUpyUp= Qtensors.data[ixuyu];

            spatialTerm = 1.0*laplacianStencil(L1,qCurrent,xDown,xUp,yDown,yUp,xDownyDown, xUpyDown, xDownyUp, xUpyUp);
            //spatialTerm = 1.0*laplacianStencil5(L1,qCurrent,xDown,xUp,yDown,yUp);
            //use the neighbors to compute the distortion
            force += spatialTerm;
            };
        if(zeroOutForce)
            h_f.data[currentIndex] = force;
        else
            h_f.data[currentIndex] += force;
        };
    }
