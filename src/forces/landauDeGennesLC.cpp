#include "landauDeGennesLC.h"
#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
/*! \file landauDeGennesLC.cpp */

landauDeGennesLC::landauDeGennesLC(double _A, double _B, double _C, double _L1) :
    A(_A), B(_B), C(_C), L1(_L1)
    {
    useNeighborList=false;
    numberOfConstants = distortionEnergyType::oneConstant;
    forceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    };

landauDeGennesLC::landauDeGennesLC(double _A, double _B, double _C, double _L1, double _L2) :
    A(_A), B(_B), C(_C), L1(_L1), L2(_L2)
    {
    useNeighborList=false;
    numberOfConstants = distortionEnergyType::twoConstant;
    forceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    };

landauDeGennesLC::landauDeGennesLC(double _A, double _B, double _C, double _L1, double _L2, double _L3) :
    A(_A), B(_B), C(_C), L1(_L1), L2(_L2), L3(_L3)
    {
    useNeighborList=false;
    numberOfConstants = distortionEnergyType::threeConstant;
    forceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    };

//!compute the phase and distortion parts of the Landau energy
void landauDeGennesLC::computeForceOneConstantCPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    energy=0.0;
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int currentIndex;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    scalar l = 2.0*L1;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] <= 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            xDown = Qtensors.data[neighbors[0]];
            xUp = Qtensors.data[neighbors[1]];
            yDown = Qtensors.data[neighbors[2]];
            yUp = Qtensors.data[neighbors[3]];
            zDown = Qtensors.data[neighbors[4]];
            zUp = Qtensors.data[neighbors[5]];

            //compute the phase terms depending only on the current site
            h_f.data[currentIndex] -= a*derivativeTrQ2(qCurrent);
            h_f.data[currentIndex] -= b*derivativeTrQ3(qCurrent);
            h_f.data[currentIndex] -= c*derivativeTrQ2Squared(qCurrent);

            //use the neighbors to compute the distortion
            if(latticeTypes.data[currentIndex] == 0) // if it's in the bulk, things are easy
                {
                dVec spatialTerm = l*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
                scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
                spatialTerm[0] += AxxAyy;
                spatialTerm[1] *= 2.0;
                spatialTerm[2] *= 2.0;
                spatialTerm[3] += AxxAyy;
                spatialTerm[4] *= 2.0;
                h_f.data[currentIndex] -= spatialTerm;
                }
            else
                {//distortion term first
                dVec spatialTerm(0.0);
                if(latticeTypes.data[neighbors[0]] <=0)
                    spatialTerm += qCurrent - xDown;
                if(latticeTypes.data[neighbors[1]] <=0)
                    spatialTerm += qCurrent - xUp;
                if(latticeTypes.data[neighbors[2]] <=0)
                    spatialTerm += qCurrent - yDown;
                if(latticeTypes.data[neighbors[3]] <=0)
                    spatialTerm += qCurrent - yUp;
                if(latticeTypes.data[neighbors[4]] <=0)
                    spatialTerm += qCurrent - zDown;
                if(latticeTypes.data[neighbors[5]] <=0)
                    spatialTerm += qCurrent - zUp;
                scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
                spatialTerm[0] += AxxAyy;
                spatialTerm[1] *= 2.0;
                spatialTerm[2] *= 2.0;
                spatialTerm[3] += AxxAyy;
                spatialTerm[4] *= 2.0;
                h_f.data[currentIndex] -= l*spatialTerm;
                }
            };
        };
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void landauDeGennesLC::computeForceOneConstantGPU(GPUArray<dVec> &forces, bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);

    forceTuner->begin();
    gpu_qTensor_oneConstantForce(d_force.data,
                              d_spins.data,
                              d_latticeTypes.data,
                              lattice->latticeIndex,
                              A,B,C,L1,
                              N,
                              zeroOutForce,
                              forceTuner->getParameter()
                              );
    forceTuner->end();
    };

void landauDeGennesLC::computeBoundaryForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    };

void landauDeGennesLC::computeBoundaryForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void landauDeGennesLC::computeEnergyCPU()
    {
    scalar phaseEnergy = 0.0;
    scalar distortionEnergy = 0.0;
    energy=0.0;
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int currentIndex;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    scalar l = L1;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        qCurrent = Qtensors.data[currentIndex];
        phaseEnergy += a*TrQ2(qCurrent) + b*TrQ3(qCurrent) + c* TrQ2Squared(qCurrent);

        xDown = Qtensors.data[neighbors[0]];
        xUp = Qtensors.data[neighbors[1]];
        yDown = Qtensors.data[neighbors[2]];
        yUp = Qtensors.data[neighbors[3]];
        zDown = Qtensors.data[neighbors[4]];
        zUp = Qtensors.data[neighbors[5]];

        dVec firstDerivativeX = 0.5*(xUp - xDown);
        distortionEnergy += l*(dot(firstDerivativeX,firstDerivativeX) + firstDerivativeX[0]*firstDerivativeX[3]);
        dVec firstDerivativeY = 0.5*(yUp - yDown);
        distortionEnergy += l*(dot(firstDerivativeY,firstDerivativeY) + firstDerivativeY[0]*firstDerivativeY[3]);
        dVec firstDerivativeZ = 0.5*(zUp - zDown);
        distortionEnergy += l*(dot(firstDerivativeZ,firstDerivativeZ) + firstDerivativeZ[0]*firstDerivativeZ[3]);

        };
    energy = (phaseEnergy + distortionEnergy) / lattice->getNumberOfParticles();
    };
