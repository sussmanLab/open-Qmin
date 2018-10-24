#include "landauDeGennesLC.h"
#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
#include "utilities.cuh"
/*! \file landauDeGennesLC.cpp */

landauDeGennesLC::landauDeGennesLC(double _A, double _B, double _C, double _L1) :
    A(_A), B(_B), C(_C), L1(_L1)
    {
    useNeighborList=false;
    numberOfConstants = distortionEnergyType::oneConstant;
    forceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    boundaryForceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    };

landauDeGennesLC::landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2,scalar _L3orWavenumber,
                                   distortionEnergyType _type) :
                                   A(_A), B(_B), C(_C), L1(_L1), L2(_L2), L3(_L3orWavenumber), q0(_L3orWavenumber)
    {
    useNeighborList=false;
    numberOfConstants = _type;
    forceTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    forceAssistTuner = make_shared<kernelTuner>(16,256,16,5,200000);
    };

void landauDeGennesLC::setModel(shared_ptr<cubicLattice> _model)
    {
    lattice=_model;
    model = _model;
    if(numberOfConstants == distortionEnergyType::twoConstant ||
        numberOfConstants == distortionEnergyType::threeConstant)
        {
        int N = lattice->getNumberOfParticles();
        forceCalculationAssist.resize(N);
        if(useGPU)
            {
            ArrayHandle<cubicLatticeDerivativeVector> fca(forceCalculationAssist,access_location::device,access_mode::overwrite);
            gpu_zero_array(fca.data,N);
            }
        else
            {
            ArrayHandle<cubicLatticeDerivativeVector> fca(forceCalculationAssist);
            cubicLatticeDerivativeVector zero(0.0);
            for(int ii = 0; ii < N; ++ii)
                fca.data[ii] = zero;
            };
        };
    };

//!Precompute the first derivatives at all of the LC LCSites
void landauDeGennesLC::computeFirstDerivatives()
    {
    int N = lattice->getNumberOfParticles();
    if(useGPU)
        {
        ArrayHandle<cubicLatticeDerivativeVector> d_derivatives(forceCalculationAssist,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
        ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
        forceAssistTuner->begin();
        gpu_qTensor_firstDerivatives(d_derivatives.data,
                                  d_spins.data,
                                  d_latticeTypes.data,
                                  lattice->latticeIndex,
                                  N,
                                  forceAssistTuner->getParameter()
                                  );
        forceAssistTuner->end();
        }
    else
        {
        ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist);
        ArrayHandle<dVec> Qtensors(lattice->returnPositions(),access_location::host,access_mode::read);
        ArrayHandle<int>  h_latticeTypes(lattice->returnTypes(),access_location::host,access_mode::read);
        int neighNum;
        vector<int> neighbors;
        int idx;
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        for (int i = 0; i < N; ++i)
            {
            idx = lattice->getNeighbors(i,neighbors,neighNum);
            if(h_latticeTypes.data[idx] <= 0)
                {
                qCurrent = Qtensors.data[idx];
                int ixd = neighbors[0]; int ixu = neighbors[1];
                int iyd = neighbors[2]; int iyu = neighbors[3];
                int izd = neighbors[4]; int izu = neighbors[5];
                xDown = Qtensors.data[ixd]; xUp = Qtensors.data[ixu];
                yDown = Qtensors.data[iyd]; yUp = Qtensors.data[iyu];
                zDown = Qtensors.data[izd]; zUp = Qtensors.data[izu];

                if(h_latticeTypes.data[idx] == 0) // if it's in the bulk, things are easy
                    {
                    for (int qq = 0; qq < DIMENSION; ++qq)
                        {
                        h_derivatives.data[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                        };
                    for (int qq = 0; qq < DIMENSION; ++qq)
                        {
                        h_derivatives.data[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                        };
                    for (int qq = 0; qq < DIMENSION; ++qq)
                        {
                        h_derivatives.data[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                        };
                    }
                else // boundary terms are slightly more work
                    {
                    if(h_latticeTypes.data[ixd] <=0 ||h_latticeTypes.data[ixu] <= 0) //x bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[ixd] <=0 ||h_latticeTypes.data[ixu] > 0) //right is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][qq] = (qCurrent[qq]-xDown[qq]);
                            };
                        }
                    else//left is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][qq] = (qCurrent[qq]-xUp[qq]);
                            };
                        };
                    if(h_latticeTypes.data[iyd] <=0 ||h_latticeTypes.data[iyu] <= 0) //y bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[iyd] <=0 ||h_latticeTypes.data[iyu] > 0) //up is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][DIMENSION+qq] = (qCurrent[qq]-yDown[qq]);
                            };
                        }
                    else//down is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][DIMENSION+qq] = (qCurrent[qq]-yUp[qq]);
                            };
                        };
                    if(h_latticeTypes.data[izd] <=0 ||h_latticeTypes.data[izu] <= 0) //z bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[izd] <=0 ||h_latticeTypes.data[izu] > 0) //up is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][2*DIMENSION+qq] = (qCurrent[qq]-zDown[qq]);
                            };
                        }
                    else//down is boundary
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][2*DIMENSION+qq] = (qCurrent[qq]-zUp[qq]);
                            };
                        };
                    }
                };
            };//end cpu loop over N
        }//end if -- else for using GPU
    };

//!compute the phase and distortion parts of the Landau energy. Handles all sites with type <=0
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

    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<boundaryObject> bounds(lattice->boundaries);
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int currentIndex;
    dVec qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,tempForce;

    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] < 0)
            {
            qCurrent = Qtensors.data[currentIndex];
            xDown = Qtensors.data[neighbors[0]];
            xUp = Qtensors.data[neighbors[1]];
            yDown = Qtensors.data[neighbors[2]];
            yUp = Qtensors.data[neighbors[3]];
            zDown = Qtensors.data[neighbors[4]];
            zUp = Qtensors.data[neighbors[5]];

            if(latticeTypes.data[neighbors[0]] > 0)
                computeBoundaryForce(qCurrent, xDown, bounds.data[latticeTypes.data[neighbors[0]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            if(latticeTypes.data[neighbors[1]] > 0)
                computeBoundaryForce(qCurrent, xUp, bounds.data[latticeTypes.data[neighbors[1]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            if(latticeTypes.data[neighbors[2]] > 0)
                computeBoundaryForce(qCurrent, yDown, bounds.data[latticeTypes.data[neighbors[2]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            if(latticeTypes.data[neighbors[3]] > 0)
                computeBoundaryForce(qCurrent, yUp, bounds.data[latticeTypes.data[neighbors[3]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            if(latticeTypes.data[neighbors[4]] > 0)
                computeBoundaryForce(qCurrent, zDown, bounds.data[latticeTypes.data[neighbors[4]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            if(latticeTypes.data[neighbors[5]] > 0)
                computeBoundaryForce(qCurrent, zUp, bounds.data[latticeTypes.data[neighbors[5]]-1],tempForce);
            h_f.data[currentIndex] += tempForce;
            }
        }
    };

void landauDeGennesLC::computeBoundaryForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<boundaryObject> d_bounds(lattice->boundaries,access_location::device,access_mode::read);
    boundaryForceTuner->begin();
    gpu_qTensor_computeBoundaryForcesGPU(d_force.data,
                              d_spins.data,
                              d_latticeTypes.data,
                              d_bounds.data,
                              lattice->latticeIndex,
                              N,
                              zeroOutForce,
                              boundaryForceTuner->getParameter()
                              );
    boundaryForceTuner->end();
    };

//!As an example of usage, we'll implement an n-Vector model force w/ nearest-neighbor interactions
void landauDeGennesLC::computeEnergyCPU()
    {
    scalar phaseEnergy = 0.0;
    scalar distortionEnergy = 0.0;
    scalar anchoringEnergy = 0.0;
    energy=0.0;
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<boundaryObject> bounds(lattice->boundaries);
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors;
    int currentIndex;
    dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    scalar l = L1;
    int LCSites = 0;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        qCurrent = Qtensors.data[currentIndex];
        if(latticeTypes.data[currentIndex] <=0)
            {
            LCSites +=1;
            phaseEnergy += a*TrQ2(qCurrent) + b*TrQ3(qCurrent) + c* TrQ2Squared(qCurrent);

            xDown = Qtensors.data[neighbors[0]];
            xUp = Qtensors.data[neighbors[1]];
            yDown = Qtensors.data[neighbors[2]];
            yUp = Qtensors.data[neighbors[3]];
            zDown = Qtensors.data[neighbors[4]];
            zUp = Qtensors.data[neighbors[5]];

            dVec firstDerivativeX = 0.5*(xUp - xDown);
            dVec firstDerivativeY = 0.5*(yUp - yDown);
            dVec firstDerivativeZ = 0.5*(zUp - zDown);
            if(latticeTypes.data[currentIndex] <0)
                {
                if(latticeTypes.data[neighbors[0]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, xDown, bounds.data[latticeTypes.data[neighbors[0]]-1]);
                    firstDerivativeX = xUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[1]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, xUp, bounds.data[latticeTypes.data[neighbors[1]]-1]);
                    firstDerivativeX = qCurrent - xDown;
                    }
                if(latticeTypes.data[neighbors[2]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, yDown, bounds.data[latticeTypes.data[neighbors[2]]-1]);
                    firstDerivativeY = yUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[3]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, yUp, bounds.data[latticeTypes.data[neighbors[3]]-1]);
                    firstDerivativeY = qCurrent - yDown;
                    }
                if(latticeTypes.data[neighbors[4]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, zDown, bounds.data[latticeTypes.data[neighbors[4]]-1]);
                    firstDerivativeZ = zUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[5]]>0)
                    {
                    anchoringEnergy += computeBoundaryEnergy(qCurrent, zUp, bounds.data[latticeTypes.data[neighbors[5]]-1]);
                    firstDerivativeZ = qCurrent - zDown;
                    }
                }
            distortionEnergy += l*(dot(firstDerivativeX,firstDerivativeX) + firstDerivativeX[0]*firstDerivativeX[3]);
            distortionEnergy += l*(dot(firstDerivativeY,firstDerivativeY) + firstDerivativeY[0]*firstDerivativeY[3]);
            distortionEnergy += l*(dot(firstDerivativeZ,firstDerivativeZ) + firstDerivativeZ[0]*firstDerivativeZ[3]);


            }
        };
    energy = (phaseEnergy + distortionEnergy + anchoringEnergy) / LCSites;
    };
