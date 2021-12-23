#include "landauDeGennesLC.h"
#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
#include "utilities.cuh"
/*! \file landauDeGennesLCOtherForces.cpp */

/*
This file splits out (parts of the LDG force calculation that aren't related to bulk and distortion
terms (utility functions, objects, external fields).
Keeps files and compilation more managable.
 */

void landauDeGennesLC::computeFirstDerivatives()
    {
    int N = lattice->getNumberOfParticles();
    if(forceCalculationAssist.getNumElements() < N)
        forceCalculationAssist.resize(N);
    if(useGPU)
        {
        ArrayHandle<cubicLatticeDerivativeVector> d_derivatives(forceCalculationAssist,access_location::device,access_mode::readwrite);
        ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
        ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
        ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::device,access_mode::read);
        forceAssistTuner->begin();
        gpu_qTensor_firstDerivatives(d_derivatives.data,
                                     d_spins.data,
                                     d_latticeTypes.data,
                                     latticeNeighbors.data,
                                     lattice->neighborIndex,
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
        for (int i = 0; i < N; ++i)
            {
            int neighNum;
            vector<int> neighbors(6);
            int idx;
            dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
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
                    if(h_latticeTypes.data[ixd] <=0 && h_latticeTypes.data[ixu] <= 0) //x bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][qq] = 0.5*(xUp[qq]-xDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[ixu] > 0) //right is boundary
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
                            h_derivatives.data[idx][qq] = (xUp[qq]-qCurrent[qq]);
                            };
                        };
                    if(h_latticeTypes.data[iyd] <=0 && h_latticeTypes.data[iyu] <= 0) //y bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][DIMENSION+qq] = 0.5*(yUp[qq]-yDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[iyu] > 0) //up is boundary
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
                            h_derivatives.data[idx][DIMENSION+qq] = (yUp[qq]-qCurrent[qq]);
                            };
                        };
                    if(h_latticeTypes.data[izd] <=0 && h_latticeTypes.data[izu] <= 0) //z bulk
                        {
                        for (int qq = 0; qq < DIMENSION; ++qq)
                            {
                            h_derivatives.data[idx][2*DIMENSION+qq] = 0.5*(zUp[qq]-zDown[qq]);
                            };
                        }
                    else if (h_latticeTypes.data[izu] > 0) //up is boundary
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
                            h_derivatives.data[idx][2*DIMENSION+qq] = (zUp[qq]-qCurrent[qq]);
                            };
                        };
                    }
                };
            };//end cpu loop over N
        }//end if -- else for using GPU
    };

void landauDeGennesLC::computeObjectForces(int objectIdx)
    {
    GPUArray<Matrix3x3> stressTensors;
    computeStressTensors(lattice->surfaceSites[objectIdx],stressTensors);

    if(!useGPU)
        {
        ArrayHandle<int> sites(lattice->surfaceSites[objectIdx],access_location::host,access_mode::read);
        ArrayHandle<int>  latticeTypes(lattice->returnTypes(),access_location::host,access_mode::read);
        ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::host,access_mode::read);
        ArrayHandle<Matrix3x3> stress(stressTensors,access_location::host,access_mode::read);
        for(int ii = 0; ii < lattice->surfaceSites[objectIdx].getNumElements();++ii)
            {
            int currentIndex = sites.data[ii];
            int ixd =latticeNeighbors.data[lattice->neighborIndex(0,currentIndex)];
            int ixu =latticeNeighbors.data[lattice->neighborIndex(1,currentIndex)];
            int iyd =latticeNeighbors.data[lattice->neighborIndex(2,currentIndex)];
            int iyu =latticeNeighbors.data[lattice->neighborIndex(3,currentIndex)];
            int izd =latticeNeighbors.data[lattice->neighborIndex(4,currentIndex)];
            int izu =latticeNeighbors.data[lattice->neighborIndex(5,currentIndex)];
            scalar3 surfaceArea = make_scalar3(0,0,0);
            if(latticeTypes.data[ixd] >0)
                surfaceArea.x = -1.0;
            if(latticeTypes.data[ixu] >0)
                surfaceArea.x = 1.0;
            if(latticeTypes.data[iyd] >0)
                surfaceArea.y = -1.0;
            if(latticeTypes.data[iyu] >0)
                surfaceArea.y = 1.0;
            if(latticeTypes.data[izd] >0)
                surfaceArea.z = -1.0;
            if(latticeTypes.data[izu] >0)
                surfaceArea.z = 1.0;
            lattice->boundaryForce[objectIdx] = lattice->boundaryForce[objectIdx]+ surfaceArea*stress.data[ii];
            }
        }
    else
        {
        int nSites = lattice->surfaceSites[objectIdx].getNumElements();
        if(objectForceArray.getNumElements() < nSites)
            objectForceArray.resize(nSites);
        {
        ArrayHandle<int> sites(lattice->surfaceSites[objectIdx],access_location::device,access_mode::read);
        ArrayHandle<int>  latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
        ArrayHandle<int> latticeNeighbors(lattice->neighboringSites,access_location::device,access_mode::read);
        ArrayHandle<Matrix3x3> stress(stressTensors,access_location::device,access_mode::read);
        ArrayHandle<scalar3> objectForces(objectForceArray,access_location::device,access_mode::overwrite);
        gpu_qTensor_computeObjectForceFromStresses(sites.data,
                                           latticeTypes.data,
                                           latticeNeighbors.data,
                                           stress.data,
                                           objectForces.data,
                                           lattice->neighborIndex,
                                           nSites,512);//temp maxBlockSize
        }//scope for device call
        ArrayHandle<scalar3> objectForces(objectForceArray,access_location::host,access_mode::read);
        for (int ii = 0; ii < nSites; ++ii)
            lattice->boundaryForce[objectIdx] = lattice->boundaryForce[objectIdx] + objectForces.data[ii];
        }
    printf("%f\t%f\t%f\n",lattice->boundaryForce[objectIdx].x,lattice->boundaryForce[objectIdx].y,lattice->boundaryForce[objectIdx].z);
    }

/*!
expression from "Hierarchical self-assembly of nematic colloidal superstructures"
PHYSICAL REVIEW E 77, 061706 (2008)
*/
void landauDeGennesLC::computeStressTensors(GPUArray<int> &sites,GPUArray<Matrix3x3> &stresses)
    {
    computeFirstDerivatives();
    int n = sites.getNumElements();
    if(stresses.getNumElements() < n)
        stresses.resize(n);
    if(numberOfConstants == distortionEnergyType::oneConstant)
        {
        if(true)//        if(!useGPU)
            {
            computeEnergyCPU(false);
            ArrayHandle<int> targetSites(sites,access_location::host,access_mode::read);
            ArrayHandle<Matrix3x3> stress(stresses,access_location::host,access_mode::overwrite);
            ArrayHandle<cubicLatticeDerivativeVector> h_derivatives(forceCalculationAssist,access_location::host,access_mode::read);
            ArrayHandle<scalar> energyPerSite(energyDensity,access_location::host,access_mode::read);
            for (int ii = 0; ii < n; ++ii)
                {
                int s=targetSites.data[ii];
                cubicLatticeDerivativeVector firstDerivative = h_derivatives.data[s];
                stress.data[ii].set(
                                    -2*L1*(firstDerivative[0]*firstDerivative[0] + 2*(firstDerivative[1]*firstDerivative[1]) + 2*(firstDerivative[2]*firstDerivative[2]) + firstDerivative[3]*firstDerivative[3] + 2*(firstDerivative[4]*firstDerivative[4]) + firstDerivative[0]*firstDerivative[3]),
                                    L1*(-(firstDerivative[5]*(2*firstDerivative[0] + firstDerivative[3])) - firstDerivative[8]*(firstDerivative[0] + 2*firstDerivative[3]) - 4*(firstDerivative[6]*firstDerivative[1] + firstDerivative[7]*firstDerivative[2] + firstDerivative[9]*firstDerivative[4])),
                                    L1*(-(firstDerivative[10]*(2*firstDerivative[0] + firstDerivative[3])) - firstDerivative[13]*(firstDerivative[0] + 2*firstDerivative[3]) - 4*(firstDerivative[11]*firstDerivative[1] + firstDerivative[12]*firstDerivative[2] + firstDerivative[14]*firstDerivative[4])),
                                    L1*(-(firstDerivative[5]*(2*firstDerivative[0] + firstDerivative[3])) - firstDerivative[8]*(firstDerivative[0] + 2*firstDerivative[3]) - 4*(firstDerivative[6]*firstDerivative[1] + firstDerivative[7]*firstDerivative[2] + firstDerivative[9]*firstDerivative[4])),
                                    -2*L1*(firstDerivative[5]*firstDerivative[5] + 2*(firstDerivative[6]*firstDerivative[6]) + 2*(firstDerivative[7]*firstDerivative[7]) + firstDerivative[8]*firstDerivative[8] + 2*(firstDerivative[9]*firstDerivative[9]) + firstDerivative[5]*firstDerivative[8]),
                                    L1*(-(firstDerivative[10]*(2*firstDerivative[5] + firstDerivative[8])) - firstDerivative[13]*(firstDerivative[5] + 2*firstDerivative[8]) - 4*(firstDerivative[11]*firstDerivative[6] + firstDerivative[12]*firstDerivative[7] + firstDerivative[14]*firstDerivative[9])),
                                    L1*(-(firstDerivative[10]*(2*firstDerivative[0] + firstDerivative[3])) - firstDerivative[13]*(firstDerivative[0] + 2*firstDerivative[3]) - 4*(firstDerivative[11]*firstDerivative[1] + firstDerivative[12]*firstDerivative[2] + firstDerivative[14]*firstDerivative[4])),
                                    L1*(-(firstDerivative[10]*(2*firstDerivative[5] + firstDerivative[8])) - firstDerivative[13]*(firstDerivative[5] + 2*firstDerivative[8]) - 4*(firstDerivative[11]*firstDerivative[6] + firstDerivative[12]*firstDerivative[7] + firstDerivative[14]*firstDerivative[9])),
                                    -2*L1*(firstDerivative[10]*firstDerivative[10] + 2*(firstDerivative[11]*firstDerivative[11]) + 2*(firstDerivative[12]*firstDerivative[12]) + firstDerivative[13]*firstDerivative[13] + 2*(firstDerivative[14]*firstDerivative[14]) + firstDerivative[10]*firstDerivative[13])
                                    );
                stress.data[ii].x11 += energyPerSite.data[s];
                stress.data[ii].x22 += energyPerSite.data[s];
                stress.data[ii].x33 += energyPerSite.data[s];
                }
            }//if not GPU
        }//if oneConstant
    };

void landauDeGennesLC::computeEorHFieldForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce,
                        scalar3 field, scalar anisotropicSusceptibility,scalar vacuumPermeability)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    fieldForceTuner->begin();
    gpu_qTensor_computeUniformFieldForcesGPU(d_force.data,
                              d_latticeTypes.data,
                              N,field,anisotropicSusceptibility,vacuumPermeability,zeroOutForce,
                              boundaryForceTuner->getParameter());
    fieldForceTuner->end();
    };

void landauDeGennesLC::computeSpatiallyVaryingFieldGPU(GPUArray<dVec> &forces,bool zeroOutForce,
                        GPUArray<scalar3> field, scalar anisotropicSusceptibility,scalar vacuumPermeability)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<scalar3> d_field(field,access_location::device,access_mode::read);
    fieldForceTuner->begin();
    gpu_qTensor_computeSpatiallyVaryingFieldForcesGPU(d_force.data,
                              d_latticeTypes.data,
                              N,d_field.data,anisotropicSusceptibility,vacuumPermeability,zeroOutForce,
                              boundaryForceTuner->getParameter());
    fieldForceTuner->end();
    };

void landauDeGennesLC::computeSpatiallyVaryingFieldCPU(GPUArray<dVec> &forces,bool zeroOutForce,GPUArray<scalar3> externalField, scalar anisotropicSusceptibility, scalar vacuumPermeability)
    {
    ArrayHandle<dVec> h_f(forces);
    ArrayHandle<scalar3> h_field(externalField,access_location::host,access_mode::read);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors(6);
    int currentIndex;
    scalar fieldProduct = anisotropicSusceptibility*vacuumPermeability;
    dVec fieldForce(0.);
    scalar3 field;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] > 0)//skip boundary sites
            continue;
        field = h_field.data[currentIndex];
        fieldForce[0] = -0.5*fieldProduct*(field.x*field.x-field.z*field.z);
        fieldForce[1] = -fieldProduct*field.x*field.y;
        fieldForce[2] = -fieldProduct*field.x*field.z;
        fieldForce[3] = -0.5*fieldProduct*(field.y*field.y-field.z*field.z);
        fieldForce[4] = -fieldProduct*field.y*field.z;
        h_f.data[currentIndex] -= fieldForce;
        };
    };

void landauDeGennesLC::computeEorHFieldForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce,
                    scalar3 field, scalar anisotropicSusceptibility,scalar vacuumPermeability)
    {
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    //the current scheme for getting the six nearest neighbors
    int neighNum;
    vector<int> neighbors(6);
    int currentIndex;
    scalar fieldProduct = anisotropicSusceptibility*vacuumPermeability;
    dVec fieldForce(0.);
    fieldForce[0] = -0.5*fieldProduct*(field.x*field.x-field.z*field.z);
    fieldForce[1] = -fieldProduct*field.x*field.y;
    fieldForce[2] = -fieldProduct*field.x*field.z;
    fieldForce[3] = -0.5*fieldProduct*(field.y*field.y-field.z*field.z);
    fieldForce[4] = -fieldProduct*field.y*field.z;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        if(latticeTypes.data[currentIndex] > 0)//skip boundary sites
            continue;
        h_f.data[currentIndex] -= fieldForce;
        };
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

void landauDeGennesLC::computeBoundaryForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    ArrayHandle<dVec> h_f(forces);
    if(zeroOutForce)
        for(int pp = 0; pp < lattice->getNumberOfParticles(); ++pp)
            h_f.data[pp] = make_dVec(0.0);
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<boundaryObject> bounds(lattice->boundaries);
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int currentIndex;
        dVec qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,tempForce;
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
                {
                computeBoundaryForce(qCurrent, xDown, bounds.data[latticeTypes.data[neighbors[0]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                }
            if(latticeTypes.data[neighbors[1]] > 0)
                {
                computeBoundaryForce(qCurrent, xUp, bounds.data[latticeTypes.data[neighbors[1]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                };
            if(latticeTypes.data[neighbors[2]] > 0)
                {
                computeBoundaryForce(qCurrent, yDown, bounds.data[latticeTypes.data[neighbors[2]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                }
            if(latticeTypes.data[neighbors[3]] > 0)
                {
                computeBoundaryForce(qCurrent, yUp, bounds.data[latticeTypes.data[neighbors[3]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                }
            if(latticeTypes.data[neighbors[4]] > 0)
                {
                computeBoundaryForce(qCurrent, zDown, bounds.data[latticeTypes.data[neighbors[4]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                }
            if(latticeTypes.data[neighbors[5]] > 0)
                {
                computeBoundaryForce(qCurrent, zUp, bounds.data[latticeTypes.data[neighbors[5]]-1],tempForce);
                h_f.data[currentIndex] += tempForce;
                }
            }
        }
    };

void landauDeGennesLC::setNumberOfConstants(distortionEnergyType _type)
    {
    numberOfConstants = _type;
    //if(numberOfConstants == distortionEnergyType::multiConstant)
//        printf("\n\n ***WARNING*** \nSome users have reported that the expressions used in multi-constant expressions for the distortion free energy forces may have an error in them. We are currently investigating\n***WARNING***\n\n");
//      DMS, Feb 22, 2021: I believe I have resolved the error in the lcForces.h file that gave rise to the problems with the multi-constant expressions
    };
