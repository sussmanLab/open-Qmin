#include "landauDeGennesLC.h"
#include "landauDeGennesLC.cuh"
#include "qTensorFunctions.h"
#include "utilities.cuh"
/*! \file landauDeGennesLC.cpp */

landauDeGennesLC::landauDeGennesLC(bool _neverGPU)
    {
    neverGPU = _neverGPU;
    if(neverGPU)
        {
        energyDensity.noGPU =true;
        energyDensityReduction.noGPU=true;
        objectForceArray.noGPU = true;
        forceCalculationAssist.noGPU=true;
        energyPerParticle.noGPU = true;
        }
    baseInitialization();
    }

landauDeGennesLC::landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1) :
    A(_A), B(_B), C(_C), L1(_L1)
    {
    baseInitialization();
    setNumberOfConstants(distortionEnergyType::oneConstant);
    };

landauDeGennesLC::landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2,scalar _L3orWavenumber,
                                   distortionEnergyType _type) :
                                   A(_A), B(_B), C(_C), L1(_L1), L2(_L2), L3(_L3orWavenumber), q0(_L3orWavenumber)
    {
    baseInitialization();
    setNumberOfConstants(_type);
    };

landauDeGennesLC::landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2,scalar _L3, scalar _L4, scalar _L6) :
                                   A(_A), B(_B), C(_C), L1(_L1), L2(_L2), L3(_L3), L4(_L4), L6(_L6)
    {
    baseInitialization();
    setNumberOfConstants(distortionEnergyType::multiConstant);
    };

void landauDeGennesLC::baseInitialization()
    {
    useNeighborList=false;
    computeEfieldContribution=false;
    computeHfieldContribution=false;
    forceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    boundaryForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    l24ForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    fieldForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    forceAssistTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    energyComponents.resize(5);
    }

void landauDeGennesLC::setModel(shared_ptr<cubicLattice> _model)
    {
    lattice=_model;
    model = _model;
    if(numberOfConstants == distortionEnergyType::multiConstant)
        {
        lattice->fillNeighborLists(0);//fill neighbor lists to allow computing mixed partials
        }
    else // one constant approx
        {
        lattice->fillNeighborLists(0);
        }
    int N = lattice->getNumberOfParticles();
    energyDensity.resize(N);
    energyDensityReduction.resize(N);
    /*
    forceCalculationAssist.resize(N);
    if(useGPU)
        {
        ArrayHandle<cubicLatticeDerivativeVector> fca(forceCalculationAssist,access_location::device,access_mode::overwrite);
        cubicLatticeDerivativeVector zero(0.0);
        gpu_set_array(fca.data,zero,N,512);
        }
    else
        {
        ArrayHandle<cubicLatticeDerivativeVector> fca(forceCalculationAssist);
        cubicLatticeDerivativeVector zero(0.0);
        for(int ii = 0; ii < N; ++ii)
            fca.data[ii] = zero;
        };
    */
    };

/*!
a function that loads the strength of a spatially varying field from specified files. THIS FUNCTION ASSUMES
that the files to be loaded are formatted so that each line looks like
x y z Hz Hy Hz
The expected format for every line is, thus:
%i %i %i %f %f %f
Additionally, so that the function works correctly on multi-rank simulations, the file names MUST HAVE
the _x%i_y%i_z%i.txt ending expected to specifiy what rank the file corresponds to.
(so, even if you are doing non-MPI simulations, your file must be named yourFileName_x0y0z0.txt and you would call the function with something like
string loadingFileName = "yourFileName";
sim->loadSpatiallyVaryingField(loadingFileName);
*/
void landauDeGennesLC::setSpatiallyVaryingField(string fname, scalar chi, scalar _mu0,scalar _deltaChi, int3 rankParity)
    {
    Chi =chi;
    mu0=_mu0;
    deltaChi=_deltaChi;
    spatiallyVaryingFieldContribution = true;
    int N = lattice->getNumberOfParticles();

    //initialize field to all zero values
    double3 zero = make_double3(0,0,0);
    vector<double3> zeroVector(N,zero);
    fillGPUArrayWithVector(zeroVector, spatiallyVaryingField);
    char fn[256];
    sprintf(fn,"%s_x%iy%iz%i.txt",fname.c_str(),rankParity.x,rankParity.y,rankParity.z);

    printf("loading spatially varying field from file name %s...\n",fn);
    int xOffset = rankParity.x*lattice->latticeSites.x;
    int yOffset = rankParity.y*lattice->latticeSites.y;
    int zOffset = rankParity.z*lattice->latticeSites.z;
    ArrayHandle<scalar3> hh(spatiallyVaryingField);
    ifstream myfile;
    myfile.open(fn);
    if(myfile.fail())
        {
        printf("\nERROR trying to load file named %s\n",fn);
        printf("\nYou have tried to load a file that either does not exist or that you do not have permission to access! \n Error in file %s at line %d\n",__FILE__,__LINE__);
        throw std::exception();
        }
    int px,py,pz;
    double Hx,Hy,Hz;
    while(myfile >> px >> py >>pz >> Hx >> Hy >> Hz)
        {
        int3 pos;
        pos.x = px - xOffset;
        pos.y = py - yOffset;
        pos.z = pz - zOffset;
        int idx = lattice->positionToIndex(pos);
        hh.data[idx].x = Hx;
        hh.data[idx].y = Hy;
        hh.data[idx].z = Hz;
        }
    myfile.close();
    }

void landauDeGennesLC::computeForces(GPUArray<dVec> &forces,bool zeroOutForce, int type)
    {
    if(useGPU)
        computeForceGPU(forces,zeroOutForce);
    else
        computeForceCPU(forces,zeroOutForce,type);

    correctForceFromMetric(forces);
    }

void landauDeGennesLC::correctForceFromMetric(GPUArray<dVec> &forces)
    {
    int N = lattice->getNumberOfParticles();
    if(useGPU)
        {
        ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
        gpuCorrectForceFromMetric(d_force.data,N,forceTuner->getParameter());
        }
    else
        {
        ArrayHandle<dVec> h_force(forces,access_location::host,access_mode::readwrite);
        scalar QxxOld, QyyOld;
        scalar twoThirds = 2./3.;
        for (int i = 0; i < N ; ++i)
            {
            QxxOld = h_force.data[i][0];
            QyyOld = h_force.data[i][3];
            h_force.data[i][0] = twoThirds*(2.*QxxOld - QyyOld);
            h_force.data[i][3] = twoThirds*(2.*QyyOld - QxxOld);
            };
        }
    }

void landauDeGennesLC::computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeNeighbors(lattice->neighboringSites,access_location::device,access_mode::read);
    forceTuner->begin();
    switch (numberOfConstants)
        {
        case distortionEnergyType::oneConstant :
            {
            gpu_qTensor_oneConstantForce(d_force.data, d_spins.data, d_latticeTypes.data, d_latticeNeighbors.data,
                                         lattice->neighborIndex,
                                         A,B,C,L1,N,
                                         zeroOutForce,forceTuner->getParameter());
            break;
            };
        case distortionEnergyType::multiConstant:
            {
            bool zeroForce = zeroOutForce;
            computeFirstDerivatives();
            ArrayHandle<cubicLatticeDerivativeVector> d_derivatives(forceCalculationAssist,access_location::device,access_mode::read);
            gpu_qTensor_multiConstantForce(d_force.data, d_spins.data, d_latticeTypes.data, d_derivatives.data,
                                        d_latticeNeighbors.data, lattice->neighborIndex,
                                         A,B,C,L1,L2,L3,L4,L6,N,
                                         zeroOutForce,forceTuner->getParameter());
            break;
            };
        };
    forceTuner->end();
    if(lattice->boundaries.getNumElements() >0)
        {
        computeBoundaryForcesGPU(forces,false);
        };
    if(computeEfieldContribution)
        computeEorHFieldForcesGPU(forces,false,Efield,deltaEpsilon,epsilon0);
    if(computeHfieldContribution)
        computeEorHFieldForcesGPU(forces,false,Hfield,deltaChi,mu0);
    if(spatiallyVaryingFieldContribution)
        computeSpatiallyVaryingFieldGPU(forces,false,spatiallyVaryingField, deltaChi,mu0);
    };

void landauDeGennesLC::computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce, int type)
    {
    switch (numberOfConstants)
        {
        case distortionEnergyType::oneConstant :
            {
            if(type ==0)
                computeL1BulkCPU(forces,zeroOutForce);
            if(type ==1)
                computeL1BoundaryCPU(forces,zeroOutForce);
            break;
            };
        case distortionEnergyType::multiConstant :
            {
            bool zeroForce = zeroOutForce;
            computeFirstDerivatives();
            if(type ==0)
                computeAllDistortionTermsBulkCPU(forces,zeroForce);
            if(type ==1)
                computeAllDistortionTermsBoundaryCPU(forces,zeroForce);
            break;
            };
        };
    if(type != 0 )
        {
        if(lattice->boundaries.getNumElements() >0)
            {
            computeBoundaryForcesCPU(forces,false);
            };
        if(computeEfieldContribution)
            {
            computeEorHFieldForcesCPU(forces,false, Efield,deltaEpsilon,epsilon0);
            };
        if(computeHfieldContribution)
            {
            computeEorHFieldForcesCPU(forces,false,Hfield,deltaChi,mu0);
            };
        if(spatiallyVaryingFieldContribution)
            computeSpatiallyVaryingFieldCPU(forces,false,spatiallyVaryingField, deltaChi,mu0);
        };
    };

void landauDeGennesLC::computeEnergyGPU(bool verbose)
    {
    int N = lattice->getNumberOfParticles();
    energy=0.0;
    {//scope for initial arrays
    ArrayHandle<dVec> Qtensors(lattice->returnPositions(),access_location::device,access_mode::read);
    ArrayHandle<int> latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<boundaryObject> bounds(lattice->boundaries,access_location::device,access_mode::read);
    ArrayHandle<scalar> energyPerSite(energyDensity,access_location::device,access_mode::readwrite);
    ArrayHandle<int>  latticeNeighbors(lattice->neighboringSites,access_location::device,access_mode::read);
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    gpu_computeAllEnergyTerms(energyPerSite.data,Qtensors.data,latticeTypes.data,bounds.data,
                                latticeNeighbors.data, lattice->neighborIndex,
                            a,b,c,L1,L2,L3,L4,L6,
                            computeEfieldContribution,computeHfieldContribution,
                            epsilon,epsilon0,deltaEpsilon,Efield,
                            Chi,mu0,deltaChi,Hfield,
                            N);
    }

    //now reduce to get total energy
    int numBlocks = 0;
    int numThreads = 0;
    int maxBlocks = 64;
    int maxThreads = 256;
    getNumBlocksAndThreads(N, maxBlocks, maxThreads, numBlocks, numThreads);
    ArrayHandle<scalar> energyPerSite(energyDensity,access_location::device,access_mode::read);
    ArrayHandle<scalar> energyPerSiteReduction(energyDensityReduction,access_location::device,access_mode::readwrite);
    energy = gpuReduction(N,numThreads,numBlocks,maxThreads,maxBlocks,energyPerSite.data,energyPerSiteReduction.data);
    }

void landauDeGennesLC::computeEnergyCPU(bool verbose)
    {
    scalar phaseEnergy = 0.0;
    scalar distortionEnergy = 0.0;
    scalar anchoringEnergy = 0.0;
    scalar eFieldEnergy = 0.0;
    scalar hFieldEnergy = 0.0;
    energy=0.0;
    ArrayHandle<dVec> Qtensors(lattice->returnPositions());
    ArrayHandle<int> latticeTypes(lattice->returnTypes());
    ArrayHandle<boundaryObject> bounds(lattice->boundaries);
    ArrayHandle<scalar> energyPerSite(energyDensity);
    ArrayHandle<scalar3> externalField(spatiallyVaryingField);
    scalar a = 0.5*A;
    scalar b = B/3.0;
    scalar c = 0.25*C;
    int LCSites = 0;
    for (int i = 0; i < lattice->getNumberOfParticles(); ++i)
        {
        energyPerSite.data[i] = 0.0;
        //the current scheme for getting the six nearest neighbors
        int neighNum;
        vector<int> neighbors(6);
        int currentIndex;
        dVec qCurrent, xDown, xUp, yDown,yUp,zDown,zUp;
        currentIndex = lattice->getNeighbors(i,neighbors,neighNum);
        qCurrent = Qtensors.data[currentIndex];
        if(latticeTypes.data[currentIndex] <=0)
            {
            LCSites +=1;
            scalar phaseAtSite = a*TrQ2(qCurrent) + b*TrQ3(qCurrent) + c* TrQ2Squared(qCurrent);
            energyPerSite.data[i] += phaseAtSite;
            phaseEnergy += phaseAtSite;

            if(computeEfieldContribution)
                {
                    scalar eFieldAtSite = epsilon0*(-0.5*Efield.x*Efield.x*(epsilon + deltaEpsilon*qCurrent[0]) -
                              deltaEpsilon*Efield.x*Efield.y*qCurrent[1] - deltaEpsilon*Efield.x*Efield.z*qCurrent[2] -
                              0.5*Efield.z*Efield.z*(epsilon - deltaEpsilon*qCurrent[0] - deltaEpsilon*qCurrent[3]) -
                              0.5*Efield.y*Efield.y*(epsilon + deltaEpsilon*qCurrent[3]) - deltaEpsilon*Efield.y*Efield.z*qCurrent[4]);
                    eFieldEnergy+=eFieldAtSite;
                    energyPerSite.data[i] +=eFieldAtSite;
                }
            if(computeHfieldContribution)
                {
                    scalar hFieldAtSite=mu0*(-0.5*Hfield.x*Hfield.x*(Chi + deltaChi*qCurrent[0]) -
                              deltaChi*Hfield.x*Hfield.y*qCurrent[1] - deltaChi*Hfield.x*Hfield.z*qCurrent[2] -
                              0.5*Hfield.z*Hfield.z*(Chi - deltaChi*qCurrent[0] - deltaChi*qCurrent[3]) -
                              0.5*Hfield.y*Hfield.y*(Chi + deltaChi*qCurrent[3]) - deltaChi*Hfield.y*Hfield.z*qCurrent[4]);
                    hFieldEnergy+=hFieldAtSite;
                    energyPerSite.data[i] +=hFieldAtSite;
                }
            if(spatiallyVaryingFieldContribution)
                {
                    scalar3 field = externalField.data[currentIndex];
                    scalar hFieldAtSite=mu0*(-0.5*field.x*field.x*(Chi + deltaChi*qCurrent[0]) -
                              deltaChi*field.x*field.y*qCurrent[1] - deltaChi*field.x*field.z*qCurrent[2] -
                              0.5*field.z*field.z*(Chi - deltaChi*qCurrent[0] - deltaChi*qCurrent[3]) -
                              0.5*field.y*field.y*(Chi + deltaChi*qCurrent[3]) - deltaChi*field.y*field.z*qCurrent[4]);
                    hFieldEnergy+=hFieldAtSite;
                    energyPerSite.data[i] +=hFieldAtSite;
                }
            xDown = Qtensors.data[neighbors[0]];
            xUp = Qtensors.data[neighbors[1]];
            yDown = Qtensors.data[neighbors[2]];
            yUp = Qtensors.data[neighbors[3]];
            zDown = Qtensors.data[neighbors[4]];
            zUp = Qtensors.data[neighbors[5]];

            dVec firstDerivativeX = 0.5*(xUp - xDown);
            dVec firstDerivativeY = 0.5*(yUp - yDown);
            dVec firstDerivativeZ = 0.5*(zUp - zDown);
            scalar anchoringEnergyAtSite = 0.0;
            if(latticeTypes.data[currentIndex] <0)
                {
                if(latticeTypes.data[neighbors[0]]>0)
                    {
                    anchoringEnergyAtSite+= computeBoundaryEnergy(qCurrent, xDown, bounds.data[latticeTypes.data[neighbors[0]]-1]);
                    firstDerivativeX = xUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[1]]>0)
                    {
                    anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, xUp, bounds.data[latticeTypes.data[neighbors[1]]-1]);
                    firstDerivativeX = qCurrent - xDown;
                    }
                if(latticeTypes.data[neighbors[2]]>0)
                    {
                    anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, yDown, bounds.data[latticeTypes.data[neighbors[2]]-1]);
                    firstDerivativeY = yUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[3]]>0)
                    {
                    anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, yUp, bounds.data[latticeTypes.data[neighbors[3]]-1]);
                    firstDerivativeY = qCurrent - yDown;
                    }
                if(latticeTypes.data[neighbors[4]]>0)
                    {
                    anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, zDown, bounds.data[latticeTypes.data[neighbors[4]]-1]);
                    firstDerivativeZ = zUp - qCurrent;
                    }
                if(latticeTypes.data[neighbors[5]]>0)
                    {
                    anchoringEnergyAtSite += computeBoundaryEnergy(qCurrent, zUp, bounds.data[latticeTypes.data[neighbors[5]]-1]);
                    firstDerivativeZ = qCurrent - zDown;
                    }
                anchoringEnergy += anchoringEnergyAtSite;
                energyPerSite.data[i] +=anchoringEnergyAtSite;
                }
            scalar distortionEnergyAtSite=0.0;
            if(L1 !=0 )
        		{
        		distortionEnergyAtSite+=L1*(firstDerivativeX[0]*firstDerivativeX[3] + firstDerivativeY[0]*firstDerivativeY[3] + firstDerivativeZ[0]*firstDerivativeZ[3] + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeX[3]*firstDerivativeX[3] + firstDerivativeX[4]*firstDerivativeX[4] + firstDerivativeY[0]*firstDerivativeY[0]
                                        + firstDerivativeY[1]*firstDerivativeY[1] + firstDerivativeY[2]*firstDerivativeY[2] + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4] + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[1]*firstDerivativeZ[1] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]);
        		};
        	if(L2 !=0 )
        		{
        		distortionEnergyAtSite+=(L2*(2*firstDerivativeX[2]*firstDerivativeY[4] - 2*firstDerivativeX[2]*firstDerivativeZ[0] - 2*firstDerivativeY[4]*firstDerivativeZ[0] + 2*firstDerivativeY[1]*firstDerivativeZ[2] + 2*firstDerivativeX[0]*(firstDerivativeY[1] + firstDerivativeZ[2]) - 2*firstDerivativeX[2]*firstDerivativeZ[3] - 2*firstDerivativeY[4]*firstDerivativeZ[3] + 2*firstDerivativeZ[0]*firstDerivativeZ[3]
                                        + 2*firstDerivativeY[3]*firstDerivativeZ[4] + 2*firstDerivativeX[1]*(firstDerivativeY[3] + firstDerivativeZ[4]) + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeY[1]*firstDerivativeY[1] + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4]
                                        + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]))/2.;
        		};
        	if(L3 !=0 )
        		{
        		distortionEnergyAtSite+=(L3*(2*firstDerivativeX[1]*firstDerivativeY[0] + 2*firstDerivativeX[3]*firstDerivativeY[1] + 2*firstDerivativeX[4]*firstDerivativeY[2] + 2*firstDerivativeX[2]*firstDerivativeZ[0] + 2*firstDerivativeX[4]*firstDerivativeZ[1] + 2*firstDerivativeY[2]*firstDerivativeZ[1] - 2*firstDerivativeX[0]*firstDerivativeZ[2] - 2*firstDerivativeX[3]*firstDerivativeZ[2]
                                        + 2*firstDerivativeY[4]*firstDerivativeZ[3] + 2*firstDerivativeZ[0]*firstDerivativeZ[3] - 2*firstDerivativeY[0]*firstDerivativeZ[4] - 2*firstDerivativeY[3]*firstDerivativeZ[4] + firstDerivativeX[0]*firstDerivativeX[0] + firstDerivativeX[1]*firstDerivativeX[1] + firstDerivativeX[2]*firstDerivativeX[2] + firstDerivativeY[1]*firstDerivativeY[1]
                                        + firstDerivativeY[3]*firstDerivativeY[3] + firstDerivativeY[4]*firstDerivativeY[4] + firstDerivativeZ[0]*firstDerivativeZ[0] + firstDerivativeZ[2]*firstDerivativeZ[2] + firstDerivativeZ[3]*firstDerivativeZ[3] + firstDerivativeZ[4]*firstDerivativeZ[4]))/2.;
        		};
        	if(L4 !=0 )
        		{
        		distortionEnergyAtSite+=(L4*(-(firstDerivativeY[4]*qCurrent[0]) + firstDerivativeZ[4]*qCurrent[0] + firstDerivativeX[2]*qCurrent[1] - firstDerivativeY[4]*qCurrent[1] - firstDerivativeZ[2]*qCurrent[1] + firstDerivativeZ[4]*qCurrent[1] - firstDerivativeY[4]*qCurrent[2] + firstDerivativeZ[4]*qCurrent[2] + firstDerivativeX[2]*qCurrent[3] - firstDerivativeZ[2]*qCurrent[3]
                                        + firstDerivativeX[1]*(qCurrent[0] - qCurrent[2] + qCurrent[3] - qCurrent[4]) + firstDerivativeX[2]*qCurrent[4] - firstDerivativeZ[2]*qCurrent[4] + firstDerivativeY[1]*(-qCurrent[0] + qCurrent[2] - qCurrent[3] + qCurrent[4])))/2.;
        		};
        	if(L6 !=0 )
        		{
        		distortionEnergyAtSite+=L6*(-(firstDerivativeZ[0]*firstDerivativeZ[3]*qCurrent[0]) + firstDerivativeX[0]*firstDerivativeX[0]*qCurrent[0] + firstDerivativeX[1]*firstDerivativeX[1]*qCurrent[0] + firstDerivativeX[2]*firstDerivativeX[2]*qCurrent[0] + firstDerivativeX[3]*firstDerivativeX[3]*qCurrent[0] + firstDerivativeX[4]*firstDerivativeX[4]*qCurrent[0]
                                        - firstDerivativeZ[0]*firstDerivativeZ[0]*qCurrent[0] - firstDerivativeZ[1]*firstDerivativeZ[1]*qCurrent[0] - firstDerivativeZ[2]*firstDerivativeZ[2]*qCurrent[0] - firstDerivativeZ[3]*firstDerivativeZ[3]*qCurrent[0] - firstDerivativeZ[4]*firstDerivativeZ[4]*qCurrent[0] + firstDerivativeX[3]*firstDerivativeY[0]*qCurrent[1]
                                        + 2*firstDerivativeX[2]*firstDerivativeY[2]*qCurrent[1] + 2*firstDerivativeX[3]*firstDerivativeY[3]*qCurrent[1] + 2*firstDerivativeX[4]*firstDerivativeY[4]*qCurrent[1] + firstDerivativeX[3]*firstDerivativeZ[0]*qCurrent[2] + 2*firstDerivativeX[2]*firstDerivativeZ[2]*qCurrent[2] + 2*firstDerivativeX[3]*firstDerivativeZ[3]*qCurrent[2]
                                        + 2*firstDerivativeX[4]*firstDerivativeZ[4]*qCurrent[2] + 2*firstDerivativeX[1]*(firstDerivativeY[1]*qCurrent[1] + firstDerivativeZ[1]*qCurrent[2]) + firstDerivativeX[0]*(firstDerivativeX[3]*qCurrent[0] + 2*firstDerivativeY[0]*qCurrent[1] + firstDerivativeY[3]*qCurrent[1] + 2*firstDerivativeZ[0]*qCurrent[2] + firstDerivativeZ[3]*qCurrent[2])
                                        + firstDerivativeY[0]*firstDerivativeY[3]*qCurrent[3] - firstDerivativeZ[0]*firstDerivativeZ[3]*qCurrent[3] + firstDerivativeY[0]*firstDerivativeY[0]*qCurrent[3] + firstDerivativeY[1]*firstDerivativeY[1]*qCurrent[3] + firstDerivativeY[2]*firstDerivativeY[2]*qCurrent[3] + firstDerivativeY[3]*firstDerivativeY[3]*qCurrent[3]
                                        + firstDerivativeY[4]*firstDerivativeY[4]*qCurrent[3] - firstDerivativeZ[0]*firstDerivativeZ[0]*qCurrent[3] - firstDerivativeZ[1]*firstDerivativeZ[1]*qCurrent[3] - firstDerivativeZ[2]*firstDerivativeZ[2]*qCurrent[3] - firstDerivativeZ[3]*firstDerivativeZ[3]*qCurrent[3] - firstDerivativeZ[4]*firstDerivativeZ[4]*qCurrent[3]
                                        + 2*firstDerivativeY[0]*firstDerivativeZ[0]*qCurrent[4] + firstDerivativeY[3]*firstDerivativeZ[0]*qCurrent[4] + 2*firstDerivativeY[1]*firstDerivativeZ[1]*qCurrent[4] + 2*firstDerivativeY[2]*firstDerivativeZ[2]*qCurrent[4] + firstDerivativeY[0]*firstDerivativeZ[3]*qCurrent[4] + 2*firstDerivativeY[3]*firstDerivativeZ[3]*qCurrent[4] + 2*firstDerivativeY[4]*firstDerivativeZ[4]*qCurrent[4]);
        		};

            distortionEnergy+=distortionEnergyAtSite;
            energyPerSite.data[i] +=distortionEnergyAtSite;
            }

        };
    energy = (phaseEnergy + distortionEnergy + anchoringEnergy + eFieldEnergy + hFieldEnergy);
    energyComponents[0] = phaseEnergy;
    energyComponents[1] = distortionEnergy;
    energyComponents[2] = anchoringEnergy;
    energyComponents[3] = eFieldEnergy;
    energyComponents[4] = hFieldEnergy;

    if(verbose)
        printf("%f %f %f %f %f\n",phaseEnergy , distortionEnergy , anchoringEnergy , eFieldEnergy , hFieldEnergy);
    };
