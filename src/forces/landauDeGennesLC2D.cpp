#include "landauDeGennesLC2D.h"
#include "landauDeGennesLC2D.cuh"
#include "qTensorFunctions2D.h"
#include "utilities.cuh"
/*! \file landauDeGennesLC2D.cpp */

landauDeGennesLC2D::landauDeGennesLC2D(bool _neverGPU)
    {
    neverGPU = _neverGPU;
    baseInitialization();
    }

landauDeGennesLC2D::landauDeGennesLC2D(scalar _A, scalar _C, scalar _L1, bool _neverGPU) :
    A(_A), C(_C), L1(_L1)
    {
    neverGPU = _neverGPU;
    baseInitialization();
    setNumberOfConstants(distortionEnergyType2D::oneConstant);
    };

void landauDeGennesLC2D::baseInitialization()
    {
    if(neverGPU)
        {
        energyDensity.noGPU =true;
        energyDensityReduction.noGPU=true;
        objectForceArray.noGPU = true;
        forceCalculationAssist.noGPU=true;
        energyPerParticle.noGPU = true;
        }
    useNeighborList=false;
    forceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    boundaryForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    l24ForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    fieldForceTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    forceAssistTuner = make_shared<kernelTuner>(128,256,32,10,200000);
    energyComponents.resize(5);
    }

void landauDeGennesLC2D::setModel(shared_ptr<squareLattice> _model)
    {
    lattice=_model;
    model = _model;
    //DMS: Note that we have moved away from the stenciled neighbor list approach and instead only need various first derivatives. Hence the precomputation of the first derivatives and the cubicLatticeDerivative vectors
    //I've left the functionality to (in principle) allow for more complex distortion terms, but for now we only need to use stencilType=1, which corresponds to a nine-point stencil in 2D
    lattice->fillNeighborLists(1);//fill neighbor lists to allow computing mixed partials
    int N = lattice->getNumberOfParticles();
    energyDensity.resize(N);
    energyDensityReduction.resize(N);
    };

void landauDeGennesLC2D::setNumberOfConstants(distortionEnergyType2D _type)
    {
    numberOfConstants = _type;
    //if(numberOfConstants == distortionEnergyType::multiConstant)
//        printf("\n\n ***WARNING*** \nSome users have reported that the expressions used in multi-constant expressions for the distortion free energy forces may have an error in them. We are currently investigating\n***WARNING***\n\n");
//      DMS, Feb 22, 2021: I believe I have resolved the error in the lcForces.h file that gave rise to the problems with the multi-constant expressions
    };

void landauDeGennesLC2D::computeForces(GPUArray<dVec> &forces,bool zeroOutForce, int type)
    {
    if(useGPU)
        computeForceGPU(forces,zeroOutForce);
    else
        computeForceCPU(forces,zeroOutForce,type);
    

    //In 2D we are actually using an orthogonal basis, so no need to correctForceFromMetric
    //correctForceFromMetric(forces);
    }

void landauDeGennesLC2D::computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
//        UNWRITTENCODE("All GPU code unwritten so far");

    int N = lattice->getNumberOfParticles();
    ArrayHandle<dVec> d_force(forces,access_location::device,access_mode::readwrite);
    ArrayHandle<dVec> d_spins(lattice->returnPositions(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeTypes(lattice->returnTypes(),access_location::device,access_mode::read);
    ArrayHandle<int>  d_latticeNeighbors(lattice->neighboringSites,access_location::device,access_mode::read);
    forceTuner->begin();
    switch (numberOfConstants)
        {
        case distortionEnergyType2D::oneConstant :
            {
            gpu_2DqTensor_oneConstantForce(d_force.data, d_spins.data, d_latticeTypes.data, d_latticeNeighbors.data,
                                         lattice->neighborIndex, A, C, L1, N,zeroOutForce,forceTuner->getParameter());
            break;
            };
        case distortionEnergyType2D::multiConstant:
            {
            UNWRITTENCODE("All GPU code unwritten so far");

            /*
            bool zeroForce = zeroOutForce;
            computeFirstDerivatives();
            ArrayHandle<cubicLatticeDerivativeVector> d_derivatives(forceCalculationAssist,access_location::device,access_mode::read);
            gpu_qTensor_multiConstantForce(d_force.data, d_spins.data, d_latticeTypes.data, d_derivatives.data,
                                        d_latticeNeighbors.data, lattice->neighborIndex,
                                         A,C,L1,L2,L3,L4,L6,N,
                                         zeroOutForce,forceTuner->getParameter());
            */
            break;
            
            };
        };
    forceTuner->end();
    /*
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
    */
    };

void landauDeGennesLC2D::computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce, int type)
    {
    switch (numberOfConstants)
        {
        case distortionEnergyType2D::oneConstant :
            {
            if(type ==0)
                computeL1Bulk2DCPU(forces,zeroOutForce);
            if(type ==1)
                {
                UNWRITTENCODE("Boundary stuff not written yet");
                //computeL1BoundaryCPU(forces,zeroOutForce);
                }
            break;
            };
        case distortionEnergyType2D::multiConstant :
            {
            //multi-component stuff not written yet; need to also do squarelatticederivative vectors?
            UNWRITTENCODE("multi-constant stuff not written yet");
            /*
            bool zeroForce = zeroOutForce;
            computeFirstDerivatives();
            if(type ==0)
                computeAllDistortionTermsBulkCPU(forces,zeroForce);
            if(type ==1)
                {
                UNWRITTENCODE("Boundary stuff not written yet");
                //computeAllDistortionTermsBoundaryCPU(forces,zeroForce);
                }
            */
            break;
            };
        };
    if(type != 0 )
        {
        /*
        Boundary logic not written yet
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
        */
        };
    };
void landauDeGennesLC2D::computeFirstDerivatives()
    {
    UNWRITTENCODE("first derivatives");
    };

void landauDeGennesLC2D::computeBoundaryForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    UNWRITTENCODE("All boundary stuff on cpu");
    }

void landauDeGennesLC2D::computeBoundaryForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce)
    {
    UNWRITTENCODE("All boundary stuff on gpu");
    }

void landauDeGennesLC2D::computeEnergyGPU(bool verbose)
    {
    UNWRITTENCODE("computeEnergy GPU not done yet");
    };

void landauDeGennesLC2D::computeEnergyCPU(bool verbose)
    {
    UNWRITTENCODE("computeEnergy CPU not done yet");
    };
