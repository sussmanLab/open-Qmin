#ifndef landauDeGennesLC_H
#define landauDeGennesLC_H

#include "baseLatticeForce.h"
#include "landauDeGennesLCBoundary.h"
#include "multirankSimulation.h"
/*! \file landauDeGennesLC.h */

enum class distortionEnergyType {oneConstant,multiConstant};

//!A landau-de gennes  q-tensor framework force computer...currently working with the one-constant approximation for the distortion term
class landauDeGennesLC : public baseLatticeForce
    {
    public:

        landauDeGennesLC(bool _neverGPU = false);
        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1);
        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1,scalar _L2, scalar _L3, scalar _L4, scalar _L6);
        landauDeGennesLC(scalar _A, scalar _B, scalar _C, scalar _L1, scalar _L2,scalar _L3orWavenumber, distortionEnergyType _type);

        virtual string reportSelfName(){string ans = "landauDeGennesLC"; return ans;};

        //!set up a few basic things (common force tuners, number of energy components, etc.)
        void baseInitialization();
        //!The model setting creates an additional data structure to help with 2- or 3- constant approximation
        virtual void setModel(shared_ptr<cubicLattice> _model);

        //!use the "type" flag to select either bulk or boundary routines
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true, int type = 0);

        //select the force routing based on the number of elastic constants
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        //when using the Qxx,Qxy,Qxz,Qyy,Qyz basis, correct forces according to the non-orthogonal metric
        virtual void correctForceFromMetric(GPUArray<dVec> &forces);

        void setPhaseConstants(scalar _a=-1, scalar _b =-12.325581395, scalar _c =  10.058139535){A=_a;B=_b;C=_c;};
        void setElasticConstants(scalar _l1=2.32,scalar _l2=0, scalar _l3=0, scalar _l4 = 0, scalar _l6=0){L1=_l1;L2=_l2;L3=_l3; L4=_l4; L6 = _l6;};
        void setNumberOfConstants(distortionEnergyType _type);

        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true, int type = 0);

        //!compute the forces on the objects in the system
        virtual void computeObjectForces(int objectIdx);

        //!Precompute the first derivatives at all of the LC Sites
        virtual void computeFirstDerivatives();

        //!compute the stress tensors at the given set of sites
        virtual void computeStressTensors(GPUArray<int> &sites,GPUArray<Matrix3x3> &stress);

        virtual void computeBoundaryForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce);
        virtual void computeBoundaryForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce);

        virtual void computeEorHFieldForcesCPU(GPUArray<dVec> &forces,bool zeroOutForce,
                                    scalar3 field, scalar anisotropicSusceptibility,scalar vacuumPermeability);

        virtual void computeEorHFieldForcesGPU(GPUArray<dVec> &forces,bool zeroOutForce,
                                    scalar3 field, scalar anisotropicSusceptibility,scalar vacuumPermeability);

        virtual void computeSpatiallyVaryingFieldCPU(GPUArray<dVec> &forces,bool zeroOutForce,
                                    GPUArray<scalar3> field, scalar anisotropicSusceptibility,scalar vacuumPermeability);
        virtual void computeSpatiallyVaryingFieldGPU(GPUArray<dVec> &forces,bool zeroOutForce,
                                    GPUArray<scalar3> field, scalar anisotropicSusceptibility,scalar vacuumPermeability);
        virtual void computeEnergyCPU(bool verbose = false);
        virtual void computeEnergyGPU(bool verbose = false);

        //!A vector storing the components of energy (phase,distortion,anchoring)
        vector<scalar> energyComponents;

        void printTuners()
            {
            printf("forceTuner\n");
            forceTuner->printTimingData();
            if(numberOfConstants != distortionEnergyType::oneConstant)
                {
                printf("forceAssistTuner\n");
                forceAssistTuner->printTimingData();
                };
            if(lattice->boundaries.getNumElements() >0)
                {
                printf("BoundaryForceTuner\n");
                boundaryForceTuner->printTimingData();
                };
            };
        void setEField(scalar3 field, scalar eps, scalar eps0,scalar deltaEps)
            {
            computeEfieldContribution = true;
            Efield = field;
            epsilon =eps;
            epsilon0=eps0;
            deltaEpsilon=deltaEps;
            };
        void setHField(scalar3 field, scalar chi, scalar _mu0,scalar _deltaChi)
            {
            computeHfieldContribution = true;
            Hfield = field;
            Chi =chi;
            mu0=_mu0;
            deltaChi=_deltaChi;
            };

        void setSpatiallyVaryingField(string fname, scalar chi, scalar _mu0, scalar _deltaChi,int3 rankParity);

        //!the free energy density at each lattice site
        GPUArray<scalar> energyDensity;
        //!A helper array for energy reductions
        GPUArray<scalar> energyDensityReduction;
        //!the force from stresses at the surface of an object
        GPUArray<scalar3> objectForceArray;

        virtual scalar getClassSize()
            {
            scalar thisClassSize = sizeof(scalar)*(energyComponents.size() + energyDensity.getNumElements() + 3*objectForceArray.getNumElements()+20) + 3*sizeof(bool) + 4*sizeof(kernelTuner)+ sizeof(cubicLatticeDerivativeVector)*forceCalculationAssist.getNumElements();
            return 0.000000001*thisClassSize + baseLatticeForce::getClassSize();
            }

    protected:
        //!constants, etc.
        scalar A;
        scalar B;
        scalar C;
        scalar L1;
        scalar L2;
        scalar L3;
        scalar L4;
        scalar L6;

        scalar q0;

        scalar3 Efield;
        scalar deltaEpsilon;
        scalar epsilon0;
        scalar epsilon;
        scalar3 Hfield;
        scalar deltaChi;
        scalar Chi;
        scalar mu0;

        GPUArray<scalar3> spatiallyVaryingField;

        //!number of elastic constants
        distortionEnergyType numberOfConstants;
        //!switches for extra parts of the energy/force calculations
        bool computeEfieldContribution;
        bool computeHfieldContribution;
        bool spatiallyVaryingFieldContribution;

        //!for 2- and 3- constant approximations, the force calculation is helped by first pre-computing first derivatives
        GPUArray<cubicLatticeDerivativeVector> forceCalculationAssist;
        /*
        The layout for forceCalculationAssist is the following... forceCalculationAssist[i] is site i, and the
        derivates are laid out like
        {d (dVec[0])/dx, d (dVec[1])/dx, ... , d (dVec[0])/dy, ...d (dVec[0])/dz,...d (dVec[DIMENSION-1])/dz}
        */

        //!performance for the first derivative calculation
        shared_ptr<kernelTuner> forceAssistTuner;
        //!performance for the boundary force kernel
        shared_ptr<kernelTuner> boundaryForceTuner;
        //!performance for the l24 force kernel
        shared_ptr<kernelTuner> l24ForceTuner;
        //!performance for the E/H field force kernel
        shared_ptr<kernelTuner> fieldForceTuner;

        //!Compute L1 distortion terms in the bulk *and* the phase force
        virtual void computeL1BulkCPU(GPUArray<dVec> &forces,bool zeroOutForce);
        //!Compute L1 distortion terms at boundaries *and* the phase force
        virtual void computeL1BoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce);

        //!Compute all distortion terms in the bulk *and* the phase force
        virtual void computeAllDistortionTermsBulkCPU(GPUArray<dVec> &forces,bool zeroOutForce);
        //!Compute all distortion terms at boundaries *and* the phase force
        virtual void computeAllDistortionTermsBoundaryCPU(GPUArray<dVec> &forces,bool zeroOutForce);
    };

#endif
