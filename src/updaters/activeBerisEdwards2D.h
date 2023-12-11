#ifndef ACTIVEBERISEDWARDS_H
#define ACTIVEBERISEDWARDS_H

#include "equationOfMotion.h"
#include "activeQTensorModel2D.h"
/*! \file activeBerisEdwards2D.h */

class activeBerisEdwards2D : public equationOfMotion
    {
    public:
        activeBerisEdwards2D(scalar _K, scalar _gamma, scalar _lambda, scalar _Re, scalar _activeLengthScale, scalar _dt, scalar pdt,scalar _dpTarget);

        virtual void integrateEOMGPU();
        virtual void integrateEOMCPU();

        virtual void initializeFromModel();

        //! A pointer to the active model that contains the extra data structures needed 
        shared_ptr<activeQTensorModel2D> activeModel;
        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<simpleModel> _model)
            {
            model=_model;
            activeModel = dynamic_pointer_cast<activeQTensorModel2D>(model);
            initializeFromModel();
            };
    protected:
        void calculateMolecularFieldAdvectionStressCPU();
        void pressurePoissonCPU();
        void updateQFieldCPU();
        void updateVelocityFieldCPU();

        //!helper function for upwind advective derivatives
        dVec upwindAdvectiveDerivative(dVec &u, dVec &f, dVec &fxd, dVec &fyd, dVec &fxu, dVec &fyu, dVec &fxdd, dVec &fydd, dVec &fxuu, dVec &fyuu);

        //!helper function for the pressure-poisson method
        double3 relaxPressureCPU();

        //!Auxilliary data structures
        GPUArray<dVec> generalizedAdvection;
        GPUArray<dVec> velocityUpdate;
        GPUArray<scalar> auxiliaryPressure;
        GPUArray<scalar> pressurePoissonHelper;

        //!model parameters
        scalar lambda = 0.1;
        scalar zeta = 400;
        scalar activeLengthScale = 2;
        scalar rotationalViscosity = 2560.;
        scalar ReynoldsNumber = 0.1;
        scalar viscosity = 655360.;
        scalar rho = 1.0; //should be one by choice of units


        scalar pseudotimestep = 0.0002;
        scalar targetRelativePressureChange = 0.0001;//better to eventually switch to a different condition for convergence of pressure-poisson method
        int pIterations;
        int maxPIterations = 1000000;
    };
#endif
