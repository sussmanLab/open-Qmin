#ifndef ACTIVEBERISEDWARDS_H
#define ACTIVEBERISEDWARDS_H

#include "equationOfMotion.h"
#include "activeQTensorModel2D.h"
/*! \file activeBerisEdwards2D.h */

class activeBerisEdwards2D : public equationOfMotion
    {
    public:
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
        void relaxPressureCPU();
        void updateQFieldCPU();
        void updateVelocityFieldCPU();

        //!helper function for upwind advective derivatives
        dVec upwindAdvectiveDerivative(dVec &u, dVec &f, dVec &fxd, dVec &fyd, dVec &fxu, dVec &fyu, dVec &fxdd, dVec &fydd, dVec &fxuu, dVec &fyuu);

        //!Auxilliary data structures
        GPUArray<dVec> generalizedAdvection;
        GPUArray<dVec> velocityUpdate;

        //!model parameters
        scalar lambda = 0.1;
        scalar zeta = 400;
        scalar activeLengthScale = 2;
        scalar rotationalViscosity = 2560.;
        scalar ReynoldsNumber = 0.1;
        scalar viscosity = 655360.;
        scalar rho = 1.0;
    };
#endif
