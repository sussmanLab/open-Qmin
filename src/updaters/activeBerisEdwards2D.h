#ifndef ACTIVEBERISEDWARDS_H
#define ACTIVEBERISEDWARDS_H

#include "equationOfMotion.h"
#include "activeQTensorModel2D.h"
/*! \file activeBerisEdwards2D.h */

/*!
This class implements active Beris-Edwards equations for a 2D incompressible active Q-tensor flowing fluid.
It functions via a pressure-poisson method for imposing the incompressibility (which is a potential target of improvement).

In brief, the Q tensor evolves according to 
\partial_t Q_{ij} = \gamma^{-1} H_{ij} + S_{ij} - u_k\partial_k Q_{ij},
i.e., it relaxes according to the molecular field (H = -\frac{\delta F_{LdG}}{\delta Q}), it is advected by the fluid flow, and experiences a generalized advection term
S_{ij} = 0.5*\lambda*S*(\partial_i u_j + \partial_j u_i) + Q_{ik}\omega_{kj} - \omega_{ik}Q_{kj},
where S is the degree of local order and \omega_{xy} = 0.5*(\partial_x u_y-\partial_y u_x)

At the same time, the velocity is evolved according to
\partial_t \vec{u} = -(\vec{u}\cdot\nabla)\vec{u} + (viscosity)*\nabla^2\vec{u} + (1/rho)*(\vec{F} - \nabla p),
where
F_i = \partial_j \Pi_{ij},
for a stress tensors \Pi composed of both elastic and active components. We can usefully rearrange these contributions into a part of the stress tensor which is symmetric and that which is anti-symmetric:
\Pi^{sym} = -\lambda H - \zeta Q,
\Pi^{antisym} = 2*(Q_{xx}H_{xy} - H_{xx}Q_{xy}),
where in two dimensions the symmetric part has two independent components, and the antisymmetric part has only one.

See, e.g., Luca Giomi, PRX 2015 (https://journals.aps.org/prx/pdf/10.1103/PhysRevX.5.031003), for more details
*/
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
        scalar viscosity = 809.542;
        scalar rho = 1.0; //should be one by choice of units


        scalar pseudotimestep = 0.0002;
        scalar targetRelativePressureChange = 0.0001;//better to eventually switch to a different condition for convergence of pressure-poisson method
        int pIterations;
        int maxPIterations = 10000;
    };
#endif
