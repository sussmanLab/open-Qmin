#ifndef baseForce_H
#define baseForce_H

#include "std_include.h"
#include "simpleModel.h"

//forward declare simulation class
class Simulation;

/*! \file baseForce.h */
//!A base class for implementing force calculations
/*!
 *
 *
*/
class force
    {
    public:
        force();

        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true);

        //!some generic function to set parameters
        virtual void setForceParameters(vector<scalar> &params);

        //! A pointer to the governing simulation
        shared_ptr<Simulation> sim;
        //!set the simulation
        void setSimulation(shared_ptr<Simulation> _sim){sim=_sim;};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<simpleModel> _model){model=_model;};

        //! A pointer to a simpleModel that the updater acts on
        shared_ptr<simpleModel> model;
        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU=true){useGPU = _useGPU;};
        //!whether the updater does its work on the GPU or not
        bool useGPU;

    };

typedef shared_ptr<force> ForcePtr;
typedef weak_ptr<force> WeakForcePtr;
#endif
