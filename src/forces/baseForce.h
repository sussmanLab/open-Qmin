#ifndef baseForce_H
#define baseForce_H

#include "std_include.h"
#include "simpleModel.h"
#include "basicSimulation.h"
#include "neighborList.h"

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
        shared_ptr<basicSimulation> sim;
        //!set the simulation
        void setSimulation(shared_ptr<basicSimulation> _sim){sim=_sim;};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<simpleModel> _model){model=_model;};

        //! A pointer to a simpleModel that the updater acts on
        shared_ptr<simpleModel> model;
        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU=true)
            {
            useGPU = _useGPU;
            if(useNeighborList)
                    neighbors->setGPU(_useGPU);
            };
        //!whether the updater does its work on the GPU or not
        bool useGPU;

        //!Forces might update the total energy associated with them
        scalar energy;
        //!does the force get an assist from a neighbor list?
        bool useNeighborList;

        //!a pointer to a neighbor list the force might use
        shared_ptr<neighborList> neighbors;

        //!tell the force to use a neighbor list
        void setNeighborList(shared_ptr<neighborList> _neighbor){neighbors = _neighbor;useNeighborList = true;};
    };

typedef shared_ptr<force> ForcePtr;
typedef weak_ptr<force> WeakForcePtr;
#endif
