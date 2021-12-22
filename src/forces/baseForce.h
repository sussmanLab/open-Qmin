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

        virtual string reportSelfName(){string ans = "unnamed"; return ans;};

        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true, int type = 0);

        //!some generic function to set parameters
        virtual void setForceParameters(vector<scalar> &params);

        //! A pointer to the governing simulation
        SimPtr sim;
        //!set the simulation
        void setSimulation(shared_ptr<basicSimulation> _sim){sim=_sim;};

        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<simpleModel> _model){model=_model;};

        //!compute the energy associated with this force
        virtual scalar computeEnergy(bool verbose = false){return 0.;};

        //! compute the system-averaged pressure tensor; return identity if the force hasn't defined this yet
        virtual MatrixDxD computePressureTensor(){MatrixDxD temp; return temp;};

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
        //!whether the updater never does work on the GPU
        bool neverGPU;

        //!Forces might update the total energy associated with them
        scalar energy;
        //!on the gpu, this is per particle and then a reduction can be called
        GPUArray<scalar> energyPerParticle;
        //!does the force get an assist from a neighbor list?
        bool useNeighborList;

        //!a pointer to a neighbor list the force might use
        shared_ptr<neighborList> neighbors;

        //!tell the force to use a neighbor list
        void setNeighborList(shared_ptr<neighborList> _neighbor){neighbors = _neighbor;useNeighborList = true;};
        //!allow for setting multiple threads
        virtual void setNThreads(int n){nThreads = n;};
        //!number of threads to use if compiled with openmp
        int nThreads=1;

        virtual scalar getClassSize()
            {
            return 0.000000001*(2*sizeof(bool) + (1+energyPerParticle.getNumElements())*sizeof(scalar));
            }
    };

typedef shared_ptr<force> ForcePtr;
typedef weak_ptr<force> WeakForcePtr;
#endif
