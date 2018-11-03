#ifndef SIMULATION_H
#define SIMULATION_H

#include "periodicBoundaryConditions.h"
#include "basicSimulation.h"
#include "simpleModel.h"
#include "baseUpdater.h"
#include "baseForce.h"

/*! \file simulation.h */

//! A class that ties together all the parts of a simulation
/*!
Simulation objects should have a configuration set, and then at least one updater (such as an equation of motion). In addition to
being a centralized object controlling the progression of a simulation of cell models, the Simulation
class provides some interfaces to cell configuration and updater parameter setters.
*/
class Simulation : public basicSimulation, public enable_shared_from_this<Simulation>
    {
    public:
        //!Pass in a reference to the configuration
        void setConfiguration(ConfigPtr _config);

        //!Call the force computer to compute the forces
        virtual void computeForces();
        //!Call the configuration to move particles around
        virtual void moveParticles(GPUArray<dVec> &displacements);
        //!Call every updater to advance one time step
        void performTimestep();

        //!return a shared pointer to this Simulation
        shared_ptr<Simulation> getPointer(){ return shared_from_this();};
//
        //! A vector of updaters that the simulation will loop through
        vector<WeakUpdaterPtr> updaters;
        //! A vector of force computes the simulation will loop through
        vector<WeakForcePtr> forceComputers;

        //!Add an updater
        void addUpdater(UpdaterPtr _upd){updaters.push_back(_upd);};
        //!Add an updater with a reference to a configuration
        void addUpdater(UpdaterPtr _upd, ConfigPtr _config);
        //!Add a force computer configuration
        void addForce(ForcePtr _force){forceComputers.push_back(_force);};
        //!Add a force computer configuration
        void addForce(ForcePtr _force, ConfigPtr _config);

        //!Clear out the vector of forceComputes
        void clearForceComputers(){forceComputers.clear();};
        //!Clear out the vector of updaters
        void clearUpdaters(){updaters.clear();};

        //!compute the potential energy associated with all of the forces
        virtual scalar computePotentialEnergy();
        //!compute the kinetic energy
        virtual scalar computeKineticEnergy();
        //!compute the total energy
        virtual scalar computeEnergy()
            {
            return computeKineticEnergy() + computePotentialEnergy();
            };
        //!compute the pressure tensor
        virtual void computePressureTensor(MatrixDxD &P);

        //!Set the simulation timestep
        void setIntegrationTimestep(scalar dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        //!Enforce reproducible dynamics
        void setReproducible(bool reproducible);

        //!work with openMP threads
        void setNThreads(int n);
    };
typedef shared_ptr<Simulation> SimulationPtr;
#endif
