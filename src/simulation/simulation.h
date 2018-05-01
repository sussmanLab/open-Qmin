#ifndef SIMULATION_H
#define SIMULATION_H

#include "simpleModel.h"
#include "baseUpdater.h"
#include "baseForce.h"
#include "periodicBoundaryConditions.h"

/*! \file simulation.h */

//! A class that ties together all the parts of a simulation
/*!
Simulation objects should have a configuration set, and then at least one updater (such as an equation of motion). In addition to
being a centralized object controlling the progression of a simulation of cell models, the Simulation
class provides some interfaces to cell configuration and updater parameter setters.
*/
class Simulation : public enable_shared_from_this<Simulation>
    {
    public:
        //!Initialize all the shared pointers, etc.
        Simulation();
        //!Pass in a reference to the configuration
        void setConfiguration(ConfigPtr _config);

        //!Call the force computer to compute the forces
        void computeForces();
        //!Call the configuration to move particles around
        void moveParticles(GPUArray<dVec> &displacements);
        //!Call every updater to advance one time step
        void performTimestep();

        //!return a shared pointer to this Simulation
        shared_ptr<Simulation> getPointer(){ return shared_from_this();};
        //!The configuration of particles
        WeakConfigPtr configuration;
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
        void addForce(ForcePtr _force, ConfigPtr _config);

        //!Clear out the vector of forceComputes
        void clearForceComputers(){forceComputers.clear();};
        //!Clear out the vector of updaters
        void clearUpdaters(){updaters.clear();};

        //!The domain of the simulation
        BoxPtr Box;
        //!This changes the contents of the Box pointed to by Box to match that of _box
        void setBox(BoxPtr _box);
/*
        //!A neighbor list assisting the simulation
        cellListGPU *cellList;;
        //!Pass in a reference to the box
        void setCellList(cellListGPU &_cl){cellList = &_cl;};
*/

        //!Set the simulation timestep
        void setIntegrationTimestep(scalar dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        //!Enforce reproducible dynamics
        void setReproducible(bool reproducible);

        //!Set the time between spatial sorting operations.
        void setSortPeriod(int sp){sortPeriod = sp;};

        //!reset the simulation clock
        virtual void setCurrentTime(scalar _cTime);
        //!reset the simulation clock counter
        virtual void setCurrentTimestep(int _cTime){integerTimestep =_cTime;};
        //! An integer that keeps track of how often performTimestep has been called
        int integerTimestep;
        //!The current simulation time
        scalar Time;
        //! The dt of a time step
        scalar integrationTimestep;
        //! A flag controlling whether to use the GPU
        bool useGPU;

    protected:
        //! Determines how frequently the spatial sorter be called...once per sortPeriod Timesteps. When sortPeriod < 0 no sorting occurs
        int sortPeriod;
        //!A flag that determins if a spatial sorting is due to occur this Timestep
        bool spatialSortThisStep;

    };
typedef shared_ptr<Simulation> SimulationPtr;
#endif
