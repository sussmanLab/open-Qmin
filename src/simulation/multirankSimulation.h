#ifndef multirankSIMULATION_H
#define multirankSIMULATION_H

#include "profiler.h"
#include "periodicBoundaryConditions.h"
#include "basicSimulation.h"
#include "baseUpdater.h"
#include "baseForce.h"
#include "multirankQTensorLatticeModel.h"
#include <mpi.h>

/*! \file multirankSimulation.h */

class multirankSimulation : public basicSimulation, public enable_shared_from_this<multirankSimulation>
    {
    public:
        multirankSimulation(int _myRank,int xDiv, int yDiv, int zDiv, bool _edges, bool _corners)
            {
            myRank = _myRank;
            setRankTopology(xDiv,yDiv,zDiv);
            determineCommunicationPattern(_edges,_corners);
            }
        //!move particles, and also communicate halo sites
        virtual void moveParticles(GPUArray<dVec> &displacements,scalar scale = 1.0);

        //!handles calls to all necessary halo site transfer
        virtual void communicateHaloSitesRoutine();

        //! synchronize mpi and make transfer buffers
        virtual void synchronizeAndTransferBuffers();

        profiler p1 = profiler("total communication time");
        profiler p4 = profiler("MPI recv");
        profiler p3 = profiler("MPI send");
        profiler p2 = profiler("GPU data buffering kernels time");
        //!The configuration of latticeSites
        WeakMConfigPtr mConfiguration;

        //!return a shared pointer to this Simulation
        shared_ptr<multirankSimulation> getPointer(){ return shared_from_this();};
        //!Pass in a reference to the configuration
        void setConfiguration(MConfigPtr _config);
        //! A vector of updaters that the simulation will loop through
        vector<WeakUpdaterPtr> updaters;
        //! A vector of force computes the simulation will loop through
        vector<WeakForcePtr> forceComputers;

        //!Call the force computer to compute the forces
        virtual void computeForces();

        //!Add an updater
        void addUpdater(UpdaterPtr _upd){updaters.push_back(_upd);};
        //!Add an updater with a reference to a configuration
        void addUpdater(UpdaterPtr _upd, MConfigPtr _config);
        //!Add a force computer configuration
        virtual void addForce(ForcePtr _force){forceComputers.push_back(_force);};
        //!Add a force computer configuration
        virtual void addForce(ForcePtr _force, MConfigPtr _config);

        //!Clear out the vector of forceComputes
        void clearForceComputers(){forceComputers.clear();};
        //!Clear out the vector of updaters
        void clearUpdaters(){updaters.clear();};

        //!A utility function that just checks the first updater for a max force
        scalar getMaxForce()
            {
            cout <<"needs to be multiranked " << endl;
            auto upd = updaters[0].lock();
            return upd->getMaxForce();
            };
        //!Call every updater to advance one time step
        void performTimestep();

        //!compute the potential energy associated with all of the forces
        virtual scalar computePotentialEnergy(bool verbose = false);
        //!compute the kinetic energy
        virtual scalar computeKineticEnergy(bool verbose = false);
        //!compute the total energy
        virtual scalar computeEnergy(bool verbose = false)
            {
            return computeKineticEnergy(verbose) + computePotentialEnergy(verbose);
            };

        //!Set the simulation timestep
        void setIntegrationTimestep(scalar dt);
        //!turn on CPU-only mode for all components
        void setCPUOperation(bool setcpu);
        //!Enforce reproducible dynamics
        void setReproducible(bool reproducible);

        //!save a file for each rank recording the expanded lattice
        void saveState(string fname);

    protected:
        void setRankTopology(int x, int y, int z);

        void determineCommunicationPattern(bool _edges, bool _corners);

        vector<int2> communicationDirections;
        vector<bool> communicationDirectionParity;
        vector<int> communicationTargets;

        int myRank;
        int nRanks;
        //!the number of ranks per {x,y,z} axis
        int3 rankTopology;
        //! the local {x,y,z} rank coordinate
        int3 rankParity;//even is even, odd is odd...makes sense
        Index3D parityTest;
        //!do edges need to be communicated?
        bool edges;
        //!do corner sites?
        bool corners;

        //!have the halo sites been communicated?
        bool transfersUpToDate;

        MPI_Status mpiStatus;
        vector<MPI_Status> mpiStatuses;
        vector<MPI_Request> mpiRequests;

    };
#endif
