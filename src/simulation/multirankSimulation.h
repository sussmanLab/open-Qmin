#ifndef multirankSIMULATION_H
#define multirankSIMULATION_H

#include "profiler.h"
#include "periodicBoundaryConditions.h"
#include "basicSimulation.h"
#include "baseUpdater.h"
#include "baseForce.h"
#include "multirankQTensorLatticeModel.h"
#include "latticeBoundaries.h"
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

        //A section dedicated to various boundary objects. For convenience, these are implemented in a separate (mmultrankSimulationBoundaries) cpp file
        //!transfer buffers and make sure sites on the skin of each rank have correct type
        void finalizeObjects();
        //!the flexible base function...given lattice sites composing an object, determine which are on this rank and add the object
        void createMultirankBoundaryObject(vector<int3> &latticeSites, vector<dVec> &qTensors, boundaryType _type, scalar Param1, scalar Param2);
        //!make a wall with x, y, or z normal
        void createWall(int xyz, int plane, boundaryObject &bObj);
        //!make a simple sphere, setting all points within radius of center to be the object
        void createSphericalColloid(scalar3 center, scalar radius, boundaryObject &bObj);
        //!make a simple sphere, setting all points farther than radius of center to be the object
        void createSphericalCavity(scalar3 center, scalar radius, boundaryObject &bObj);
        //! make a cylindrical object, with either the inside or outside defined as the object
        void createCylindricalObject(scalar3 cylinderStart, scalar3 cylinderEnd, scalar radius, bool colloidOrCapillary, boundaryObject &bObj);
        //!make a spherocylinder, defined by the start and end of the cylindrical section and the radius
        void createSpherocylinder(scalar3 cylinderStart, scalar3 cylinderEnd, scalar radius, boundaryObject &bObj);

        //!import a boundary object from a (carefully prepared) text file
        virtual void createBoundaryFromFile(string fname, bool verbose = false);

        //!a function of convenience... make a dipolar field a la Lubensky et al.
        void setDipolarField(scalar3 center, scalar ThetaD, scalar radius,scalar range, scalar S0);
        //!a function of convenience... make a dipolar field a la Lubensky et al. Simpler, uses the ravnik expression
        void setDipolarField(scalar3 center, scalar3 direction, scalar radius,scalar range, scalar S0);

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
            auto upd = updaters[0].lock();
            return upd->getMaxForce();
            };
        //!Call every updater to advance one time step
        void performTimestep();

        //! manipulate data from updaters
        virtual void sumUpdaterData(vector<scalar> &data);

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

        //!save a file for each rank recording the expanded lattice; lattice skip controls the sparsity of saved sites
        void saveState(string fname, int latticeSkip = 1, int defectType = 0);

        //!load the Q-tensor values for each lattice site from a specified file. DOES NOT load any logic about the nature of various sites (boundary, etc)
        void loadState(string fname);

        //!in multi-rank simulations, this stores the lowest (x,y,z) coordinate controlled by the current rank
        int3 latticeMinPosition;

        virtual void reportSelf(){cout << "in the multirank simulation class" << endl;};

        //! the local {x,y,z} rank coordinate...currently shuffled to basicSimulation.h
        int3 rankParity;//even is even, odd is odd...makes sense

    protected:
        void setRankTopology(int x, int y, int z);

        void determineCommunicationPattern(bool _edges, bool _corners);

        vector<int2> communicationDirections;
        vector<bool> communicationDirectionParity;
        vector<int> communicationTargets;


        //!the number of ranks per {x,y,z} axis
        int3 rankTopology;
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
        vector<scalar> dataBuffer;
    };
#endif
