#ifndef SIMPLEMODEL_H
#define SIMPLEMODEL_H

#include "std_include.h"
#include "gpuarray.h"
#include "functions.h"
/*! \file simple2DModel.h
 * \brief defines an interface for models that compute forces
 */

//! A base interfacing class that defines common operations
/*!
This provides an interface, guaranteeing that SimpleModel S will provide access to
S.setGPU();
S.getNumberOfParticles();
S.computeForces();
S.moveDegreesOfFreedom();
S.returnForces();
S.returnPositions();
S.returnVelocities();
S.returnMasses();
S.spatialSorting();
S.returnAdditionalData();
*/
class simple2DModel
    {
    public:
        //!The base constructor requires the number of particles
        simple2DModel(int n, bool _useGPU = false);
        //!initialize the size of the basic data structure arrays
        void initializeSimple2DModel(int n);

        //!Enforce GPU operation
        virtual void setGPU(bool _useGPU){useGPU = _useGPU;};
        //!get the number of degrees of freedom, defaulting to the number of cells
        virtual int getNumberOfParticles(){return N;};
        //!move the degrees of freedom
        virtual void moveDegreesOfFreedom(GPUArray<dVec> &displacements,scalar scale = 1.) = 0;
        //!do everything necessary to compute forces in the current model
        virtual void computeForces() = 0;
        //!do everything necessary to perform a Hilbert sort
        virtual void spatialSorting(){};

        //!return a reference to the GPUArray of positions
        virtual GPUArray<dVec> & returnPositions(){return positions;};
        //!return a reference to the GPUArray of the current forces
        virtual GPUArray<dVec> & returnForces(){return forces;};
        //!return a reference to the GPUArray of the masses
        virtual GPUArray<scalar> & returnMasses(){return masses;};
        //!return a reference to the GPUArray of the current velocities
        virtual GPUArray<dVec> & returnVelocities(){return velocities;};

        //void setBox(BoxPtr Box);

    protected:
        //!The number of particles
        int N;
        //!particle  positions
        GPUArray<dVec> positions;
        //!particle velocities
        GPUArray<dVec> velocities;
        //!Forces on particles
        GPUArray<dVec> forces;
        //!particle masses
        GPUArray<scalar> masses;

        //!The space in which the particles live
//        BoxPtr Box;
        

        //!Whether the GPU should be used to compute anything
        bool useGPU;

    };
#endif
