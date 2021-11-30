#ifndef qTensorLatticeModel_H
#define qTensorLatticeModel_H

#include "cubicLattice.h"
#include "qTensorFunctions.h"
#include "qTensorLatticeModel.cuh"

/*! \file qTensorLatticeModel.h */

//! Each site on the underlying lattice gets a local Q-tensor
/*!
The Q-tensor has five independent components, which will get passed around in dVec structures...
a dVec of q[0,1,2,3,4] corresponds to the symmetric traceless tensor laid out as
    (q[0]    q[1]        q[2]    )
Q = (q[1]    q[3]        q[4]    )
    (q[2]    q[4]   -(q[0]+q[3]) )

Boundaries are implemented by making use of the "type" data structure that is inherited from the base simpleModel
class...: each bulk LC lattice site will have type zero (the default), and lattice sites *adjacent to a boundary* will
have type < 0 (-1 for now, possibly optimized later). A lattice site type[i] > 0 will mean the lattice site is part of
whatever object boundaries[type[i]-1] refers to.

The qTensorLatticeModel implements a "create boundary" method which takes an array of lattice sites, appends a new
boundaryObject to boundaries (so that boundaries[j] now exists), and then sets the type of the lattice sites so that
type[i] = j+1
 */
class qTensorLatticeModel : public cubicLattice
    {
    public:
        //! construct an underlying cubic lattice
        qTensorLatticeModel(int l,bool _useGPU = false, bool _neverGPU=false);
        qTensorLatticeModel(int lx,int ly,int lz,bool _useGPU = false, bool _neverGPU=false);

        //!(possibly) need to rewrite how the Q tensors update with respect to a displacement call
        virtual void moveParticles(GPUArray<dVec> &displacements, scalar scale = 1.);

        //!initialize each d.o.f., also passing in the value of the nematicity
        void setNematicQTensorRandomly(noiseSource &noise, scalar s0,bool globallyAligned = false);

        //!get field-averaged eigenvalues
        void getAverageEigenvalues(bool verbose = true);
        //!get field-averaged eigenvector corresponding to largest eigenvalue
        void getAverageMaximalEigenvector(vector<scalar> &averageN);

        //!import a boundary object from a (carefully prepared) text file
        void createBoundaryFromFile(string fname, bool verbose = false);

        //!create a flat wall at the specified location, on either an x,y, or z plane, with anchoring given by the boundary object
        void createSimpleFlatWallNormal(int plane, int xyz, boundaryObject &bObj);

        //!create a flat wall (with z-normal) at the specified location, with anchoring given by the boundary object
        void createSimpleFlatWallZNormal(int zPlane, boundaryObject &bObj);

        //!create a spherical colloid with anchoring given by the boundary object (with surface normal the director direction)
        void createSimpleSpherialColloid(scalar3 center, scalar radius, boundaryObject &bObj);

        //!compute different measures of whether a site is a defect
        void computeDefectMeasures(int defectType);

        virtual scalar getClassSize()
            {
            return cubicLattice::getClassSize();
            }
    };
#endif
