#ifndef qTensorLatticeModel2D_H
#define qTensorLatticeModel2D_H

#include "squareLattice.h"
#include "qTensorFunctions2D.h"
#include "latticeBoundaries.h"

/*! \file qTensorLatticeModel2D.h */

//! Each site on the underlying lattice gets a local Q-tensor
/*!
The Q-tensor in two dimensions has two independent components, which will get passed around in dVec structures...
a dVec of q[0,1] corresponds to the symmetric traceless tensor laid out as
    (q[0]     q[1]  )
Q = (q[1]    -q[0] )

Convention is that, for a director n, the corresponding Q =2*s (n x n - I/2)

Boundaries are implemented by making use of the "type" data structure that is inherited from the base simpleModel
class...: each bulk LC lattice site will have type zero (the default), and lattice sites *adjacent to a boundary* will
have type < 0 (-1 for now, possibly optimized later). A lattice site type[i] > 0 will mean the lattice site is part of
whatever object boundaries[type[i]-1] refers to.

The qTensorLatticeModel2D implements a "create boundary" method which takes an array of lattice sites, appends a new
boundaryObject to boundaries (so that boundaries[j] now exists), and then sets the type of the lattice sites so that
type[i] = j+1
 */
class qTensorLatticeModel2D : public squareLattice
    {
    public:
        //! construct an underlying cubic lattice
        qTensorLatticeModel2D(int l,bool _useGPU = false, bool _neverGPU=false);
        qTensorLatticeModel2D(int lx,int ly,bool _useGPU = false, bool _neverGPU=false);

        //!(possibly) need to rewrite how the Q tensors update with respect to a displacement call
        virtual void moveParticles(GPUArray<dVec> &displacements, scalar scale = 1.);

        //!initialize each d.o.f., also passing in the value of the nematicity
        void setNematicQTensorRandomly(noiseSource &noise, scalar s0,bool globallyAligned = false);

        //!get field-averaged eigenvalues
        void getAverageEigenvalues(bool verbose = true);
        //!get field-averaged eigenvector corresponding to largest eigenvalue
        void getAverageMaximalEigenvector(vector<scalar> &averageN);


        //!assign a collection of lattice sites to a new boundaryObject
        void createBoundaryObject(vector<int> &latticeSites, boundaryType _type, scalar Param1, scalar Param2);

        //!create a flat wall (with z-normal) at the specified location, with anchoring given by the boundary object
        void createSimpleFlatWall(int xyPlane, boundaryObject &bObj);
        //!import a boundary object from a (carefully prepared) text file
        void createBoundaryFromFile(string fname, bool verbose = false);

        //!compute different measures of whether a site is a defect
        void computeDefectMeasures(int defectType);

        virtual scalar getClassSize()
            {
            return squareLattice::getClassSize();
            }

        GPUArray<boundaryObject> boundaries;
        //!A vector that keeps track of the sites associated with each boundary object
        vector<GPUArray<int> > boundarySites;
        //!A vector that keeps track of the surface sites associated with each boundary object
        vector<GPUArray<int> > surfaceSites;
        //!A vector of flags that specifies the state of each boundary object...most schemes will be 0 = fixed boundary, 1 = movable boudnary
    };
#endif
