#ifndef periodicBoundaryConditions_h
#define periodicBoundaryConditions_h

#include "std_include.h"

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file periodicBoundaryConditions.h */
//!A simple box defining a hypercubic periodic domain
/*!
 * gives access to
Box.setBoxDimensions(vector<scalar> dimensions);
Box.returnBoxDimensions(); //return a vector with box dimensions laid out in some way
Box.movePoint(dvec &A, dvec disp); // A = putInBox(A +disp). i.e., move the particle by some amound and if necessary, e.g., wrap it back into the primary unit cell
Box.minDist(dvec &A, dvec &B, dvec &result); // stores in "result" the minimum distance between A and B
*/
struct periodicBoundaryConditions
    {
    public:
        HOSTDEVICE periodicBoundaryConditions(){};
        //!Construct a hyper-cubic box with side-lengths given by the specified value
        HOSTDEVICE periodicBoundaryConditions(scalar sideLength){setHyperCubic(sideLength);};

        //!Get the dimensions of the box
        HOSTDEVICE void getBoxDims(dVec &boxDims)
            {for (int dd =0; dd < DIMENSION; ++dd) boxDims.x[dd] = boxDimensions.x[dd];};
        //!Get the inverse of the box transformation matrix
        HOSTDEVICE void getBoxInvDims(dVec &iBoxDims)
            {for (int dd =0; dd < DIMENSION; ++dd) iBoxDims.x[dd] = inverseBoxDimensions.x[dd];};

        //!Set the box to some new hypercube
        HOSTDEVICE void setHyperCubic(scalar sideLength);
        //!Set the box to some new rectangular specification
        HOSTDEVICE void setBoxDims(dVec &bDims);

        //! Take a point in the unit square and find its position in the box
        HOSTDEVICE void Transform(const dVec &p1, dVec &pans);
        //! Take a point in the box and find its position in the unit square
        HOSTDEVICE void invTransform(const dVec p1, dVec &pans);
        //!Take the point and put it back in the unit cell
        HOSTDEVICE void putInBoxReal(dVec &p1);
        //!Calculate the minimum distance between two points
        HOSTDEVICE void minDist(const dVec &p1, const dVec &p2, dVec &pans);
        //!Move p1 by the amount disp, then put it in the box
        HOSTDEVICE void move(dVec &p1, const dVec &disp);

        //!compute volume
        HOSTDEVICE scalar Volume()
            {
            scalar vol = 1.0;
            for (int dd = 0; dd < DIMENSION; ++dd)
                vol *= boxDimensions.x[dd];
            return vol;
            };
/*
        HOSTDEVICE void operator=(periodicBoundaryConditions &other)
            {
            Dscalar b11,b12,b21,b22;
            other.getBoxDims(b11,b12,b21,b22);
            setGeneral(b11,b12,b21,b22);
            };
*/
    protected:
        dVec boxDimensions;
        dVec halfBoxDimensions;
        dVec inverseBoxDimensions;

        HOSTDEVICE void putInBox(dVec &vp);
    };

void periodicBoundaryConditions::setHyperCubic(scalar sideLength)
    {
    for (int dd = 0; dd < DIMENSION; ++dd)
        {
        boxDimensions.x[dd] = sideLength;
        halfBoxDimensions.x[dd] = 0.5*sideLength;
        inverseBoxDimensions.x[dd] = 1.0/sideLength;
        };
    };

void periodicBoundaryConditions::setBoxDims(dVec &bDims)
    {
    for (int dd = 0; dd < DIMENSION; ++dd)
        {
        boxDimensions.x[dd] = bDims.x[dd];;
        halfBoxDimensions.x[dd] = 0.5*bDims.x[dd];
        inverseBoxDimensions.x[dd] = 1.0/bDims.x[dd];
        };
    };

void periodicBoundaryConditions::Transform(const dVec &p1, dVec &pans)
    {
    for (int dd = 0; dd < DIMENSION; ++dd)
        pans.x[dd] = boxDimensions.x[dd]*p1.x[dd];
    };

void periodicBoundaryConditions::invTransform(const dVec p1, dVec &pans)
    {
    for (int dd = 0; dd < DIMENSION; ++dd)
        pans.x[dd] = inverseBoxDimensions.x[dd]*p1.x[dd];
    };

void periodicBoundaryConditions::putInBox(dVec &vp)
    {//acts on points in the virtual space
    for (int dd = 0; dd< DIMENSION; ++dd)
        {
        while(vp.x[dd] < 0) vp.x[dd] += 1.0;
        while(vp.x[dd] >= 1.0) vp.x[dd] -= 1.0;
        };
    };

void periodicBoundaryConditions::putInBoxReal(dVec &p1)
    {//assume real space entries. Puts it back in box
    dVec virtualPosition;
    invTransform(p1,virtualPosition);
    putInBox(virtualPosition);
    Transform(virtualPosition,p1);
    };

void periodicBoundaryConditions::minDist(const dVec &p1, const dVec &p2, dVec &pans)
    {
    for (int dd = 0; dd< DIMENSION; ++dd)
        {
        pans.x[dd] = p1.x[dd]-p2.x[dd];
        while(pans.x[dd] < halfBoxDimensions.x[dd]) pans.x[dd] += boxDimensions.x[dd];
        while(pans.x[dd] > halfBoxDimensions.x[dd]) pans.x[dd] -= boxDimensions.x[dd];
        };
    };


void periodicBoundaryConditions::move(dVec &p1, const dVec &disp)
    {//assume real space entries. Moves p1 by disp, and puts it back in box
    for (int dd = 0; dd < DIMENSION; ++dd)
        p1.x[dd] += disp.x[dd];
    putInBoxReal(p1);
    };

typedef shared_ptr<periodicBoundaryConditions> BoxPtr;

#undef HOSTDEVICE
#endif
