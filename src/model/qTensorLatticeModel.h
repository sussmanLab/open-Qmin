#ifndef qTensorLatticeModel_H
#define qTensorLatticeModel_H

#include "cubicLattice.h"
#include "qTensorFunctions.h"
/*! \file qTensorLatticeModel.h */

//! Each site on the underlying lattice gets a local Q-tensor
/*!
The Q-tensor has five independent components, which will get passed around in dVec structures...
a dVec of q[0,1,2,3,4] corresponds to the symmetric traceless tensor laid out as
    (q[0]    q[1]        q[2]    )
Q = (q[1]    q[3]        q[4]    )
    (q[2]    q[4]   -(q[0]+q[3]) )
 */
class qTensorLatticeModel : public cubicLattice
    {
    public:
        //! construct an underlying cubic lattice
        qTensorLatticeModel(int l,bool _useGPU = false);

        //!(possibly) need to rewrite how the Q tensors update with respect to a displacement call
        virtual void moveParticles(GPUArray<dVec> &displacements, scalar scale = 1.);

        //!initialize each d.o.f., also passing in the value of the nematicity
        void setNematicQTensorRandomly(noiseSource &noise, scalar s0);

        //!get field-averaged eigenvalues
        void getAverageEigenvalues()
            {
            scalar a,b,c;
            dVec meanQ = averagePosition();
            eigenvaluesOfQ(meanQ,a,b,c);
            printf("eigenvalues: %f\t%f\t%f\n",a,b,c);
            };
    };
#endif
