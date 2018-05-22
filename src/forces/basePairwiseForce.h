#ifndef basePairwiseForce_H
#define basePairwiseForce_H

#include "baseForce.h"
#include "indexer.h"
/*! \file basePairwiseForce.h */
/*!
by default pairwise forces will only compute half of the interactions (if for p1 < p2), and assume f_{p1,p2}= - f_{p2,p1}
*/
class basePairwiseForce : public force
    {
    public:
        //!the call to compute forces, and store them in the referenced variable
        virtual void computeForces(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceGPU(GPUArray<dVec> &forces,bool zeroOutForce = true);
        virtual void computeForceCPU(GPUArray<dVec> &forces,bool zeroOutForce = true);

        virtual scalar computeEnergy()
                {
                if (!useGPU)
                    {
                    GPUArray<dVec> f; f.resize(model->getNumberOfParticles());
                    computeForceCPU(f,true);
                    return energy;
                    }
                else
                    return computeEnergyGPU();
                };

        virtual MatrixDxD computePressureTensor();
        virtual scalar computeEnergyGPU(){return 0.0;};

        virtual void computePairwiseForce(dVec &relativeDistance, scalar distance,vector<scalar> &parameters, dVec &f){printf("in base pairwiseForce...why?\n");};

        virtual void getParametersForParticlePair(int index1, int index2, vector<scalar> &parameters){};

        //!a vector of parameters that can be set...
        GPUArray<scalar> parameters;
        //!a small vector that gets passed around when actually calculating the forces
        vector<scalar> pairParameters;
        //! the number of different types the force knows about
        int nTypes;
        //!an indexer for accessing type-based parameters
        Index2D particleTypeIndexer;
    };
#endif
