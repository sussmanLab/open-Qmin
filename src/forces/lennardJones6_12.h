#ifndef lennardJones6_12_H
#define lennardJones6_12_H

#include "basePairwiseForce.h"
/*! \file lennardJones6_12.h"*/
/*!
Computes the 6-12 lj interaction, i.e. of the form
U(r) = 
Computes harmonic overlap repulsions. i.e., of the form
U(\delta) = 4*epsilon *((simga/r)^12-(sigma/r)^6), where
epsilon = sets the energy scale
\sigma_0 = (r_i+r_j) (or possibly non-additive)

a flag can be set to compute the shift-and-truncated version
by default, the maximum range is assumed to be 2.5
*/
class lennardJones6_12 : public basePairwiseForce
    {
        public:
            lennardJones6_12(){pairParameters.resize(2);};
/*
           virtual void computePairwiseForce(dVec &relativeDistance, scalar distance,vector<scalar> &parameters, dVec &f);

            virtual void getParametersForParticlePair(int index1, int index2, vector<scalar> &parameters);

            virtual void setForceParameters(vector<scalar> &params);

            virtual void computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce);

            virtual void allPairsForceGPU(GPUArray<dVec> &forces, bool zeroOutForce);
*/
            bool shiftAndCut;
            scalar rc;
    };

#endif

