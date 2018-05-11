#ifndef harmonicRepulsion_H
#define harmonicRepulsion_H

#include "basePairwiseForce.h"
/*! \file harmonicRepulsion.h"*/
/*!
Computes harmonic overlap repulsions. i.e., of the form
U(\delta) = 0.5*k*\delta^2, where
\delta = 1-r/\sigma_0, and
\sigma_0 = (r_i+r_j)
*/
class harmonicRepulsion : public basePairwiseForce
    {
        public:
            harmonicRepulsion(){pairParameters.resize(2);monodisperse = false;};
        virtual void computePairwiseForce(dVec &relativeDistance, scalar distance,vector<scalar> &parameters, dVec &f);

        virtual void getParametersForParticlePair(int index1, int index2, vector<scalar> &parameters);

        virtual void setForceParameters(vector<scalar> &params);

        virtual void computeForceGPU(GPUArray<dVec> &forces, bool zeroOutForce);

        virtual void allPairsForceGPU(GPUArray<dVec> &forces, bool zeroOutForce);

        void setMonodisperse(){monodisperse = true;};

        bool monodisperse;
    };

#endif
