#ifndef harmonicRepulsion_H
#define harmonicRepulsion_H

#include "basePairwiseForce.h"
/*! \file harmonicRepulsion.h"*/

class harmonicRepulsion : public basePairwiseForce
    {
        virtual void computePairwiseForce(dVec &relativeDistance, vector<scalar> &parameters, dVec &f);

        virtual void getParametersForParticlePair(int index1, int index2, vector<scalar> &parameters);
    };

#endif
