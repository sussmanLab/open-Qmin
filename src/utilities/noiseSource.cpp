#include "noiseSource.h"

/*! \file noiseSource.cpp */

int noiseSource::getInt(int minimum, int maximum)
    {
    int answer;
    uniform_int_distribution<int> uniIntRand(minimum,maximum);
    if (Reproducible)
        answer = uniIntRand(gen);
    else
        answer = uniIntRand(genrd);
    return answer;
    };

Dscalar noiseSource::getRealUniform(Dscalar minimum, Dscalar maximum)
    {
    Dscalar answer;
    uniform_real_distribution<Dscalar> uniRealRand(minimum,maximum);
    if (Reproducible)
        answer = uniRealRand(gen);
    else
        answer = uniRealRand(genrd);
    return answer;
    };
Dscalar noiseSource::getRealNormal(Dscalar mean, Dscalar std)
    {
    Dscalar answer;
    normal_distribution<> normal(mean,std);
    if (Reproducible)
        answer = normal(gen);
    else
        answer = normal(genrd);
    return answer;
    };

void noiseSource::setReproducibleSeed(int _seed)
    {
    RNGSeed = _seed;
    mt19937 Gener(13377);
    gen = Gener;
#ifdef DEBUGFLAGUP
    mt19937 GenerRd(13377);
    genrd=GenerRd;
#endif
    };
