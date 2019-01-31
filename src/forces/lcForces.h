#ifndef lcForces_H
#define lcForces_H
#include "std_include.h"
/*! \file lcForces.h */

namespace lcForce
    {
    inline void bulkOneConstantForce(const scalar L1, const dVec &qCurrent,
                const dVec &xDown, const dVec &xUp, const dVec &yDown, const dVec &yUp, const dVec &zDown, const dVec &zUp,
                dVec &spatialTerm)
        {
        spatialTerm = L1*(6.0*qCurrent-xDown-xUp-yDown-yUp-zDown-zUp);
        scalar AxxAyy = spatialTerm[0]+spatialTerm[3];
        spatialTerm[0] += AxxAyy;
        spatialTerm[1] *= 2.0;
        spatialTerm[2] *= 2.0;
        spatialTerm[3] += AxxAyy;
        spatialTerm[4] *= 2.0;
        };

    inline void boundaryOneConstantForce(const scalar L1, const dVec &qCurrent,
                const dVec &xDown, const dVec &xUp, const dVec &yDown, const dVec &yUp, const dVec &zDown, const dVec &zUp,
                const int &ixd,const int &ixu,const int &iyd,const int &iyu,const int &izd,const int &izu,
                dVec &spatialTerm)
        {
        if(ixd <= 0 && ixu <= 0 && iyd <= 0 && iyu <= 0 && izd <= 0 && izu <=0)
            {
            bulkOneConstantForce(L1,qCurrent,xDown,xUp,yDown,yUp,zDown,zUp,spatialTerm);
            }
        else
            {
            if(ixd >0)//xDown is a boundary
                spatialTerm -= (xUp-qCurrent);
            if(ixu >0)//xUp is a boundary
                spatialTerm -= (xDown-qCurrent);//negative derivative and negative nu_x cancel
            if(iyd >0)//ydown is a boundary
                spatialTerm -= (yUp-qCurrent);
            if(iyu >0)
                spatialTerm -= (yDown-qCurrent);//negative derivative and negative nu_y cancel
            if(izd >0)//zDown is boundary
                spatialTerm -= (zUp-qCurrent);
            if(izu >0)
                spatialTerm -= (zDown-qCurrent);//negative derivative and negative nu_z cancel
            spatialTerm = spatialTerm*L1;
            scalar crossTerm = spatialTerm[0]+spatialTerm[3];
            spatialTerm[0] += crossTerm;
            spatialTerm[1] *= 2.0;
            spatialTerm[2] *= 2.0;
            spatialTerm[3] += crossTerm;
            spatialTerm[4] *= 2.0;
            };
        };
    }

#endif
