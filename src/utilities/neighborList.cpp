#include "neighborList.h"
//#include "neighborList.h"
/*! \file neighborList.cpp */

neighborList::neighborList(scalar range, BoxPtr _box)
    {
    useGPU = false;
    Box = _box;
    cellList = make_shared<hyperrectangularCellList>(range,Box);
    };

/*!
\param points the set of points to find neighbors for
 */
void neighborList::computeCPU(GPUArray<dVec> &points)
    {
    };

/*!
\param points the set of points to find neighbors for
 */
void neighborList::computeGPU(GPUArray<dVec> &points)
    {
    };

