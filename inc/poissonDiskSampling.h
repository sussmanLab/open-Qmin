#ifndef poissonDiskSampling_H
#define poissonDiskSampling_H
#include "std_include.h"
#include "noiseSource.h"
#include "periodicBoundaryConditions.h"
/*! \file poissonDiskSampling.h */
//! a class that implements the Bridson fast sampling algorithm, in a periodic domain
/*!
Robert Bridson, "Fast Poisson Disk Sampling in Arbitrary Dimensions"
SIGGRAPH sketches (2007)
*/
class poissonDiskSampling
    {
    public:
        poissonDiskSampling(int desiredNumberOfPoints,scalar radius, vector<dVec> &sample,noiseSource &noise, shared_ptr<periodicBoundaryConditions> &Box,int max_sample_attempts = 30);

        int nDimArrayIndex(const iVec &dimensions, const iVec &x);
        int nDimArrayIndex(const iVec &dimensions, const dVec &x);
        void sample_annulus_point(scalar radius, const dVec &center, dVec &answer, noiseSource &noise);
    };

/*!
the consructor also does all of the work
*/
poissonDiskSampling::poissonDiskSampling(int pts,scalar radius, vector<dVec> &sample,noiseSource &noise, shared_ptr<periodicBoundaryConditions> &Box,int max_sample_attempts)
    {
    sample.clear();
    vector<int> activeList;
    dVec max;
    Box->getBoxDims(max);
    dVec min(.5);
    max = max-min;


    //set up the "accelearation grid"... the grid cell size specifies something that can fit one entry
    //and the accel grid tells you if a sample is in a given cell
    scalar gridDx = (0.9999)*radius/sqrt((scalar)DIMENSION);
    iVec dimensions;
    int totalArraySize = 1;
    for (int ii = 0; ii < DIMENSION; ++ii)
        {
        dimensions.x[ii] = ceil((max.x[ii]-min.x[ii])/gridDx);
        totalArraySize *= dimensions.x[ii];
        };
    vector<int> accel(totalArraySize,-1);//value < 0 specifies the cell is empty
    dVec x;
    for (int dd = 0; dd < DIMENSION; ++dd)
        x.x[dd] = noise.getRealUniform(min.x[dd],max.x[dd]);
    sample.push_back(x);
    activeList.push_back(0);
    int k  = nDimArrayIndex(dimensions,(1/gridDx)*(x-min));
    accel[k] = 0;

    //The main loop
    for(int jj = 0; jj <2*pts-1; ++jj)
    {
    if(sample.size() >=pts) break;
    for(int ii = 0; ii < sample.size(); ++ii)
        {
        activeList.push_back(ii);
        dVec xTemp = sample[ii];
        int kTemp = nDimArrayIndex(dimensions,(1/gridDx)*(xTemp-min));
        accel[kTemp] = ii;
        };
    while(!activeList.empty() && sample.size() < pts)
        {
        int r = noise.getInt(0,activeList.size()-0.0001f);
        int cur = activeList[r];
        bool foundSample = false;
        iVec j, jmin,jmax;
        for (int attempt =0; attempt < max_sample_attempts; ++attempt)
            {
            sample_annulus_point(radius,sample[cur],x,noise);
            Box->putInBoxReal(x);
            bool skip = false;
            for (int dd = 0; dd < DIMENSION; ++dd)
                {
                if(x.x[dd] < gridDx*0.5|| x.x[dd] > max.x[dd]-gridDx*0.5)
                skip = true;
                };
            if(skip) continue;
            //test if there is already a close sample to this point
            for (int dd = 0; dd < DIMENSION; ++dd)
                {
                int thisMin = (int)((x.x[dd]-radius-min.x[dd])/gridDx);
                if(thisMin < 0) thisMin = 0;
                if(thisMin >= (int) dimensions.x[dd]) thisMin = dimensions.x[dd]-1;
                jmin.x[dd] = thisMin;
                int thisMax = (int)((x.x[dd]+radius-min.x[dd])/gridDx);
                if(thisMax < 0) thisMax =0;
                if(thisMax >= (int) dimensions.x[dd]) thisMax = dimensions.x[dd]-1;
                jmax.x[dd] = thisMax;
                if(jmax.x[dd] < jmin.x[dd])
                    {
                    int temp = jmax.x[dd];
                    jmax.x[dd] = jmin.x[dd];
                    jmin.x[dd] = temp;
                    };
                };
            j=jmin;
            j.x[0]-=1;
            dVec disp;
            bool cellChecksGood = true;
            while(iVecIterate(j,jmin,jmax))
                {
                //is there a sample at j that is too close?
                k = nDimArrayIndex(dimensions,j);
                if(accel[k] >=0 && accel[k] != cur)
                    {
                    Box->minDist(x,sample[accel[k]],disp);
                    if(norm(disp) < radius)
                        {
                        cellChecksGood = false;
                        break;
                        };
                    };
                };
            if(cellChecksGood)
                foundSample = true;
            break;
            };//end of "loop over attempts"
        if(foundSample)
            {
            size_t q = sample.size();
            sample.push_back(x);
            activeList.push_back(q);
            k = nDimArrayIndex(dimensions,(1/gridDx)*(x-min));
            accel[k] = (int)q;
            break;
            }
        else
            {
            activeList[r] = activeList.back();
            activeList.pop_back();
            };
        };
    };
    };

/*!
simple 1D indexing of a hypercubic cell structure
 */
int poissonDiskSampling::nDimArrayIndex(const iVec &dimensions, const iVec &x)
    {
    int k =0;
    if(x.x[DIMENSION-1] >=0)
        {
        k = (int) x.x[DIMENSION-1];
        if(k>=dimensions.x[DIMENSION-1]) k = dimensions.x[DIMENSION-1]-1;
        };
    for (int ii =  DIMENSION-1; ii >0; --ii)
        {
        k *= dimensions.x[ii-1];
        if(x.x[ii-1] >=0)
            {
            int j = (int)x.x[ii-1];
            if(j >= dimensions.x[ii-1]) j = dimensions.x[ii-1]-1;
            k += j;
            };
        };
    return k;
    };
/*!
simple 1D indexing of a hypercubic cell structure
 */
int poissonDiskSampling::nDimArrayIndex(const iVec &dimensions, const dVec &x)
    {
    int k =0;
    if(x.x[DIMENSION-1] >=0)
        {
        k = (int) x.x[DIMENSION-1];
        if(k>=dimensions.x[DIMENSION-1]) k = dimensions.x[DIMENSION-1]-1;
        };
    for (int ii =  DIMENSION-1; ii >0; --ii)
        {
        k *= dimensions.x[ii-1];
        if(x.x[ii-1] >=0)
            {
            int j = (int)x.x[ii-1];
            if(j >= dimensions.x[ii-1]) j = dimensions.x[ii-1]-1;
            k += j;
            };
        };
    return k;
    };

/*!
Sample and reject to get a point in an anulus
 */
void poissonDiskSampling::sample_annulus_point(scalar radius, const dVec &center, dVec &answer,noiseSource &noise)
    {
    dVec r;
    bool pointFound = false;
    while(!pointFound)
        {
        for (int dd = 0; dd < DIMENSION; ++dd)
            r.x[dd] = 4*noise.getRealUniform(-.5,.5);
        scalar r2 = dot(r,r);
        if(r2 > 1 && r2 <=4)
            pointFound = true;
        };
    answer = center + radius*r;
    };
#endif
