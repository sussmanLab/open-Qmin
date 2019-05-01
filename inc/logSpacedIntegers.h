#ifndef logSpacedInts_H
#define logSpacedInts_H

#include "std_include.h"

//!A small function of convenience to keep track of log spaced integers
class logSpacedIntegers
    {
    public:
        //!start with a number and an exponent
        logSpacedIntegers(int firstSave = 0, scalar _exp = 0.05)
            {
            nextSave = firstSave;
            exponent = _exp;
            base = pow(10.0,exponent);
            if(nextSave == 0)
                {
                logSaveIdx = 0;
                }
            else
                {
                logSaveIdx = 0;
                int tempCur = 0;
                while(tempCur < nextSave)
                    {
                    logSaveIdx += 1;
                    tempCur = (int)round(pow(base,logSaveIdx));
                    }
                }
            };

        void update()
            {
            logSaveIdx +=1;
            int curSave = (int)round(pow(base,logSaveIdx));
            while(curSave == nextSave)
                {
                logSaveIdx +=1;
                curSave = (int)round(pow(base,logSaveIdx));
                }
            nextSave = curSave;
            }
        int nextSave;
        int logSaveIdx;
        scalar exponent;
        scalar base;
    };

#endif
