#ifndef kernelTuner_H
#define kernelTuner_H

#include "std_include.h"
/*!\file kernelTuner.h */

//!A class that tries to dynamically optimize a kernel parameter
class kernelTuner
    {
    public:
        //!Base constructor takes (start,end,step) values to scan, sample number, and period
        kernelTuner(int start, int end, int step, int nSamples, int _period);
        //!destroy the cuda events
        ~kernelTuner();

        void begin();
        void end();
        //!return the parameter to use for the kernel
        int getParameter()
            {
            return parameterValue;
            };

        //! print timing data to screen
        void printTimingData()
            {
            cout << "parameter used: " << parameterValue << endl;
            for (int ii = 1; ii < possibleParameters.size();++ii)
                {
                cout <<"tuner value " << possibleParameters[ii] << " median time " << sampleMedian[ii]  << endl;
                }
            };

        //!Is initial sampling complete?
        bool samplingComplete()
            {
            return (internalState != STARTUP);
            };

    protected:

        int computeOptimalParameter();
        //!names for the internal state
        enum State
            {
            STARTUP,
            IDLE,
            SCANNING
            };

        int parameterValue;
        int samplesPerValue;
        int period;
        vector<int> possibleParameters;
        State internalState;
        int currentSample;
        int currentParameterIndex;
        int callsSinceLastSample;
        vector<vector< float> > sampleData;
        vector<float> sampleMedian;

        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;
    };
#endif
