#ifndef kernelTuner_H
#define kernelTuner_H

#include <chrono>
#include <iostream>
#include <vector>
/*!\file kernelTuner.h */

//!A class that tries to dynamically optimize a kernel parameter
class kernelTuner
    {
    public:
        kernelTuner(){};
        //!Base constructor takes (start,end,step) values to scan, sample number, and period
        kernelTuner(int start, int end, int step, int nSamples, int _period);

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
                std::cout << "parameter used: " << parameterValue << std::endl;
            for (int ii = 1; ii < possibleParameters.size();++ii)
                {
                    std::cout <<"tuner value " << possibleParameters[ii] << " median time " << sampleMedian[ii]  << std::endl;
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
        std::vector<int> possibleParameters;
        State internalState;
        int currentSample;
        int currentParameterIndex;
        int callsSinceLastSample;
        std::vector<std::vector< float> > sampleData;
        std::vector<float> sampleMedian;

        //cudaEvent_t startEvent;
        //cudaEvent_t stopEvent;
        std::chrono::time_point<std::chrono::high_resolution_clock>  startTime;
        std::chrono::time_point<std::chrono::high_resolution_clock>  endTime;

    };
#endif
