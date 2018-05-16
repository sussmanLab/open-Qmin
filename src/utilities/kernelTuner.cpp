#include "kernelTuner.h"
/*!\file kernelTuner.cpp */

kernelTuner::kernelTuner(int start, int end, int step, int nSamples, int _period)
    {
    internalState=STARTUP;
    currentSample = 0;
    currentParameterIndex = 0;
    callsSinceLastSample = 0;
    period = _period;

    parameterValue = start;
    //set vector of possible parameters
    for(int ii = start; ii <=end; ii +=step)
        possibleParameters.push_back(ii);
    //force samplesPerValue to be odd
    samplesPerValue=nSamples;
    if(samplesPerValue%2==1)
        samplesPerValue +=1;
    sampleData.resize(possibleParameters.size());
    sampleMedian.resize(possibleParameters.size());
    for(int ii = 0; ii < possibleParameters.size();++ii)
        sampleData[ii].resize(samplesPerValue);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    };

kernelTuner::~kernelTuner()
    {
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    };

void kernelTuner::begin()
    {
    if (internalState != IDLE)
        cudaEventRecord(startEvent,0);
    };

void kernelTuner::end()
    {
    //record the timing data
    if(internalState != IDLE)
        {
        cudaEventRecord(stopEvent,0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&sampleData[currentParameterIndex][currentSample],startEvent,stopEvent);
        };

    //handle all of the parameter scanning updates
    if(internalState ==STARTUP)
        {
        currentSample += 1;
        if(currentSample >= samplesPerValue)
            {
            currentSample = 0;
            currentParameterIndex += 1;

            if(currentParameterIndex >= possibleParameters.size())
                {
                currentParameterIndex = 0;
                internalState = IDLE;
                parameterValue = computeOptimalParameter();
                }
            else
                {
                parameterValue = possibleParameters[currentParameterIndex];
                }
            }
        }
    else if (internalState == SCANNING)
        {
        currentParameterIndex += 1;
        //if that's past the end, transition to idle state
        if(currentParameterIndex >= possibleParameters.size())
            {
            currentParameterIndex = 0;
            internalState = IDLE;
            parameterValue = computeOptimalParameter();
            currentSample = (currentSample+1)%samplesPerValue;
            }
        else
            {
            parameterValue = possibleParameters[currentParameterIndex];
            }
        }
    else if (internalState == IDLE)
        {
        callsSinceLastSample += 1;
        //if it's been longer than (period), transition back to scanning state
        if(callsSinceLastSample > period)
            {
            callsSinceLastSample = 0;
            parameterValue = possibleParameters[currentParameterIndex];
            internalState = SCANNING;
            }
        }
    };

int kernelTuner::computeOptimalParameter()
    {
    //compute the median time for each parameter value
    vector<float> times;
    for (int ii  = 0; ii < possibleParameters.size(); ++ii)
        {
        times = sampleData[ii];
        size_t middle = times.size()/2;
        nth_element(times.begin(),times.begin()+middle,times.end());
        sampleMedian[ii] = times[middle];
        }
    //with the medians in hand, find the fastest one
    int optimumIndex = 0;
    scalar fastest = sampleMedian[0];
    for (int ii = 1; ii < possibleParameters.size();++ii)
        {
//        cout <<"tuner value " << possibleParameters[ii] << " median time " << sampleMedian[ii]  << endl;
        if(sampleMedian[ii] < fastest)
            {
            fastest = sampleMedian[ii];
            optimumIndex = ii;
            }
        };

    return possibleParameters[optimumIndex];
    };
