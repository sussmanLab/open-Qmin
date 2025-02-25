#ifndef profiler_H
#define profiler_H

#include <chrono>
#include <string>
#include <iostream>
class profiler
    {
    public:
        profiler(std::string profilerName) : name(profilerName) {functionCalls = 0; timeTaken = 0;};

        void start()
            {
            startTime = std::chrono::high_resolution_clock::now();
            };
        void end()
            {
            endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> difference = endTime-startTime;
            timeTaken += difference.count();
            functionCalls +=1;
            };

        double timing()
            {
            if(functionCalls>0)
                return timeTaken/functionCalls;
            else
                return 0;
            };

        void print()
            {
            std::cout << "profiler \"" << name << "\" took an average of " << timing() << " per call over " << functionCalls << " calls...total time = "<<timing()*functionCalls << std::endl;
            }

        std::chrono::time_point<std::chrono::high_resolution_clock>  startTime;
        std::chrono::time_point<std::chrono::high_resolution_clock>  endTime;
        int functionCalls;
        double timeTaken;
        std::string name;
    };
#endif
