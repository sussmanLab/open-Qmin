#include <iostream>
#include <unistd.h>
#include "dataTypesAndContainers.h"
#include "noiseSource.h"

int main(int argc, char*argv[])
{
    int c;

    int n = 1;
    while((c=getopt(argc,argv,"n:")) != -1)
        switch(c)
            {
            case 'n': n = atoi(optarg); break;
            case '?':
                if(optopt=='c')
                    std::cerr<<"Option -" << optopt << "requires an argument.\n";
                else if(isprint(optopt))
                    std::cerr<<"Unknown option '-" << optopt << "'.\n";
                else
                    std::cerr << "Unknown option character.\n";
                return 1;
            default:
               abort();
            };
    #ifdef ENABLE_CUDA
    printf("cuda enabled. x = %i\n",n);
    #endif


    //initialize with data
    GPUArray<float2> test(n);
    float2 intSum; intSum.x=0;intSum.y=0;
    {
    ArrayHandle<float2> t(test);
    for (int ii = 0; ii < n; ++ii)
        {
        t.data[ii].x = ii;
        t.data[ii].y = -ii;
        intSum = intSum - t.data[ii];
        }
    }

    float sum1 = 0;
    float sum2 = 0;
    #ifdef ENABLE_CUDA
    //sum reduction on device?
    UNWRITTENCODE("not done yet");
    #else
    noiseSource noise(false);
    {//sum reduction on host
    ArrayHandle<float2> t(test);
    for (int ii = 0; ii < n; ++ii)
        {
        sum1 += t.data[ii].x;
        sum2 += t.data[ii].y + noise.getRealUniform(-.1,.1);
        }
    }
    #endif
    std::cout << sum1<< "  " << sum2 << std::endl;
    std::cout << intSum.x<< "  " << intSum.y << std::endl;
    return 0;
};

