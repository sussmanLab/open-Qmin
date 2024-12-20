#ifndef INITIALIZATIONFUNCTIONS_H
#define INITIALIZATIONFUNCTIONS_H

#include <string>
#include <vector>

#ifdef ENABLE_CUDA
//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        cout << "Requested GPU (device " << USE_GPU<<") does not exist." << endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)
        {
        cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
        for (int ii=0; ii < nDev; ++ii)
            {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop,ii);
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        }
    else
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    }

//!Get basic stats about the chosen GPU (if it exists)
__host__ inline bool getAvailableGPUs(std::vector<std::string> &devices)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        std::string curName(prop.name);
        devices.push_back(curName);
        };
    return true;
    }
#else
bool chooseGPU(int USE_GPU,bool verbose=false)
    {
    return false;
    }
bool getAvailableGPUs(std::vector<std::string> &devices)
    {
    return false;
    }
#endif

#endif
