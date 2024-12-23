#ifndef INITIALIZATIONFUNCTIONS_H
#define INITIALIZATIONFUNCTIONS_H

#include <string>
#include <vector>
#include <iostream>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
//!Get basic stats about the chosen GPU (if it exists)
bool chooseGPU(int USE_GPU,bool verbose = false)
    {
    int nDev;
    cudaGetDeviceCount(&nDev);
    if (USE_GPU >= nDev)
        {
        std::cout << "Requested GPU (device " << USE_GPU<<") does not exist." << std::endl;
        return false;
        };
    if (USE_GPU <nDev)
        cudaSetDevice(USE_GPU);
    if(verbose)
        {
        std::cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << std::endl;
        for (int ii=0; ii < nDev; ++ii)
            {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop,ii);
            if (ii == USE_GPU) std::cout << "********************************" << std::endl;
            if (ii == USE_GPU) std::cout << "****Using the following gpu ****" << std::endl;
            std::cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << std::endl;
            if (ii == USE_GPU) std::cout << "*******************************" << std::endl;
            };
        }
    else
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        std::cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << std::endl << std::endl;
        };
    return true;
    }

//!Get basic stats about the chosen GPU (if it exists)
bool getAvailableGPUs(std::vector<std::string> &devices)
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
