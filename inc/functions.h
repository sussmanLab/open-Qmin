#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "std_include.h"
#include "gpuarray.h"
#include <set>

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file functions.h */

/** @defgroup Functions functions
 * @{
 \brief Utility functions that can be called from host or device
 */

//!remove duplicate elements from a vector, preserving the order, using sets
template<typename T>
inline __attribute__((always_inline)) void removeDuplicateVectorElements(vector<T> &data)
    {
    set<T> currentElements;
    auto newEnd = std::remove_if(data.begin(),data.end(),[&currentElements](const T& value)
        {
        if (currentElements.find(value) != std::end(currentElements))
            return true;
        currentElements.insert(value);
        return false;
        });
    data.erase(newEnd,data.end());
    };

//!shrink a GPUArray by removing the i'th element and shifting any elements j > i into place
template<typename T>
inline __attribute__((always_inline)) void removeGPUArrayElement(GPUArray<T> &data, int index)
    {
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n-1);
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::read);
    ArrayHandle<T> h1(newData,access_location::host,access_mode::overwrite);
    int idx = 0;
    for (int i = 0; i < n; ++i)
        {
        if (i != index)
            {
            h1.data[idx] = h.data[i];
            idx += 1;
            };
        };
    };
    data = newData;
    };

//!shrink a GPUArray by removing the elements [i1,i2,...in] of a vector and shifting any elements j > i_i into place
template<typename T>
inline __attribute__((always_inline)) void removeGPUArrayElement(GPUArray<T> &data, vector<int> indices)
    {
    std::sort(indices.begin(),indices.end());
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n-indices.size());
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::read);
    ArrayHandle<T> h1(newData,access_location::host,access_mode::overwrite);
    int idx = 0;
    int vectorIndex = 0;
    for (int i = 0; i < n; ++i)
        {
        if (i != indices[vectorIndex])
            {
            h1.data[idx] = h.data[i];
            idx += 1;
            }
        else
            {
            vectorIndex += 1;
            if (vectorIndex >= indices.size())
                vectorIndex -= 1;
            };
        };
    };
    data = newData;
    };

//!grow a GPUArray, leaving the current elements the same but with extra capacity at the end of the array
template<typename T>
inline __attribute__((always_inline)) void growGPUArray(GPUArray<T> &data, int extraElements)
    {
    int n = data.getNumElements();
    GPUArray<T> newData;
    newData.resize(n+extraElements);
    {//scope for array handles
    ArrayHandle<T> h(data,access_location::host,access_mode::readwrite);
    ArrayHandle<T> hnew(newData,access_location::host,access_mode::overwrite);
    for (int i = 0; i < n; ++i)
        hnew.data[i] = h.data[i];
    };
    //no need to resize?
    data.swap(newData);
    };

//!fill the first data.size() elements of a GPU array with elements of the data vector
template<typename T>
inline __attribute__((always_inline)) void fillGPUArrayWithVector(vector<T> &data, GPUArray<T> &copydata)
    {
    int Narray = copydata.getNumElements();
    int Nvector = data.size();
    if (Nvector > Narray)
        copydata.resize(Nvector);
    ArrayHandle<T> handle(copydata,access_location::host,access_mode::overwrite);
    for (int i = 0; i < Nvector; ++i)
        handle.data[i] = data[i];
    };

//!copy a GPUarray to a vector
template<typename T>
inline __attribute__((always_inline)) void copyGPUArrayData(GPUArray<T> &data, vector<T> &copydata)
    {
    int n = data.getNumElements();
    ArrayHandle<T> handle(data,access_location::host,access_mode::read);
    copydata.resize(n);
    for (int i = 0; i < n; ++i) copydata[i] = handle.data[i];
    };

//!The dot product between d-Dimensional vectors.
HOSTDEVICE scalar dot(const dVec &p1, const dVec &p2)
    {
    scalar ans = 0.0;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans+=p1.x[dd]*p2.x[dd];

    return ans;
    };

//! an integer to the dth power... the slow way
HOSTDEVICE int idPow(int i)
    {
    int ans = i;
    for (int dd = 1; dd < DIMENSION; ++dd)
        ans *= i;
    return ans;
    };

//!The dot product between d-Dimensional iVecs.
HOSTDEVICE int dot(const iVec &p1, const iVec &p2)
    {
    int ans = 0;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans+=p1.x[dd]*p2.x[dd];

    return ans;
    };

//!The norm of a d-Dimensional vector
HOSTDEVICE scalar norm(const dVec &p)
    {
    return sqrt(dot(p,p));
    };

//!fit integers into non-negative domains
HOSTDEVICE int wrap(int x,int m)
    {
    int ans = x;
    if(x >= m)
        ans = x % m;
    while(ans <0)
        ans += m;
    return ans;
    }


//!compute the sign of a scalar, and return zero if x = 0
HOSTDEVICE int computeSign(scalar x)
    {
    return ((x>0)-(x<0));
    };

//!compute the sign of a scalar, and return zero if x = 0...but return a scalar to avoid expensive casts on the GPU
HOSTDEVICE scalar computeSignNoCast(scalar x)
    {
    if (x > 0.) return 1.0;
    if (x < 0.) return -1.0;
    if (x == 0.) return 0.;
    return 0.0;
    };

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
    if(verbose)    cout << "Device # \t\t Device Name \t\t MemClock \t\t MemBusWidth" << endl;
    for (int ii=0; ii < nDev; ++ii)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,ii);
        if (verbose)
            {
            if (ii == USE_GPU) cout << "********************************" << endl;
            if (ii == USE_GPU) cout << "****Using the following gpu ****" << endl;
            cout << ii <<"\t\t\t" << prop.name << "\t\t" << prop.memoryClockRate << "\t\t" << prop.memoryBusWidth << endl;
            if (ii == USE_GPU) cout << "*******************************" << endl;
            };
        };
    if (!verbose)
        {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop,USE_GPU);
        cout << "using " << prop.name << "\t ClockRate = " << prop.memoryClockRate << " memBusWidth = " << prop.memoryBusWidth << endl << endl;
        };
    return true;
    };

/** @} */ //end of group declaration
#undef HOSTDEVICE
#endif
