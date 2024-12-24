#include "gpuarray.h"
#include "cudaDataTypes.h"
#include "dDimensionalVectorTypes.h"
#include "latticeBoundaries.h"
#include "matrix.h"
#include "std_include.h"
#include <cstddef>
#include <cstring>
#include <utility> //the new include algorithm, since c++11
#include <stdlib.h>
#include <stdexcept>
#ifdef ENABLE_CUDA
#include "curand_kernel.h"
#include <cuda_runtime.h>
#endif


template<class T>
ArrayHandle<T>::ArrayHandle( GPUArray<T>& _gpu_array,  access_location::Enum location,
                                               access_mode::Enum mode) :
        data(_gpu_array.acquire(location, mode)), gpu_array(_gpu_array)
    {
    }

template<class T>
ArrayHandle<T>::~ArrayHandle()
    {
    gpu_array.Acquired = false;
    }

template<class T>
GPUArray<T>::GPUArray() :
        whereIsTheData(data_location::host), arraySize(0), Acquired(false), RegisterArray(false),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        dataHasTouchedDevice(false),
        h_data(NULL)
    {
    }

template<class T>
GPUArray<T>::GPUArray(unsigned int num_elements, bool _register) :
        whereIsTheData(data_location::host), arraySize(num_elements), Acquired(false), RegisterArray(_register),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        dataHasTouchedDevice(false),
        h_data(NULL)
    {
    // allocate and clear memory
    allocate();
    memclear();
    }

template<class T>
GPUArray<T>::~GPUArray()
    {
    deallocate();
    }
template <class T>
GPUArray<T>::GPUArray(GPUArray<T>&& other)  
    : whereIsTheData(other.whereIsTheData), 
      arraySize(other.arraySize), 
      Acquired(other.Acquired),
#ifdef ENABLE_CUDA
      d_data(other.d_data),
#endif
      h_data(other.h_data)
    {
    // Transfer ownership of the data
    other.whereIsTheData = data_location::host; // Or another appropriate default
    other.arraySize = 0;
    other.Acquired = false;
#ifdef ENABLE_CUDA
    other.d_data = NULL; 
#endif
    other.h_data = NULL;
    }

template<class T>
GPUArray<T>::GPUArray( GPUArray& from) : whereIsTheData(data_location::host),
        arraySize(from.arraySize), Acquired(false),
#ifdef ENABLE_CUDA
        d_data(NULL),
#endif
        h_data(NULL)
    {
    // allocate and clear new memory the same size as the data in from
    allocate();
    memclear();

    // copy over the data to the new GPUArray
    if (arraySize > 0)
        {
        ArrayHandle<T> h_handle(from, access_location::host, access_mode::read);
        memcpy(h_data, h_handle.data, sizeof(T)*arraySize);
        }
    }

template<class T>
GPUArray<T>& GPUArray<T>::operator=( GPUArray& rhs)
    {
    if (this != &rhs) // protect against invalid self-assignment
        {
        // free current memory
        deallocate();

        // is the array registered
        RegisterArray = rhs.RegisterArray;
        dataHasTouchedDevice = false;

        // copy over basic elements
        arraySize = rhs.arraySize;

        // initialize state variables
        whereIsTheData = data_location::host;

        // allocate and clear new memory the same size as the data in rhs
        allocate();
        memclear();

        // copy over the data to the new GPUArray
        if (arraySize > 0)
            {
            ArrayHandle<T> h_handle(rhs, access_location::host, access_mode::read);
            memcpy(h_data, h_handle.data, sizeof(T)*arraySize);
            }
        }

    return *this;
    }

/*!
    a.swap(b) is:
        GPUArray c(a);
        a = b;
        b = c;
    It just swaps internal pointers
*/
template<class T>
void GPUArray<T>::swap(GPUArray& from)
    {
    std::swap(arraySize, from.arraySize);
    std::swap(Acquired, from.Acquired);
    std::swap(whereIsTheData, from.whereIsTheData);
    std::swap(RegisterArray,from.RegisterArray);
#ifdef ENABLE_CUDA
    std::swap(d_data, from.d_data);
#endif
    std::swap(h_data, from.h_data);
    }

template<class T>
void GPUArray<T>::allocate()
    {
    // don't allocate anything if there are zero elements
    if (arraySize == 0)
        return;
    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void**)&h_data, 32, arraySize*sizeof(T));
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating GPUArray.");
        }

    /*
    if(RegisterArray)
        cudaHostRegister(h_data,arraySize*sizeof(T),cudaHostRegisterDefault);
      */
#ifdef ENABLE_CUDA
    if(dataHasTouchedDevice)
        cudaMalloc(&d_data, arraySize*sizeof(T));
#endif
    }

template<class T>
void GPUArray<T>::deallocate()
    {
    // don't do anything if there are no elements
    if (arraySize == 0)
        return;
    // free memory
#ifdef ENABLE_CUDA
    if(dataHasTouchedDevice)
        {
        cudaFree(d_data);
        d_data = NULL;
        };
#endif
    /*
    if(RegisterArray)
        cudaHostUnregister(h_data);
    */

    free(h_data);

    // set pointers to NULL
    h_data = NULL;
    }

template<class T>
void GPUArray<T>::memclear(unsigned int first)
    {
    // don't do anything if there are no elements
    if (arraySize == 0)
        return;

    // clear memory
    memset(h_data+first, 0, sizeof(T)*(arraySize-first));
#ifdef ENABLE_CUDA
    if(dataHasTouchedDevice)
        cudaMemset(d_data+first, 0, (arraySize-first)*sizeof(T));
#endif
    }

/*!
    Acquire does all the work, keeping track of when data needs to be copied, etc.
    It is called by the ArrayHandle class
*/
template<class T>
T* GPUArray<T>::acquire( access_location::Enum location,  access_mode::Enum mode) 
    {
    Acquired = true;
    //only allocate memory on the device the first time it is needed
#ifdef ENABLE_CUDA
    if(location == access_location::device && !dataHasTouchedDevice)
        {
        resizeDeviceArray(arraySize);
        }
#endif
    // (1) where do we want the data? (2) where *is* the data? (3) copy if necessary
    // if only reading, often avoid a copy
    if (location == access_location::host)
        {
        if (whereIsTheData == data_location::host)
            {
            return h_data;
            }
#ifdef ENABLE_CUDA
        else if (whereIsTheData == data_location::hostdevice)
            {
            if (mode == access_mode::read)
                whereIsTheData = data_location::hostdevice;
            else if (mode == access_mode::readwrite)
                whereIsTheData = data_location::host;
            else if (mode == access_mode::overwrite)
                whereIsTheData = data_location::host;
            else
                {
                throw std::runtime_error("Error acquiring data7");
                }

            return h_data;
            }
        else if (whereIsTheData == data_location::device)
            {
            if (mode == access_mode::read)
                {
                memcpyDeviceToHost();
                whereIsTheData = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                memcpyDeviceToHost();
                whereIsTheData = data_location::host;
                }
            else if (mode == access_mode::overwrite)
                {
                whereIsTheData = data_location::host;
                }
            else
                {
                throw std::runtime_error("Error acquiring data6");
                }

            return h_data;
            }
#endif
        else
            {
            throw std::runtime_error("Error acquiring data5");
            }
        }
#ifdef ENABLE_CUDA
    else if (location == access_location::device)
        {
        if (whereIsTheData == data_location::host)
            {
            if (mode == access_mode::read)
                {
                memcpyHostToDevice();
                whereIsTheData = data_location::hostdevice;
                }
            else if (mode == access_mode::readwrite)
                {
                memcpyHostToDevice();
                whereIsTheData = data_location::device;
                }
            else if (mode == access_mode::overwrite)
                {
                whereIsTheData = data_location::device;
                }
            else
                {
                throw std::runtime_error("Error acquiring data4");
                }

            return d_data;
            }
        else if (whereIsTheData == data_location::hostdevice)
            {
            if (mode == access_mode::read)
                whereIsTheData = data_location::hostdevice;
            else if (mode == access_mode::readwrite)
                whereIsTheData = data_location::device;
            else if (mode == access_mode::overwrite)
                whereIsTheData = data_location::device;
            else
                {
                throw std::runtime_error("Error acquiring data3");
                }
            return d_data;
            }
        else if (whereIsTheData == data_location::device)
            {
            return d_data;
            }
        else
            {
            throw std::runtime_error("Error acquiring data2");
            }
        }
#endif
    else
        {
        throw std::runtime_error("Error acquiring data1");
        }
    }

template<class T>
T* GPUArray<T>::resizeHostArray(unsigned int num_elements)
    {
    // allocate resized array
    T *h_tmp = NULL;

    // allocate host memory
    // at minimum, alignment needs to be 32 bytes for AVX
    int retval = posix_memalign((void**)&h_tmp, 32, num_elements*sizeof(T));
    if (retval != 0)
        {
        throw std::runtime_error("Error allocating GPUArray.");
        }

//    if(RegisterArray)
//        cudaHostRegister(h_tmp,arraySize*sizeof(T),cudaHostRegisterDefault);

    // clear memory
    memset(h_tmp, 0, sizeof(T)*num_elements);

    // copy over data
    unsigned int num_copy_elements = arraySize > num_elements ? num_elements : arraySize;
    memcpy(h_tmp, h_data, sizeof(T)*num_copy_elements);

//    if(RegisterArray)
//        cudaHostUnregister(h_data);

    // free old memory location
    free(h_data);
    h_data = h_tmp;

    return h_data;
    }


template<class T>
void GPUArray<T>::resize(unsigned int num_elements)
    {
    resizeHostArray(num_elements);
    arraySize = num_elements;
#ifdef ENABLE_CUDA
    if(dataHasTouchedDevice)
        resizeDeviceArray(num_elements);
#endif
    }

//Some functions that are only even defined in the header if ENABLE_CUDA is on
#ifdef ENABLE_CUDA

template<class T>
void GPUArray<T>::memcpyDeviceToHost() 
    {
    // don't do anything if there are no elements
    if (arraySize == 0)
        return;


    cudaMemcpy(h_data, d_data, sizeof(T)*arraySize, cudaMemcpyDeviceToHost);

    }

template<class T>
void GPUArray<T>::memcpyHostToDevice() 
    {
    // don't do anything if there are no elements
    if (arraySize == 0)
        return;

    cudaMemcpy(d_data, h_data, sizeof(T)*arraySize, cudaMemcpyHostToDevice);
    }

template<class T>
void GPUArray<T>::setRegistered(bool _reg)
    {
    RegisterArray=_reg;
    if(RegisterArray)
        cudaHostRegister(h_data,arraySize*sizeof(T),cudaHostRegisterDefault);
    }

template<class T>
T* GPUArray<T>::resizeDeviceArray(unsigned int num_elements)
    {
    //if we've never put data on the device, allocate some space for it?
    if(!dataHasTouchedDevice)
        {
        dataHasTouchedDevice = true;
        cudaMalloc(&d_data, arraySize*sizeof(T));
        }

    // allocate resized array
    T *d_tmp;
    cudaMalloc(&d_tmp, num_elements*sizeof(T));

    // clear memory
    cudaMemset(d_tmp, 0, num_elements*sizeof(T));

    // copy over data
    unsigned int num_copy_elements = arraySize > num_elements ? num_elements : arraySize;
    cudaMemcpy(d_tmp, d_data, sizeof(T)*num_copy_elements,cudaMemcpyDeviceToDevice);

    // free old memory location
    cudaFree(d_data);

    d_data = d_tmp;
    return d_data;
    }

#endif

template class ArrayHandle<int>;
template class GPUArray<int>;
template class ArrayHandle<int2>;
template class GPUArray<int2>;
template class ArrayHandle<int3>;
template class GPUArray<int3>;
template class ArrayHandle<int4>;
template class GPUArray<int4>;

template class ArrayHandle<float>;
template class GPUArray<float>;
template class ArrayHandle<float2>;
template class GPUArray<float2>;
template class ArrayHandle<float3>;
template class GPUArray<float3>;
template class ArrayHandle<float4>;
template class GPUArray<float4>;

template class ArrayHandle<double>;
template class GPUArray<double>;
template class ArrayHandle<double2>;
template class GPUArray<double2>;
template class ArrayHandle<double3>;
template class GPUArray<double3>;
template class ArrayHandle<double4>;
template class GPUArray<double4>;

template class ArrayHandle<boundaryObject>;
template class GPUArray<boundaryObject>;

template class ArrayHandle<dVec>;
template class GPUArray<dVec>;

template class ArrayHandle<cubicLatticeDerivativeVector>;
template class GPUArray<cubicLatticeDerivativeVector>;

template class ArrayHandle<std::pair<int, dVec>>;
template class GPUArray<std::pair<int, dVec>>;

template class ArrayHandle<Matrix3x3>;
template class GPUArray<Matrix3x3>;

template class ArrayHandle<curandStateXORWOW>;
template class GPUArray<curandStateXORWOW>;
