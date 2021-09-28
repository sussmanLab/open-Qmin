#ifndef STDINCLUDE_H
#define STDINCLUDE_H

/*! \file std_include.h
a file to be included all the time... carries with it things DMS often uses.
It includes some handy debugging / testing functions, and includes too many
standard library headers
It also defines scalars as either floats or doubles, depending on
how the program is compiled
*/

#define THRESHOLD 1e-18
#define EPSILON 1e-18

#include <cmath>
#include <algorithm>
#include <memory>
#include <ctype.h>
#include <random>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/time.h>
#include <string.h>
#include <stdexcept>
#include <cassert>

using namespace std;

#include <cuda_runtime.h>
#include "vector_types.h"
#include "vector_functions.h"
#include "nvToolsExt.h"

#define PI 3.14159265358979323846
#define sqrt2 1.4142135623730950488
#define sqrt3 1.7320508075688772935

//decide whether to compute everything in floating point or double precision
#ifndef SCALARFLOAT
//double variables types
#define scalar double
#define scalar2 double2
#define scalar3 double3
#define scalar4 double4
//the netcdf variable type
#define ncscalar ncDouble
//the cuda RNG
#define cur_norm curand_normal_double
//trig and special funtions
#define Cos cos
#define Sin sin
#define Floor floor
#define Ceil ceil
#define MPI_SCALAR MPI_DOUBLE

#else
//floats
#define scalar float
#define scalar2 float2
#define scalar3 float3
#define scalar4 float4
#define ncscalar ncFloat
#define cur_norm curand_normal
#define Cos cosf
#define Sin sinf
#define Floor floorf
#define Ceil ceilf
#define MPI_SCALAR MPI_FLOAT
#endif

#include "dDimensionalVectorTypes.h"
#include "matrix.h"
#include "structures.h"

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

//!return a scalar2 from two scalars
HOSTDEVICE scalar2 make_scalar2(scalar x, scalar y)
    {
    scalar2 ans;
    ans.x=x;
    ans.y=y;
    return ans;
    }

//!return a scalar3 from three scalars
HOSTDEVICE scalar3 make_scalar3(scalar x, scalar y,scalar z)
    {
    scalar3 ans;
    ans.x=x;
    ans.y=y;
    ans.z=z;
    return ans;
    }
//!scalar multiplication of scalar3
HOSTDEVICE scalar3 operator*(const scalar3 &a, const scalar &b)
    {
    return make_scalar3(a.x*b,a.y*b,a.z*b);
    }
//!scalar multiplication of scalar3
HOSTDEVICE scalar3 operator*(const scalar b,const scalar3 &a)
    {
    return make_scalar3(a.x*b,a.y*b,a.z*b);
    }
//!component-wise addition of two scalar3s
HOSTDEVICE scalar3 operator+(const scalar3 &a, const scalar3 &b)
    {
    return make_scalar3(a.x+b.x,a.y+b.y,a.z+b.z);
    }

//!component-wise subtraction of two int3s
HOSTDEVICE int3 operator-(const int3 &a, const int3 &b)
    {
    return make_int3(a.x-b.x,a.y-b.y,a.z-b.z);
    }
//!component-wise addition of two int3s
HOSTDEVICE int3 operator+(const int3 &a, const int3 &b)
    {
    return make_int3(a.x+b.x,a.y+b.y,a.z+b.z);
    }

//!strict comparison of int3s
HOSTDEVICE bool operator<(const int3 &a,const int3 &b)
    {
    return (a.x < b.x && a.y < b.y && a.z < b.z);
    }
HOSTDEVICE bool operator>(const int3 &a,const int3 &b)
    {
    return (a.x > b.x && a.y > b.y && a.z > b.z);
    }
//!comparison of int3s
HOSTDEVICE bool operator<=(const int3 &a,const int3 &b)
    {
    return (a.x <= b.x && a.y <= b.y && a.z <= b.z);
    }
HOSTDEVICE bool operator>=(const int3 &a,const int3 &b)
    {
    return (a.x >= b.x && a.y >= b.y && a.z >= b.z);
    }

//!Handle errors in kernel calls...returns file and line numbers if cudaSuccess doesn't pan out
static void HandleError(cudaError_t err, const char *file, int line)
    {
    //as an additional debugging check, synchronize cuda threads after every kernel call
    #ifdef DEBUGFLAGUP
    cudaThreadSynchronize();
    #endif
    if (err != cudaSuccess)
        {
        printf("\nError: %s in file %s at line %d\n",cudaGetErrorString(err),file,line);
        throw std::exception();
        }
    }

//!Report somewhere that code needs to be written
static void unwrittenCode(const char *message, const char *file, int line)
    {
    printf("\nCode unwritten (file %s; line %d)\nMessage: %s\n",file,line,message);
    throw std::exception();
    }

//!A utility function for checking if a file exists
inline bool fileExists(const std::string& name)
    {
    ifstream f(name.c_str());
    return f.good();
    }

static void nvtxProfPush(const char *message)
    {
    #ifdef DEBUGFLAGUP
    nvtxRangePushA(message);
    printf("%s\n",message);
    #endif
    };

static void nvtxProfPop()
    {
    #ifdef DEBUGFLAGUP
    nvtxRangePop();
    #endif
    };

#define NVTXPUSH(message) (nvtxProfPush(message))
#define NVTXPOP(message) (nvtxProfPop())

//A macro to wrap cuda calls
#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))
//A macro to say code needs to be written
#define UNWRITTENCODE(message) (unwrittenCode(message,__FILE__,__LINE__))
//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);

#undef HOSTDEVICE
#endif
