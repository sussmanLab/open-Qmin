#ifndef STDINCLUDE
#define STDINCLUDE

/*! \file std_include.h
a file to be included all the time... carries with it things DMS often uses
Crucially, it also defines scalars as either floats or doubles, depending on
how the program is compiled
*/

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

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

#define PI 3.14159265358979323846

//decide whether to compute everything in floating point or double precision
#ifndef SCALARFLOAT
//double variables types
#define scalar double
//the netcdf variable type
#define ncscalar ncDouble
//the cuda RNG
#define cur_norm curand_normal_double
//trig and special funtions
#define Cos cos
#define Sin sin
#define Floor floor
#define Ceil ceil

#else
//floats
#define scalar float
#define ncscalar ncFloat
#define cur_norm curand_normal
#define Cos cosf
#define Sin sinf
#define Floor floorf
#define Ceil ceilf
#endif

//!dVec is an array whose length matches the dimension of the system
class dVec
    {
    public:
        HOSTDEVICE dVec(){};
        HOSTDEVICE dVec(const scalar value)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = value;
            };
        HOSTDEVICE dVec(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = other.x[dd];
            };

        scalar x[DIMENSION];

        //mutating operators
        dVec& operator=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] = other.x[dd];
            return *this;
            }
        dVec& operator-=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] -= other.x[dd];
            return *this;
            }
        dVec& operator+=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] += other.x[dd];
            return *this;
            }
    };

//!Less than operator for dVecs just sorts by the x-coordinate
HOSTDEVICE bool operator<(const dVec &a, const dVec &b)
    {
    return a.x[0]<b.x[0];
    }

//!Equality operator tests for.... equality of all elements
HOSTDEVICE bool operator==(const dVec &a, const dVec &b)
    {
    for (int dd = 0; dd <DIMENSION; ++dd)
        if(a.x[dd]!= b.x[dd]) return false;
    return true;
    }

//!return a dVec with all elements equal to one number
HOSTDEVICE dVec make_dVec(scalar value)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = value;
    return ans;
    }

//!component-wise addition of two dVecs
HOSTDEVICE dVec operator+(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]+b.x[dd];
    return ans;
    }

//!component-wise subtraction of two dVecs
HOSTDEVICE dVec operator-(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]-b.x[dd];
    return ans;
    }

//!component-wise multiplication of two dVecs
HOSTDEVICE dVec operator*(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]*b.x[dd];
    return ans;
    }

//!multiplication of dVec by scalar
HOSTDEVICE dVec operator*(const scalar &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a*b.x[dd];
    return ans;
    }

//!print a dVec to screen
inline __attribute__((always_inline)) void printdVec(dVec a)
    {
    for (int dd = 0; dd < DIMENSION; ++dd)
        cout << a.x[dd] <<"\t";
    cout << endl;
    };

//!Handle errors in kernel calls...returns file and line numbers if cudaSuccess doesn't pan out
static void HandleError(cudaError_t err, const char *file, int line)
    {
    //as an additional debugging check, if always synchronize cuda threads after every kernel call
    #ifdef CUDATHREADSYNC
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

//A macro to wrap cuda calls
#define HANDLE_ERROR(err) (HandleError( err, __FILE__,__LINE__ ))
//A macro to say code needs to be written
#define UNWRITTENCODE(message) (unwrittenCode(message,__FILE__,__LINE__))
//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);

#undef HOSTDEVICE
#endif
