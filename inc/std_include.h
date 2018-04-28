#ifndef STDINCLUDE
#define STDINCLUDE

/*! \file std_include.h
a file of convenience, carrying all kinds of (often unneeded) standard header files
It does also define Dscalars as either floats or doubles, depending on
how the program is compiled
*/

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

//HOSTDEVICE here is just an inliner... more important when mixing in CUDA code, where
//it might mean something different
#define HOSTDEVICE inline __attribute__((always_inline))

using namespace std;

#define PI 3.14159265358979323846

typedef double Dscalar;

//Dscalar2's, like CUDA double2's, have an x and y accessible part
struct Dscalar2
    {
    Dscalar x;
    Dscalar y;
    };

struct Dscalar4
    {
    Dscalar x;
    Dscalar y;
    Dscalar z;
    Dscalar w;
    };

//!Less than operator for Dscalars just sorts by the x-coordinate
HOSTDEVICE bool operator<(const Dscalar2 &a, const Dscalar2 &b)
    {
    return a.x<b.x;
    }

//!Equality operator tests for.... equality of both elements
HOSTDEVICE bool operator==(const Dscalar2 &a, const Dscalar2 &b)
    {
    return (a.x==b.x &&a.y==b.y);
    }

//!return a Dscalar2 from two Dscalars
HOSTDEVICE Dscalar2 make_Dscalar2(Dscalar x, Dscalar y)
    {
    Dscalar2 ans;
    ans.x =x;
    ans.y=y;
    return ans;
    }


//!component-wise addition of two Dscalar2s
HOSTDEVICE Dscalar2 operator+(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x+b.x,a.y+b.y);
    }

//!component-wise subtraction of two Dscalar2s
HOSTDEVICE Dscalar2 operator-(const Dscalar2 &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a.x-b.x,a.y-b.y);
    }

//!multiplication of Dscalar2 by Dscalar
HOSTDEVICE Dscalar2 operator*(const Dscalar &a, const Dscalar2 &b)
    {
    return make_Dscalar2(a*b.x,a*b.y);
    }

//!return a Dscalar4 from four Dscalars
HOSTDEVICE Dscalar4 make_Dscalar4(Dscalar x, Dscalar y, Dscalar z, Dscalar w)
    {
    Dscalar4 ans;
    ans.x =x;
    ans.y=y;
    ans.z=z;
    ans.w=w;
    return ans;
    }

//!print a Dscalar2 to screen
HOSTDEVICE void printDscalar2(Dscalar2 a)
    {
    printf("%f\t%f\n",a.x,a.y);
    };

//!A utility function for checking if a file exists
inline bool fileExists(const std::string& name)
    {
    ifstream f(name.c_str());
    return f.good();
    }

//spot-checking of code for debugging
#define DEBUGCODEHELPER printf("\nReached: file %s at line %d\n",__FILE__,__LINE__);

#endif
