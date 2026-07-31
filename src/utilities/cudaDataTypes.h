#ifndef CUDADATATYPES_H
#define CUDADATATYPES_H

// define int, float, double, 2-4. Then define sensible operators for all of them
#ifdef ENABLE_CUDA 
#include <vector_types.h>
#else

struct int2
    {
    int x, y;
    };
struct int3
    {
    int x, y, z;
    };
struct int4
    {
    int x, y, z, w;
    };

struct float2
    {
    float x, y;
    };
struct float3
    {
    float x, y, z;
    };
struct float4
    {
    float x, y, z, w;
    };

struct double2
    {
    double x, y;
    };
struct double3
    {
    double x, y, z;
    };
struct double4
    {
    double x, y, z, w;
    };

#endif


int2 operator+(const int2& a, const int2& b);
int2 operator-(const int2& a, const int2& b);
int2 operator*(const int2& a, const int2& b);

float2 operator+(const float2& a, const float2& b);
float2 operator-(const float2& a, const float2& b);
float2 operator*(const float2& a, const float2& b);

double2 operator+(const double2& a, const double2& b);
double2 operator-(const double2& a, const double2& b);
double2 operator*(const double2& a, const double2& b);

int3 operator+(const int3& a, const int3& b);
int3 operator-(const int3& a, const int3& b);
int3 operator*(const int3& a, const int3& b);

float3 operator+(const float3& a, const float3& b);
float3 operator-(const float3& a, const float3& b);
float3 operator*(const float3& a, const float3& b);

double3 operator+(const double3& a, const double3& b);
double3 operator-(const double3& a, const double3& b);
double3 operator*(const double3& a, const double3& b);

int4 operator+(const int4& a, const int4& b);
int4 operator-(const int4& a, const int4& b);
int4 operator*(const int4& a, const int4& b);

float4 operator+(const float4& a, const float4& b);
float4 operator-(const float4& a, const float4& b);
float4 operator*(const float4& a, const float4& b);

double4 operator+(const double4& a, const double4& b);
double4 operator-(const double4& a, const double4& b);
double4 operator*(const double4& a, const double4& b);


#endif
