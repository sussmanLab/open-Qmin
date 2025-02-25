#include "cudaDataTypes.h"

int2 operator+(const int2& a, const int2& b)
    {
    int2 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    return result;
    };
int2 operator-(const int2& a, const int2& b)
    {
    int2 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
    };
int2 operator*(const int2& a, const int2& b)
    {
    int2 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    return result;
    };

float2 operator+(const float2& a, const float2& b)
    {
    float2 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    return result;
    };
float2 operator-(const float2& a, const float2& b)
    {
    float2 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
    };
float2 operator*(const float2& a, const float2& b)
    {
    float2 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    return result;
    };

double2 operator+(const double2& a, const double2& b)
    {
    double2 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    return result;
    };
double2 operator-(const double2& a, const double2& b)
    {
    double2 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
    };
double2 operator*(const double2& a, const double2& b)
    {
    double2 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    return result;
    };

int3 operator+(const int3& a, const int3& b)
    {
    int3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
    };
int3 operator-(const int3& a, const int3& b)
    {
    int3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
    };
int3 operator*(const int3& a, const int3& b)
    {
    int3 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    return result;
    };

float3 operator+(const float3& a, const float3& b)
    {
    float3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
    };
float3 operator-(const float3& a, const float3& b)
    {
    float3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
    };
float3 operator*(const float3& a, const float3& b)
    {
    float3 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    return result;
    };

double3 operator+(const double3& a, const double3& b)
    {
    double3 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    return result;
    };
double3 operator-(const double3& a, const double3& b)
    {
    double3 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    return result;
    };
double3 operator*(const double3& a, const double3& b)
    {
    double3 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    return result;
    };

int4 operator+(const int4& a, const int4& b)
    {
    int4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
    };
int4 operator-(const int4& a, const int4& b)
    {
    int4 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = a.w - b.w;
    return result;
    };
int4 operator*(const int4& a, const int4& b)
    {
    int4 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
    return result;
    };

float4 operator+(const float4& a, const float4& b)
    {
    float4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
    };
float4 operator-(const float4& a, const float4& b)
    {
    float4 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = a.w - b.w;
    return result;
    };
float4 operator*(const float4& a, const float4& b)
    {
    float4 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
    return result;
    };

double4 operator+(const double4& a, const double4& b)
    {
    double4 result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
    return result;
    };
double4 operator-(const double4& a, const double4& b)
    {
    double4 result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = a.w - b.w;
    return result;
    };
double4 operator*(const double4& a, const double4& b)
    {
    double4 result;
    result.x = a.x * b.x;
    result.y = a.y * b.y;
    result.z = a.z * b.z;
    result.w = a.w * b.w;
    return result;
    };
