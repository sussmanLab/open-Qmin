#ifndef structures_H
#define structures_H

/*! \file structures.h
*/

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#undef HOSTDEVICE
#endif
