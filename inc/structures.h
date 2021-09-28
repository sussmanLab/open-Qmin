#ifndef structures_H
#define structures_H

/*! \file structures.h
*/

/*
 This stub was useful in past versions of the code; currently deprecated and, hence, empty
 */

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#undef HOSTDEVICE
#endif
