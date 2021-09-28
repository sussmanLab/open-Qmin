#ifndef INDEXER_H
#define INDEXER_H
/*
This file is based on part of the HOOMD-blue project, released under the BSD 3-Clause License:

HOOMD-blue Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions, and the following disclaimer both in the code and prominently in any materials provided with the distribution.
3. Neither the name ofthe copyright holder nor the names of its contributors may be used to enorse or promote products derived from this software without specific prior written permission

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//As you might suspect from the above, the classes and structures in this file are modifications of the Index1D.h file from the HOOMD-Blue package.
//Credit to Joshua A. Anderson

#include "functions.h"
#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file indexer.h */
//!Switch between a 2D array to a flattened, 1D index
/*!
 * A class for converting between a 2d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class Index2D
    {
    public:
        HOSTDEVICE Index2D(unsigned int w=0) : width(w), height(w) {}
        HOSTDEVICE Index2D(unsigned int w, unsigned int h) : width(w), height(h) {}

        HOSTDEVICE unsigned int operator()(unsigned int i, unsigned int j) const
            {
            return j*width + i;
            }
        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return width*height;
            }

        //!Get the width
        HOSTDEVICE unsigned int getW() const
            {
            return width;
            }

        //!get the height
        HOSTDEVICE unsigned int getH() const
            {
            return height;
            }

        unsigned int width;   //!< array width
        unsigned int height;   //!< array height
    };

//!Switch between a 3-dimensional grid to a flattened, 1D index
/*!
 * A class for converting between a 3d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class Index3D
    {
    public:
        HOSTDEVICE Index3D(unsigned int w=0){setSizes(w);};
        HOSTDEVICE Index3D(int3 w){setSizes(w);};

        HOSTDEVICE void setSizes(unsigned int w)
            {
            int3 W;
            W.x=w;W.y=w;W.z=w;
            setSizes(W);
            };
        HOSTDEVICE void setSizes(int3 w)
            {
            sizes.x = w.x;sizes.y = w.y;sizes.z = w.z;
            numberOfElements = sizes.x * sizes.y * sizes.z;
            intermediateSizes.x=1;
            intermediateSizes.y = intermediateSizes.x*sizes.x;
            intermediateSizes.z = intermediateSizes.y*sizes.y;
            };

        HOSTDEVICE unsigned int operator()(const int x, const int y, const int z) const
            {
            return x*intermediateSizes.x + y*intermediateSizes.y + z*intermediateSizes.z;
            };

        HOSTDEVICE unsigned int operator()(const int3 &i) const
            {
            return i.x*intermediateSizes.x + i.y*intermediateSizes.y + i.z*intermediateSizes.z;
            };

        //!What iVec would correspond to a given unsigned int IndexDD(iVec)
        HOSTDEVICE int3 inverseIndex(int i)
            {
            int3 ans;
            int z0 = i;
            ans.x = z0%sizes.x;
            z0= (z0-ans.x)/sizes.x;
            ans.y = z0%sizes.y;
            z0=(z0-ans.y)/sizes.y;
            ans.z = z0%sizes.z;
            return ans;
            };

        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return numberOfElements;
            };

        //!Get the iVec of sizes
        HOSTDEVICE int3 getSizes() const
            {
            return sizes;
            };

        int3 sizes; //!< a list of the size of the full array in each of the d dimensions
        int3 intermediateSizes; //!<intermediateSizes[a] = Product_{d<=a} sizes. intermediateSizes[0]=1;
        unsigned int numberOfElements; //! The total number of elements that the indexer can index
        unsigned int width;   //!< array width
    };

//!Switch between a d-dimensional grid to a flattened, 1D index
/*!
 * A class for converting between a 2d index and a 1-d array, which makes calculation on
 * the GPU a bit easier. This was inspired by the indexer class of Hoomd-blue
 */
class IndexDD
    {
    public:
        HOSTDEVICE IndexDD(unsigned int w=0){setSizes(w);};
        HOSTDEVICE IndexDD(iVec w){setSizes(w);};

        HOSTDEVICE void setSizes(unsigned int w)
            {
            sizes.x[0] = w;
            numberOfElements=sizes.x[0];
            intermediateSizes.x[0]=1;
            for (int dd = 1; dd < DIMENSION; ++dd)
                {
                sizes.x[dd] = w;
                intermediateSizes.x[dd] = intermediateSizes.x[dd-1]*sizes.x[dd];
                numberOfElements *= sizes.x[dd];
                };
            };
        HOSTDEVICE void setSizes(iVec w)
            {
            sizes.x[0] = w.x[0];
            numberOfElements=sizes.x[0];
            intermediateSizes.x[0]=1;
            for (int dd = 1; dd < DIMENSION; ++dd)
                {
                sizes.x[dd] = w.x[dd];
                intermediateSizes.x[dd] = intermediateSizes.x[dd-1]*sizes.x[dd];
                numberOfElements *= sizes.x[dd];
                };
            };

        HOSTDEVICE unsigned int operator()(const iVec &i) const
            {
            return dot(i,intermediateSizes);
            };

        //!What iVec would correspond to a given unsigned int IndexDD(iVec)
        HOSTDEVICE iVec inverseIndex(int i)
            {
            iVec ans;
            int z0 = i;
            for (int dd = 0; dd < DIMENSION; ++dd)
                {
                ans.x[dd] = z0 % sizes.x[dd];
                z0 = (z0 - ans.x[dd])/sizes.x[dd];
                };
            return ans;
            };

        //!Return the number of elements that the indexer can index
        HOSTDEVICE unsigned int getNumElements() const
            {
            return numberOfElements;
            };

        //!Get the iVec of sizes
        HOSTDEVICE iVec getSizes() const
            {
            return sizes;
            };

        iVec sizes; //!< a list of the size of the full array in each of the d dimensions
        iVec intermediateSizes; //!<intermediateSizes[a] = Product_{d<=a} sizes. intermediateSizes[0]=1;
        unsigned int numberOfElements; //! The total number of elements that the indexer can index
        unsigned int width;   //!< array width
    };
#undef HOSTDEVICE
#endif
