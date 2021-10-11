#include "utilities.cuh"
#include "functions.h"


#define FULL_MASK 0xffffffff

/*! \file utilities.cu
  defines kernel callers and kernels for some simple GPU array calculations

 \addtogroup utilityKernels
 @{
 */

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            mySum += __shfl_down_sync(FULL_MASK,mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

template <class T>
void reduce(int size, int threads, int blocks,
            T *d_idata, T *d_odata)
    {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    switch (threads)
        {
        case 512:
            reduce6<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case 256:
            reduce6<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case 128:
            reduce6<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case 64:
            reduce6<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case 32:
            reduce6<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case 16:
            reduce6<T,16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case  8:
            reduce6<T,8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case  4:
            reduce6<T,4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case  2:
            reduce6<T,2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        case  1:
            reduce6<T,1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
        }
    }


////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T gpuReduction(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  T *d_idata,
                  T *d_odata)
    {
    T gpu_result = 0;
    bool needReadBack = true;
    int cpuFinalThreshold = 1;
    gpu_result = 0;

    // execute the kernel
    reduce<T>(n, numThreads, numBlocks, d_idata, d_odata);
    HANDLE_ERROR(cudaGetLastError());

    //T *h_odata = (T *) malloc(numBlocks*sizeof(T));
    // sum partial block sums on GPU
    int s=numBlocks;
    while (s > cpuFinalThreshold)
        {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
        reduce<T>(s, threads, blocks, d_odata, d_odata);
        s = (s + (threads*2-1)) / (threads*2);
        }
        /*
        if (s > 1)
            {
            // copy result from device to host
            cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost);
            for (int i=0; i < s; i++)
                {
                gpu_result += h_odata[i];
                }
            needReadBack = false;
            }
        */
    // copy final sum from device to host
    if (needReadBack)
        cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
    HANDLE_ERROR(cudaGetLastError());
    //free(h_odata);
    return gpu_result;
    }

/*!
add the first N elements of array and put it in output[helperIdx]
*/
__global__ void gpu_serial_reduction_kernel(scalar *array, scalar *output, int helperIdx,int N)
    {
    scalar ans = 0.0;
    for (int i = 0; i < N; ++i)
        ans += array[i];
    output[helperIdx] = ans;
    return;
    };

/*!
add the first N elements of array and put it in output[helperIdx]...use shared memory a bit
*/
__global__ void gpu_serial_reduction_kernel2(scalar *array, scalar *output, int helperIdx,int N)
    {
    int tidx = threadIdx.x;
    extern __shared__ scalar partialSum[];

    partialSum[tidx] = 0.0;
    __syncthreads();
    int max = N/ blockDim.x+1;
    for (int i = 0; i < max;++i)
        {
        int pos =  blockDim.x *i+tidx;
        if(pos > N) continue;
        partialSum[tidx] += array[pos];
        }
    __syncthreads();
    if(tidx ==0)
        {
        scalar ans =0.0;
        for (int i = 0; i <  blockDim.x; ++i)
            ans += partialSum[i];
        output[helperIdx] = ans;
        }

    return;
    };

/*!
perform a block reduction, storing the partial sums of input into output
*/
__global__ void gpu_parallel_block_reduction_kernel(scalar *input, scalar *output,int N)
    {
    extern __shared__ scalar sharedArray[];

    unsigned int tidx = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    //load into shared memory and synchronize
    if(i < N)
        sharedArray[tidx] = input[i];
    else
        sharedArray[tidx] = 0.0;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2; s>0; s>>=1)
        {
        if (tidx < s)
            sharedArray[tidx] += sharedArray[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sharedArray[0];
    };

/*!
a slight optimization of the previous block reduction, c.f. M. Harris presentation
*/
__global__ void gpu_parallel_block_reduction2_kernel(scalar *input, scalar *output,int N)
    {
    extern __shared__ scalar sharedArray[];
    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    scalar sum;
    //load into shared memory and synchronize
    if(i < N)
        sum = input[i];
    else
        sum = 0.0;
    if(i + blockDim.x < N)
        sum += input[i+blockDim.x];
    sharedArray[tidx] = sum;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2; s>0; s>>=1)
        {
        if (tidx < s)
            sharedArray[tidx] = sum = sum+sharedArray[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sum;
    };

/*!
  multiple loads and loop unrolling...
a slight optimization of the previous block reduction, c.f. M. Harris presentation
*/
__global__ void gpu_parallel_block_reduction3_kernel(scalar *input, scalar *output,int N)
    {
    extern __shared__ scalar sharedArray[];
    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    if(i+blockDim.x < N)
        sharedArray[tidx] = input[i]+input[i+blockDim.x];
    else if(i < N)
        sharedArray[tidx] = input[i];
    else
        sharedArray[tidx] = 0.0;
    __syncthreads();

    //reduce
    for (int stride = blockDim.x/2;stride >32; stride >>=1)
        {
        if(tidx<stride)
            sharedArray[tidx] += sharedArray[tidx+stride];
        __syncthreads();
        }
    if(tidx < 32)
        {
        sharedArray[tidx] += sharedArray[tidx+32];
        sharedArray[tidx] += sharedArray[tidx+16];
        sharedArray[tidx] += sharedArray[tidx+8];
        sharedArray[tidx] += sharedArray[tidx+4];
        sharedArray[tidx] += sharedArray[tidx+2];
        sharedArray[tidx] += sharedArray[tidx+1];
        }
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sharedArray[0];
    };

/*!
Store the dot product of two dVecs in a scalar vec
*/
__global__ void gpu_vec_dot_product_kernel(dVec *input1, dVec *input2, scalar *output,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    output[idx] = dot(input1[idx],input2[idx]);
    return;
    };


/*!
Store the dot product of two 5-component covectors in a scalar vector
*/
__global__ void gpu_vec_covectorDotProduct_kernel(dVec *input, scalar *output,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    output[idx] =   (4./3.)*input[idx][0]*input[idx][0] 
                  + input[idx][1]*input[idx][1] 
                  + input[idx][2]*input[idx][2]
                  + (4./3.)*input[idx][3]*input[idx][3] 
                  + input[idx][4]*input[idx][4] 
                  - (4./3.)*input[idx][0]*input[idx][3];
    return;
    }
/*!
Store the dot product of two 5-component vectors in a scalar vector
*/
__global__ void gpu_vec_vectorDotProduct_kernel(dVec *input1, dVec *input2, scalar *output,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    output[idx] = input1[idx][0]*input2[idx][0] 
                  + input1[idx][1]*input2[idx][1] 
                  + input1[idx][2]*input2[idx][2]
                  + input1[idx][3]*input2[idx][3] 
                  + input1[idx][4]*input2[idx][4] 
                  + input1[idx][0]*input2[idx][3];
    return;
    }
/*!
Store the dot product of two 5-component vectors in a scalar vector
*/
__global__ void gpu_vec_vectorDotProduct_kernel(dVec *input, scalar *output,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    output[idx] = input[idx][0]*input[idx][0] 
                  + input[idx][1]*input[idx][1] 
                  + input[idx][2]*input[idx][2]
                  + input[idx][3]*input[idx][3] 
                  + input[idx][4]*input[idx][4] 
                  + input[idx][0]*input[idx][3];
    return;
    }

/*!
Store the dot product of two dVecs in a scalar vec, unrolled by dimension
*/
__global__ void gpu_vec_dot_product_unrolled_kernel(dVec *input1, dVec *input2, scalar *output,int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int p1 = idx / DIMENSION;
    int d1 = idx % DIMENSION;
    if (p1 >= N)
        return;
    output[idx] = input1[p1][d1]*input2[p1][d1];
    return;
    };


/*!
This kernel basically performs the operation of the "reduction2" kernel, but the shared memory gets
dot products...BROKEN
*/
__global__ void gpu_dVec_dot_products_kernel(dVec *input1, dVec *input2, scalar *output,int N)
    {
    extern __shared__ scalar sharedArray[];
    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    scalar tempSum;
    if(i < N)
        tempSum = dot(input1[i],input2[i]);
    else
        tempSum = 0.0;
    sharedArray[tidx] = 0.0;
    __syncthreads();

    //reduce
    for (int s = blockDim.x/2;s>0; s>>=1)
        {
        if (tidx <s)
            sharedArray[tidx] = tempSum = tempSum+sharedArray[tidx+s];
        __syncthreads();
        };
    //write to the correct block of the output array
    if(tidx==0)
        output[blockIdx.x] = tempSum;
    };

/*!
This kernel basically performs the operation of the "reduction2" kernel, but the shared memory gets dot products
*/
__global__ void gpu_unrolled_dVec_dot_products_kernel(dVec *input1, dVec *input2, scalar *output,int N)
    {
    extern __shared__ scalar sharedArray[];
    unsigned int tidx = threadIdx.x;
    unsigned int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    int p1 = i / DIMENSION;
    int d1 = i % DIMENSION;
    int p2 = (i+blockDim.x) / DIMENSION;
    int d2 = (i+blockDim.x) % DIMENSION;

    if(i+blockDim.x < N)
        sharedArray[tidx] = input1[p1][d1]*input2[p1][d1] + input1[p2][d2]*input2[p2][d2];
    else if(i < N)
        sharedArray[tidx] = input1[p1][d1]*input2[p1][d1];
    else
        sharedArray[tidx] = 0.0;
    __syncthreads();

    //reduce
    for (int stride = blockDim.x/2;stride >32; stride >>=1)
        {
        if(tidx<stride)
            sharedArray[tidx] += sharedArray[tidx+stride];
        __syncthreads();
        }
    if(tidx < 32)
        {
        sharedArray[tidx] += sharedArray[tidx+32];
        sharedArray[tidx] += sharedArray[tidx+16];
        sharedArray[tidx] += sharedArray[tidx+8];
        sharedArray[tidx] += sharedArray[tidx+4];
        sharedArray[tidx] += sharedArray[tidx+2];
        sharedArray[tidx] += sharedArray[tidx+1];
        }
    //write to the correct block of the output array
    if (tidx==0)
        output[blockIdx.x] = sharedArray[0];
    };

/*!
take a vector of dVecs, a vector of scalars, a factor, and return a vector where
every entry is
factor*scalar[i]*(dVec[i])^2
*/
__global__ void gpu_scalar_times_dVec_squared_kernel(dVec *d_vec1, scalar *d_scalars, scalar factor, scalar *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = factor * d_scalars[idx]*dot(d_vec1[idx],d_vec1[idx]);
    };
/*!
take two vectors of dVecs and return a vector of scalars, where each entry is vec1[i].vec2[i]
*/
__global__ void gpu_dot_dVec_vectors_kernel(dVec *d_vec1, dVec *d_vec2, scalar *d_ans, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = dot(d_vec1[idx],d_vec2[idx]);
    };
/*!
  multiply every element of an array of dVecs by the same scalar
  */
__global__ void gpu_dVec_times_scalar_kernel(dVec *d_vec1,scalar factor, int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_vec1[idx] = factor*d_vec1[idx];
    };
/*!
  multiply every element of an array of dVecs by the same scalar
  */
__global__ void gpu_dVec_times_scalar_kernel(dVec *d_vec1,scalar factor, dVec *d_ans,int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    d_ans[idx] = factor*d_vec1[idx];
    };


__global__ void gpu_dVec_plusEqual_dVec_kernel(dVec *d_vec1,dVec *d_vec2,scalar factor,int n)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n)
        return;
    int pIdx = idx / DIMENSION;
    int dIdx = idx % DIMENSION;

    d_vec1[pIdx][dIdx] += factor*d_vec2[pIdx][dIdx];
    };



/////
//Kernel callers
///

bool gpu_dVec_plusEqual_dVec(dVec *d_vec1,
                              dVec *d_vec2,
                              scalar factor,
                              int N,
                              int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = (DIMENSION*N)/block_size + 1;
    gpu_dVec_plusEqual_dVec_kernel<<<nblocks,block_size>>>(d_vec1,d_vec2,factor,DIMENSION*N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };
/*!
\param d_vec1 dVec input array
\param factor scalar multiplication factor
\param N      the length of the arrays
\post d_vec1 *= factor for every element
 */
bool gpu_dVec_times_scalar(dVec *d_vec1, scalar factor, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dVec_times_scalar_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                factor,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_dVec_times_scalar(dVec *d_vec1, scalar factor, dVec *d_ans,int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dVec_times_scalar_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                factor,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_scalar_times_dVec_squared(dVec *d_vec1, scalar *d_scalars, scalar factor, scalar *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_scalar_times_dVec_squared_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                d_scalars,
                                                factor,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
\param d_vec1 dVec input array
\param d_vec2 dVec input array
\param d_ans  scalar output array... d_ans[idx] = d_vec1[idx].d_vec2[idx]
\param N      the length of the arrays
\post d_ans = d_vec1.d_vec2
*/
bool gpu_dot_dVec_vectors(dVec *d_vec1, dVec *d_vec2, scalar *d_ans, int N)
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = N/block_size + 1;
    gpu_dot_dVec_vectors_kernel<<<nblocks,block_size>>>(
                                                d_vec1,
                                                d_vec2,
                                                d_ans,
                                                N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

scalar gpu_gpuarray_QT_covector_dot_product(
                        GPUArray<dVec> &input1,
                        GPUArray<scalar> &intermediate,
                        GPUArray<scalar> &intermediate2,
                        int N,
                        int block_size)
    {
    if (N == 0)
        N = input1.getNumElements();
    unsigned int nblocks  = N/block_size + 1;
    GPUArray<scalar> ans(1,false,false);
    if(intermediate.getNumElements() <N)
        intermediate.resize(N);
    if(intermediate2.getNumElements() <N)
        intermediate2.resize(N);
    scalar result = 0;
        //scope for array handles
        {
        ArrayHandle<dVec> i1(input1,access_location::device,access_mode::read);
        ArrayHandle<scalar> inter1(intermediate,access_location::device,access_mode::overwrite);
        gpu_vec_covectorDotProduct_kernel<<<nblocks,block_size>>>(i1.data,inter1.data,N);
        HANDLE_ERROR(cudaGetLastError());

        int numBlocks = 0;
        int numThreads = 0;
        int maxBlocks = 64;
        int maxThreads = 256;
        ArrayHandle<scalar> inter2(intermediate2,access_location::device,access_mode::overwrite);
        getNumBlocksAndThreads(N, maxBlocks, maxThreads, numBlocks, numThreads);
        result = gpuReduction(N,numThreads,numBlocks,maxThreads,maxBlocks,inter1.data,inter2.data);
        return result;
        }
    };

scalar gpu_gpuarray_QT_vector_dot_product(
                        GPUArray<dVec> &input1,
                        GPUArray<dVec> &input2,
                        GPUArray<scalar> &intermediate,
                        GPUArray<scalar> &intermediate2,
                        int N,
                        int block_size)
    {
    if (N == 0)
        N = input1.getNumElements();
    unsigned int nblocks  = N/block_size + 1;
    GPUArray<scalar> ans(1,false,false);
    if(intermediate.getNumElements() <N)
        intermediate.resize(N);
    if(intermediate2.getNumElements() <N)
        intermediate2.resize(N);
    scalar result = 0;
        //scope for array handles
        {
        ArrayHandle<dVec> i1(input1,access_location::device,access_mode::read);
        ArrayHandle<dVec> i2(input2,access_location::device,access_mode::read);
        ArrayHandle<scalar> inter1(intermediate,access_location::device,access_mode::overwrite);
        gpu_vec_vectorDotProduct_kernel<<<nblocks,block_size>>>(i1.data,i2.data,inter1.data,N);
        HANDLE_ERROR(cudaGetLastError());

        int numBlocks = 0;
        int numThreads = 0;
        int maxBlocks = 64;
        int maxThreads = 256;
        ArrayHandle<scalar> inter2(intermediate2,access_location::device,access_mode::overwrite);
        getNumBlocksAndThreads(N, maxBlocks, maxThreads, numBlocks, numThreads);
        result = gpuReduction(N,numThreads,numBlocks,maxThreads,maxBlocks,inter1.data,inter2.data);
        return result;
        }
    };
scalar gpu_gpuarray_QT_vector_dot_product(
                        GPUArray<dVec> &input1,
                        GPUArray<scalar> &intermediate,
                        GPUArray<scalar> &intermediate2,
                        int N,
                        int block_size)
    {
    if (N == 0)
        N = input1.getNumElements();
    unsigned int nblocks  = N/block_size + 1;
    GPUArray<scalar> ans(1,false,false);
    if(intermediate.getNumElements() <N)
        intermediate.resize(N);
    if(intermediate2.getNumElements() <N)
        intermediate2.resize(N);
    scalar result = 0;
        //scope for array handles
        {
        ArrayHandle<dVec> i1(input1,access_location::device,access_mode::read);
        ArrayHandle<scalar> inter1(intermediate,access_location::device,access_mode::overwrite);
        gpu_vec_vectorDotProduct_kernel<<<nblocks,block_size>>>(i1.data,inter1.data,N);
        HANDLE_ERROR(cudaGetLastError());

        int numBlocks = 0;
        int numThreads = 0;
        int maxBlocks = 64;
        int maxThreads = 256;
        ArrayHandle<scalar> inter2(intermediate2,access_location::device,access_mode::overwrite);
        getNumBlocksAndThreads(N, maxBlocks, maxThreads, numBlocks, numThreads);
        result = gpuReduction(N,numThreads,numBlocks,maxThreads,maxBlocks,inter1.data,inter2.data);
        return result;
        }
    };

scalar gpu_gpuarray_dVec_dot_products(
                        GPUArray<dVec> &input1,
                        GPUArray<dVec> &input2,
                        GPUArray<scalar> &intermediate,
                        GPUArray<scalar> &intermediate2,
                        int N,
                        int block_size)
    {
    if (N == 0)
        N = input1.getNumElements();
    int Nd = DIMENSION*N;
    unsigned int nblocks  = N/block_size + 1;
    GPUArray<scalar> ans(1,false,false);
    if(intermediate.getNumElements() <Nd)
        intermediate.resize(Nd);
    if(intermediate2.getNumElements() <Nd)
        intermediate2.resize(Nd);
    scalar result = 0;
    if(true) // for testing...switch to best reduction kernel
        {
        ArrayHandle<dVec> i1(input1,access_location::device,access_mode::read);
        ArrayHandle<dVec> i2(input2,access_location::device,access_mode::read);
        ArrayHandle<scalar> inter1(intermediate,access_location::device,access_mode::overwrite);
        ArrayHandle<scalar> inter2(intermediate2,access_location::device,access_mode::overwrite);
        gpu_vec_dot_product_unrolled_kernel<<<nblocks,block_size>>>(i1.data,i2.data,inter1.data,N);
        HANDLE_ERROR(cudaGetLastError());
        int numBlocks = 0;
        int numThreads = 0;
        int maxBlocks = 64;
        int maxThreads = 256;
        getNumBlocksAndThreads(Nd, maxBlocks, maxThreads, numBlocks, numThreads);
        result = gpuReduction(Nd,numThreads,numBlocks,maxThreads,maxBlocks,inter1.data,inter2.data);
        return result;
        }
    else
        {
        {
        ArrayHandle<dVec> i1(input1,access_location::device,access_mode::read);
        ArrayHandle<dVec> i2(input2,access_location::device,access_mode::read);
        ArrayHandle<scalar> inter1(intermediate,access_location::device,access_mode::overwrite);
        ArrayHandle<scalar> inter2(intermediate2,access_location::device,access_mode::overwrite);
        ArrayHandle<scalar> answer(ans,access_location::device,access_mode::overwrite);
        gpu_dVec_dot_products(i1.data,i2.data,inter1.data,inter2.data,answer.data,0,N,block_size);
        }
        ArrayHandle<scalar> answer(ans,access_location::host,access_mode::read);
        return answer.data[0];
        }
    }

/*!
takes the dot product of every element of the two input arrays and performs a reduction on the sum
\param input1 vector 1...wow!
\param input2 vector 2...wow!
\param intermediate an array that input is dot producted to
\param intermediate2 an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
\param block_size the...block size. doxygen is annoying sometimes
*/
bool gpu_dVec_dot_products(dVec *input1,dVec *input2, scalar *intermediate, scalar *intermediate2,scalar *output, int helperIdx, int N,int block_size)
    {
    //int problemSize = DIMENSION*N;
    //unsigned int nblocks  = problemSize/block_size + 1;
    unsigned int nblocks  = N/block_size + 1;

    //first dot the vectors together
    gpu_vec_dot_product_kernel<<<nblocks,block_size>>>(input1,input2,intermediate,N);
    HANDLE_ERROR(cudaGetLastError());

    //then call the parallel reduction routine to sum up the answer
    gpu_parallel_reduction(intermediate,intermediate2,output,helperIdx,N,block_size);
    //gpu_serial_reduction(intermediate,output,helperIdx,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    /*
    HANDLE_ERROR(cudaGetLastError());
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(scalar);
    //Do a block reduction of the input array
    //gpu_unrolled_dVec_dot_products_kernel<<<nblocks,block_size,smem>>>(input1,input2,intermediate, problemSize);
    gpu_dVec_dot_products_kernel<<<nblocks,block_size,smem>>>(input1,input2,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    int nb=1024;
    if(nblocks < nb) nb = 1;
    gpu_serial_reduction_kernel2<<<1,nb,nb*sizeof(scalar)>>>(intermediate,output,helperIdx,nblocks+1);
    HANDLE_ERROR(cudaGetLastError());
    */
    }

/*
A stub of a function...eventually replace with off-the-shelf solution?
*/
bool gpu_dVec_dot_products(dVec *input1,dVec *input2, scalar *output, int helperIdx, int N)
    {
    //scalar init = 0.0;
    //dVecDotProduct mult_op;
    //thrust::plus<scalar> add_op;
    //thrust::device_ptr<scalar> ptrAns = thrust::device_pointer_cast(output);
    //thrust::device_ptr<dVec> ptr1 = thrust::device_pointer_cast(input1);
    //thrust::device_ptr<dVec> ptr2 = thrust::device_pointer_cast(input2);
    //output[helperIdx] = thrust::inner_product(thrust::device,ptr1,ptr1+N,ptr2,init,add_op,mult_op);
    //output[helperIdx] = thrust::inner_product(thrust::device,input1,input1+N,input2,init,add_op,mult_op);
    //ptrAns[helperIdx] = thrust::inner_product(thrust::device,input1,input1+N,input2,init,add_op,mult_op);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
a two-step parallel reduction algorithm that first does a partial sum reduction of input into the
intermediate array, then launches a second kernel to sum reduce intermediate into output[helperIdx]
\param input the input array to sum
\param intermediate an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
\param block_size the...block size. doxygen is annoying sometimes
*/
bool gpu_parallel_reduction(scalar *input, scalar *intermediate, scalar *output, int helperIdx, int N,int block_size)
    {
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input

    //Do a block reduction of the input array
    //reduce(N, block_size, nblocks,input, intermediate);
    unsigned int smem = block_size*sizeof(scalar);
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    int nb=2048;
    if(nblocks < nb) nb = 1;
    gpu_serial_reduction_kernel2<<<1,nb,nb*sizeof(scalar)>>>(intermediate,output,helperIdx,nblocks+1);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
This serial reduction routine should probably never be called. It provides an interface to the
gpu_serial_reduction_kernel above that may be useful for testing
  */
bool gpu_serial_reduction(scalar *array, scalar *output, int helperIdx, int N)
    {
    gpu_serial_reduction_kernel<<<1,1>>>(array,output,helperIdx,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/*!
  A function of convenience... set an array on the device
  */
template <typename T>
__global__ void gpu_set_array_kernel(T *arr,T value, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    arr[idx] = value;
    return;
    };

template<typename T>
bool gpu_set_array(T *array, T value, int N,int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;
    gpu_set_array_kernel<<<nblocks, block_size>>>(array,value,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

template <typename T>
__global__ void gpu_copy_gpuarray_kernel(T *copyInto,T *copyFrom, int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    copyInto[idx] = copyFrom[idx];
    return;
    };

template<typename T>
bool gpu_copy_gpuarray(GPUArray<T> &copyInto,GPUArray<T> &copyFrom,int maxBlockSize)
    {
    int N = copyFrom.getNumElements();
    if(copyInto.getNumElements() < N)
        copyInto.resize(N);
    unsigned int block_size = maxBlockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = (N)/block_size + 1;
    ArrayHandle<T> ci(copyInto,access_location::device,access_mode::overwrite);
    ArrayHandle<T> cf(copyFrom,access_location::device,access_mode::read);
    gpu_copy_gpuarray_kernel<<<nblocks,block_size>>>(ci.data,cf.data,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

scalar host_dVec_dot_products(dVec *input1,dVec *input2,int N)
    {
    scalar ans = 0.0;
    for (int ii = 0; ii < N; ++ii)
        for (int dd = 0; dd < DIMENSION; ++dd)
            ans +=input1[ii][dd]*input2[ii][dd];
    return ans;
    }

void host_dVec_plusEqual_dVec(dVec *d_vec1,dVec *d_vec2,scalar factor,int N)
    {
    for (int ii = 0; ii < N; ++ii)
        d_vec1[ii] = d_vec1[ii] + factor*d_vec2[ii];
    }

void host_dVec_times_scalar(dVec *d_vec1, scalar factor, dVec *d_ans, int N)
    {
    for(int ii = 0; ii < N; ++ii)
        d_ans[ii] = factor*d_vec1[ii];
    }
//explicit template instantiations
template scalar gpuReduction<scalar>(int  n,int  numThreads,int  numBlocks,int  maxThreads,int  maxBlocks,scalar *d_idata,scalar *d_odata);
template int gpuReduction<int>(int  n,int  numThreads,int  numBlocks,int  maxThreads,int  maxBlocks,int *d_idata,int *d_odata);
template void reduce<int>(int size, int threads, int blocks, int *d_idata, int *d_odata);
template void reduce<scalar>(int size, int threads, int blocks, scalar *d_idata, scalar *d_odata);

template bool gpu_copy_gpuarray<dVec>(GPUArray<dVec> &copyInto,GPUArray<dVec> &copyFrom,int maxBlockSize);
template bool gpu_copy_gpuarray<scalar>(GPUArray<scalar> &copyInto,GPUArray<scalar> &copyFrom,int maxBlockSize);

template bool gpu_set_array<int>(int *,int, int, int);
template bool gpu_set_array<unsigned int>(unsigned int *,unsigned int, int, int);
template bool gpu_set_array<int2>(int2 *,int2, int, int);
template bool gpu_set_array<scalar>(scalar *,scalar, int, int);
template bool gpu_set_array<dVec>(dVec *,dVec, int, int);
template bool gpu_set_array<cubicLatticeDerivativeVector>(cubicLatticeDerivativeVector *,cubicLatticeDerivativeVector, int, int);
/** @} */ //end of group declaration
