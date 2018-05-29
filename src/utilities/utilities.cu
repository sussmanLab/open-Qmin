#include "utilities.cuh"
#include "functions.h"

/*! \file utilities.cu
  defines kernel callers and kernels for some simple GPU array calculations

 \addtogroup utilityKernels
 @{
 */

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
  A function of convenience...zero out an array on the device
  */
__global__ void gpu_zero_array_kernel(dVec *arr,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    dVec temp = make_dVec(0.0);
    arr[idx] = temp;
    for (int dd = 0; dd < DIMENSION; ++dd)
        arr[idx].x[dd] = 0.0;
    return;
    };
/*!
  A function of convenience...zero out an array on the device
  */
__global__ void gpu_zero_array_kernel(scalar *arr,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    arr[idx] = 0.;
    return;
    };
/*!
  A function of convenience...zero out an array on the device
  */
__global__ void gpu_zero_array_kernel(unsigned int *arr,
                                              int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    arr[idx] = 0;
    return;
    };

/*!
  A function of convenience...zero out an array on the device
  */
__global__ void gpu_zero_array_kernel(int *arr,
                                      int N)
    {
    // read in the particle that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    arr[idx] = 0;
    return;
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

/////
//Kernel callers
///

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

bool gpu_zero_array(dVec *arr,
                    int N
                    )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_zero_array(unsigned int *arr,
                    int N
                    )
    {
    //optimize block size later
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

bool gpu_zero_array(scalar *arr,
                    int N
                    )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }
bool gpu_zero_array(int *arr,
                    int N
                    )
    {
    unsigned int block_size = 128;
    if (N < 128) block_size = 16;
    unsigned int nblocks  = N/block_size + 1;

    gpu_zero_array_kernel<<<nblocks, block_size>>>(arr,
                                                    N
                                                    );
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/*!
a two-step parallel reduction algorithm that first does a partial sum reduction of input into the
intermediate array, then launches a second kernel to sum reduce intermediate into output[helperIdx]
\param input the input array to sum
\param intermediate an array that input is block-reduced to
\param output the intermediate array will be sum reduced and stored in one of the components of output
\param helperIdx the location in output to store the answer
\param N the size of the input and  intermediate arrays
*/
bool gpu_parallel_reduction(scalar *input, scalar *intermediate, scalar *output, int helperIdx, int N)
    {
    unsigned int block_size = 256;
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(scalar);

    //Do a block reduction of the input array
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    gpu_serial_reduction_kernel<<<1,1>>>(intermediate,output,helperIdx,nblocks);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

bool gpu_parallel_reduction(scalar *input, scalar *intermediate, scalar *output, int helperIdx, int N,int block_size)
    {
    unsigned int nblocks  = N/block_size + 1;
    //first do a block reduction of input
    unsigned int smem = block_size*sizeof(scalar);

    //Do a block reduction of the input array
    gpu_parallel_block_reduction2_kernel<<<nblocks,block_size,smem>>>(input,intermediate, N);
    HANDLE_ERROR(cudaGetLastError());

    //sum reduce the temporary array, saving the result in the right slot of the output array
    gpu_serial_reduction_kernel<<<1,1>>>(intermediate,output,helperIdx,nblocks);
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

/** @} */ //end of group declaration
