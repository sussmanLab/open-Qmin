#include "liquidCrystalElasticity.cuh"
/*! \file liquidCrystalElasticity.cu */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

__global__ void gpu_LdG_kernel(
                               int N)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    };

bool gpu_LdG_energy(
                    int N)
    {
    unsigned int block_size = 128;
    unsigned int nblocks = N/block_size+1;
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }
/** @} */ //end of group declaration
