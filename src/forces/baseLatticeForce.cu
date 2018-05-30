#include "baseLatticeForce.cuh"
/*! \file baseLatticeForce.cu */
/** @addtogroup forceKernels force Kernels
 * @{
 * \brief CUDA kernels and callers
 */

__global__ void gpu_lattice_spin_force_nn_kernel(dVec *d_force,
                                dVec *d_spins,
                                Index3D latticeIndex,
                                scalar J,
                                int N,
                                int L,
                                bool zeroForce)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if(zeroForce)
        d_force[idx] = make_dVec(0.0);
    int3 target = latticeIndex.inverseIndex(idx);
    /*
    int smx,spx,smy,spy,smz,spz;
    smx = latticeIndex(wrap(target.x-1,L),target.y,target.z);
    spx = latticeIndex(wrap(target.x+1,L),target.y,target.z);
    smy = latticeIndex(target.x,wrap(target.y-1,L),target.z);
    spy = latticeIndex(target.x,wrap(target.y+1,L),target.z);
    smz = latticeIndex(target.x,target.y,wrap(target.z-1,L));
    spz = latticeIndex(target.x,target.y,wrap(target.z+1,L));
    if(zeroForce)
        d_force[idx] = J*(d_spins[smx]+d_spins[spx] + d_spins[smy]+d_spins[spy] + d_spins[smz]+d_spins[spz]) ;
    else
        d_force[idx] += J*(d_spins[smx]+d_spins[spx] + d_spins[smy]+d_spins[spy] + d_spins[smz]+d_spins[spz]) ;
    */
    dVec smx,spx,smy,spy,smz,spz,ans;
    smx = d_spins[latticeIndex(wrap(target.x-1,L),target.y,target.z)];
    spx = d_spins[latticeIndex(wrap(target.x+1,L),target.y,target.z)];
    smy = d_spins[latticeIndex(target.x,wrap(target.y-1,L),target.z)];
    spy = d_spins[latticeIndex(target.x,wrap(target.y+1,L),target.z)];
    smz = d_spins[latticeIndex(target.x,target.y,wrap(target.z-1,L))];
    spz = d_spins[latticeIndex(target.x,target.y,wrap(target.z+1,L))];
    ans = J*(smx+smy+smz+spx+spy+spz);
    if(zeroForce)
        d_force[idx] = ans ;
    else
        d_force[idx] += ans ;
    }

bool gpu_lattice_spin_force_nn(dVec *d_force,
                                dVec *d_spins,
                                Index3D latticeIndex,
                                scalar J,
                                int N,
                                bool zeroForce,
                                int maxBlockSize)
    {
    unsigned int block_size = maxBlockSize;
    unsigned int nblocks = N/block_size+1;

    gpu_lattice_spin_force_nn_kernel<<<nblocks,block_size>>>(d_force,d_spins,latticeIndex,J,N,
            latticeIndex.getSizes().x,zeroForce);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    }

/** @} */ //end of group declaration
