#include "energyMinimizerAdam.cuh"
/*! \file energyMinimizerAdam.cu

\addtogroup updaterKernels
@{
*/

__global__ void gpu_adam_step_kernel(dVec *force,
                   dVec *biasedMomentum,
                   dVec *biasedMomentum2,
                   dVec *correctedMomentum,
                   dVec *correctedMomentum2,
                   dVec *displacement,
                   scalar deltaT,
                   scalar beta1,
                   scalar beta2,
                   scalar beta1t,
                   scalar beta2t,
                   int N)
    {
    // read in the index that belongs to this thread
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /*
    if (idx >= N)
        return;
    scalar rootvc;
    dVec f = force[idx];
    dVec bm = biasedMomentum[idx];
    dVec bm2 = biasedMomentum2[idx];
    dVec cm, cm2,ans;
    for(int dd = 0; dd < DIMENSION; ++dd)
        {
        bm[dd] = beta1*bm[dd]+(beta1-1.)*f[dd];
        bm2[dd] = beta2*bm2[dd]+(1.-beta2)*f[dd]*f[dd];
        cm[dd] = bm[dd]*(1.0/(1.-beta1t));
        cm2[dd]=bm2[dd]*(1.0/(1.-beta2t));
        rootvc = sqrt(cm2[dd]);
        if(rootvc ==0) rootvc = 1e-10;
        ans[dd] = -deltaT*cm[dd]/(rootvc);
        }
    displacement[idx] = ans;
    biasedMomentum[idx] = bm;
    biasedMomentum2[idx] = bm2;
    correctedMomentum[idx] = cm;
    correctedMomentum2[idx] = cm2;
    */
    int pidx = idx/DIMENSION;
    if(pidx>=N) return;
    int didx = idx%DIMENSION;

    scalar f = force[pidx][didx];
    scalar bm = biasedMomentum[pidx][didx];
    scalar bm2 = biasedMomentum2[pidx][didx];
    scalar cm, cm2,ans;
    bm = beta1*bm+(beta1-1.)*f;
    bm2 = beta2*bm2+(1.-beta2)*f*f;
    cm = bm*(1.0/(1.-beta1t));
    cm2=bm2*(1.0/(1.-beta2t));
    scalar rootvc = sqrt(cm2);
    if(rootvc !=0)
        ans = -deltaT*cm/rootvc;
    else
        ans = -deltaT*cm*1e8;

    displacement[pidx][didx] = ans;
    biasedMomentum[pidx][didx] = bm;
    biasedMomentum2[pidx][didx] = bm2;
    correctedMomentum[pidx][didx] = cm;
    correctedMomentum2[pidx][didx] = cm2;
    }

/*!
A memory-efficiency optimization has each thread acting on one dimension of one degree of freedom...
  */
bool gpu_adam_step(dVec *force,
                   dVec *biasedMomentum,
                   dVec *biasedMomentum2,
                   dVec *correctedMomentum,
                   dVec *correctedMomentum2,
                   dVec *displacement,
                   scalar deltaT,
                   scalar beta1,
                   scalar beta2,
                   scalar beta1t,
                   scalar beta2t,
                   int N,
                   int blockSize)
    {
    int block_size=blockSize;
    if (N < 128) block_size = 32;
    unsigned int nblocks  = DIMENSION*N/block_size + 1;
    gpu_adam_step_kernel<<<nblocks,block_size>>>(force,biasedMomentum,biasedMomentum2,
            correctedMomentum,correctedMomentum2, displacement,
            deltaT,beta1,beta2,beta1t,beta2t,N);
    HANDLE_ERROR(cudaGetLastError());
    return cudaSuccess;
    };

/** @} */ //end of group declaration
