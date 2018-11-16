#include "cubicLattice.h"
#include "cubicLattice.cuh"
#include "functions.h"
/*! \file cubicLattice.cpp */

cubicLattice::cubicLattice(int l, bool _slice, bool _useGPU)
    {
    useGPU=_useGPU;
    latticeIndex = Index3D(l);
    sliceSites = _slice;
    N = l*l*l;
    L=l;
    Box = make_shared<periodicBoundaryConditions>(L);
    selfForceCompute = false;
    initializeNSites();
    normalizeSpins = true;
    };

cubicLattice::cubicLattice(int lx, int ly, int lz, bool _slice, bool _useGPU)
    {
    useGPU=_useGPU;
    int3 idx;
    idx.x=lx;idx.y=ly;idx.z=lz;
    latticeIndex = Index3D(idx);
    sliceSites = _slice;
    N = lx*ly*lz;
    L=lx;
    Box = make_shared<periodicBoundaryConditions>(L);//DONT USE
    selfForceCompute = false;
    initializeNSites();
    normalizeSpins = true;
    };

void cubicLattice::initializeNSites()
    {
    positions.resize(N);
    types.resize(N);
    forces.resize(N);
    //temporary?
    masses.resize(N);
    velocities.resize(N);

    vector<dVec> zeroes(N,make_dVec(0.0));
    vector<int> zeroInts(N,0);
    vector<scalar> unities(N,1.0);
    fillGPUArrayWithVector(zeroInts,types);
    fillGPUArrayWithVector(unities,masses);
    fillGPUArrayWithVector(zeroes,positions);
    fillGPUArrayWithVector(zeroes,forces);
    fillGPUArrayWithVector(zeroes,velocities);
    moveParticlesTuner = make_shared<kernelTuner>(128,1024,128,10,200000);
    };

void cubicLattice::moveParticles(GPUArray<dVec> &dofs,GPUArray<dVec> &displacements,scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(dofs);
        #include "ompParallelLoopDirective.h"
        for(int pp = 0; pp < N; ++pp)
            {
            h_pos.data[pp] += scale*h_disp.data[pp];
            if(normalizeSpins)
                {
                scalar nrm = norm(h_pos.data[pp]);
                h_pos.data[pp] = (1/nrm)*h_pos.data[pp];
                }
            }
        }
    else
        {//gpu branch
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(dofs,access_location::device,access_mode::readwrite);
        gpu_update_spins(d_disp.data,d_pos.data,scale,N,normalizeSpins);
        };
    };

void cubicLattice::moveParticles(GPUArray<dVec> &displacements, scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(positions);
        #include "ompParallelLoopDirective.h"
        for(int pp = 0; pp < N; ++pp)
            {
            h_pos.data[pp] += scale*h_disp.data[pp];
            if(normalizeSpins)
                {
                scalar nrm = norm(h_pos.data[pp]);
                h_pos.data[pp] = (1/nrm)*h_pos.data[pp];
                }
            }
        }
    else
        {//gpu branch
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        gpu_update_spins(d_disp.data,d_pos.data,scale,N,normalizeSpins);
        };
    };

void cubicLattice::setSpinsRandomly(noiseSource &noise)
    {
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        #include "ompParallelLoopDirective.h"
        for(int pp = 0; pp < N; ++pp)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                pos.data[pp][dd] = noise.getRealNormal();
            scalar lambda = dot(pos.data[pp],pos.data[pp]);
            pos.data[pp] = (1/sqrt(lambda))*pos.data[pp];
            };
        }
    else
        {
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::overwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(nBlocks);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_spins(pos.data,d_curandRNGs.data, blockSize,nBlocks,N);
        }
    };

int cubicLattice::latticeSiteToLinearIndex(const int3 &target)
    {
    if(sliceSites)
        {
            /*
        int xp = wrap(target.x,L);
        int yp = wrap(target.y-target.x,L);
        int zp = wrap(target.z-target.y+target.x,L);
        return latticeIndex(xp,yp,zp);
        */
            return latticeIndex(target);
        }
    else
        {
        return latticeIndex(target);
        }
    }
/*!
stencilType here has the same meaning as in "getNeighbors" function
*/
void cubicLattice::fillNeighboLists(int stencilType)
    {

    vector<int> neighs;
    int nNeighs;
    int temp = getNeighbors(0, neighs,nNeighs,stencilType);

    neighborIndex = Index2D(nNeighs,N);
    neighboringSites.resize(nNeighs*N);

    //if(useGPU)
        {
        ArrayHandle<int> neighbors(neighboringSites);
        for (int ii = 0; ii < N; ++ii)
            {
            temp = getNeighbors(ii, neighs,nNeighs,stencilType);
            for (int jj = 0; jj < nNeighs; ++jj)
                {
                neighbors.data[neighborIndex(jj,ii)] = neighs[jj];
                }
            }
        }
    //else
    //    {
    //    }
    };

/*!
returns, in the vector "neighbors" a list of lattice neighbors of the target site.
If stencilType ==0 (the default), the result will be
neighs = 6;
neighbors = {xMinus, xPlus, yMinus, yPlus, zMinus, zPlus};
If stencilType ==1 neighbors will be suitable for computing mixed first partial derivatives \partial_a \partial b F:
neighs = 18;
neighbors = {xMinus, xPlus, yMinus, yPlus, zMinus, zPlus,
            xMinus_yMinus,xMinus_yPlus,xMinus_zMinus,xMinus_zPlus,xPlus_yMinus,xPlus_yPlus,xPlus_zMinus,xPlus_zPlus,
            yMinus_zMinus,yMinuz_zPlus,yPlus_zMinus,yPlus_zPlus}
*/
int cubicLattice::getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType)
    {
    if(stencilType==0)
        {
        neighs = 6;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        if(!sliceSites)
            {
            int3 position = latticeIndex.inverseIndex(target);
            if(position.x >0 && position.x < latticeIndex.sizes.x-1)
                {
                neighbors[0] = latticeIndex(position.x-1,position.y,position.z);
                neighbors[1] = latticeIndex(position.x+1,position.y,position.z);
                }
            else if(position.x ==0)
                {
                neighbors[0] = latticeIndex(latticeIndex.sizes.x-1,position.y,position.z);
                neighbors[1] = latticeIndex(1,position.y,position.z);
                }
            else
                {
                neighbors[0] = latticeIndex(latticeIndex.sizes.x-2,position.y,position.z);
                neighbors[1] = latticeIndex(0,position.y,position.z);
                };
            if(position.y >0 && position.y < latticeIndex.sizes.y-1)
                {
                neighbors[2] = latticeIndex(position.x,position.y-1,position.z);
                neighbors[3] = latticeIndex(position.x,position.y+1,position.z);
                }
            else if(position.y ==0)
                {
                neighbors[2] = latticeIndex(position.x,latticeIndex.sizes.y-1,position.z);
                neighbors[3] = latticeIndex(position.x,1 ,position.z);
                }
            else
                {
                neighbors[2] = latticeIndex(position.x,latticeIndex.sizes.y-2,position.z);
                neighbors[3] = latticeIndex(position.x,0,position.z);
                };
            if(position.z >0 && position.z < latticeIndex.sizes.z-1)
                {
                neighbors[4] = latticeIndex(position.x,position.y,position.z-1);
                neighbors[5] = latticeIndex(position.x,position.y,position.z+1);
                }
            else if(position.z ==0)
                {
                neighbors[4] = latticeIndex(position.x,position.y,latticeIndex.sizes.z-1);
                neighbors[5] = latticeIndex(position.x,position.y,1);
                }
            else
                {
                neighbors[4] = latticeIndex(position.x,position.y,latticeIndex.sizes.z-2);
                neighbors[5] = latticeIndex(position.x,position.y,0);
                };
            }
        return target;
        };
    if(stencilType==1)
        {
        int3 position = latticeIndex.inverseIndex(target);
        neighs = 18;
        if(neighbors.size()!=neighs) neighbors.resize(neighs);
        neighbors[0] = latticeIndex(wrap(position.x-1,latticeIndex.sizes.x),position.y,position.z);
        neighbors[1] = latticeIndex(wrap(position.x+1,latticeIndex.sizes.x),position.y,position.z);
        neighbors[2] = latticeIndex(position.x,wrap(position.y-1,latticeIndex.sizes.y),position.z);
        neighbors[3] = latticeIndex(position.x,wrap(position.y+1,latticeIndex.sizes.y),position.z);
        neighbors[4] = latticeIndex(position.x,position.y,wrap(position.z-1,latticeIndex.sizes.z));
        neighbors[5] = latticeIndex(position.x,position.y,wrap(position.z+1,latticeIndex.sizes.z));

        neighbors[6] = latticeIndex(wrap(position.x-1,latticeIndex.sizes.x),wrap(position.y-1,latticeIndex.sizes.y),position.z);
        neighbors[7] = latticeIndex(wrap(position.x-1,latticeIndex.sizes.x),wrap(position.y+1,latticeIndex.sizes.y),position.z);
        neighbors[8] = latticeIndex(wrap(position.x-1,latticeIndex.sizes.x),position.y,wrap(position.z-1,latticeIndex.sizes.z));
        neighbors[9] = latticeIndex(wrap(position.x-1,latticeIndex.sizes.x),position.y,wrap(position.z+1,latticeIndex.sizes.z));
        neighbors[10] = latticeIndex(wrap(position.x+1,latticeIndex.sizes.x),wrap(position.y-1,latticeIndex.sizes.y),position.z);
        neighbors[11] = latticeIndex(wrap(position.x+1,latticeIndex.sizes.x),wrap(position.y+1,latticeIndex.sizes.y),position.z);
        neighbors[12] = latticeIndex(wrap(position.x+1,latticeIndex.sizes.x),position.y,wrap(position.z-1,latticeIndex.sizes.z));
        neighbors[13] = latticeIndex(wrap(position.x+1,latticeIndex.sizes.x),position.y,wrap(position.z+1,latticeIndex.sizes.z));

        neighbors[14] = latticeIndex(position.x,wrap(position.y-1,latticeIndex.sizes.y),wrap(position.z-1,latticeIndex.sizes.z));
        neighbors[15] = latticeIndex(position.x,wrap(position.y-1,latticeIndex.sizes.y),wrap(position.z+1,latticeIndex.sizes.z));
        neighbors[16] = latticeIndex(position.x,wrap(position.y+1,latticeIndex.sizes.y),wrap(position.z-1,latticeIndex.sizes.z));
        neighbors[17] = latticeIndex(position.x,wrap(position.y+1,latticeIndex.sizes.y),wrap(position.z+1,latticeIndex.sizes.z));
        return target;
        }

    return target; //nope
    };

void cubicLattice::createBoundaryObject(vector<int> &latticeSites, boundaryType _type, scalar Param1, scalar Param2)
    {
    growGPUArray(boundaries,1);
    ArrayHandle<boundaryObject> boundaryObjs(boundaries);
    boundaryObject bound(_type,Param1,Param2);
    boundaryObjs.data[boundaries.getNumElements()-1] = bound;

    //set all sites in the boundary to the correct type
    int j = boundaries.getNumElements();
    ArrayHandle<int> t(types);
    for (int ii = 0; ii < latticeSites.size();++ii)
        {
        t.data[latticeSites[ii]] = j;
        };

    int neighNum;
    vector<int> neighbors;
    vector<int> surfaceSite;
    //set all neighbors of boundary sites to type -1
    for (int ii = 0; ii < latticeSites.size();++ii)
        {
        int currentIndex = getNeighbors(latticeSites[ii],neighbors,neighNum);
        for (int nn = 0; nn < neighbors.size(); ++nn)
            if(t.data[neighbors[nn]] < 1)
                {
                t.data[neighbors[nn]] = -1;
                surfaceSite.push_back(neighbors[nn]);
                }
        };
    removeDuplicateVectorElements(surfaceSite);

    //add object and surface sites to the vectors
    GPUArray<int> newBoundarySites;
    fillGPUArrayWithVector(latticeSites, newBoundarySites);
    GPUArray<int> newSurfaceSites;
    fillGPUArrayWithVector(surfaceSite, newSurfaceSites);

    boundarySites.push_back(newBoundarySites);
    surfaceSites.push_back(newSurfaceSites);
    boundaryState.push_back(0);
    printf("there are now %i boundary objects known to the configuration...",boundaries.getNumElements());
    printf(" last object had %lu sites and %lu surface sites \n",latticeSites.size(),surfaceSite.size());
    if(surfaceSite.size()>boundaryMoveAssist1.getNumElements())
        {
        boundaryMoveAssist1.resize(surfaceSite.size());
        ArrayHandle<int2> bma1(boundaryMoveAssist1);
        boundaryMoveAssist2.resize(surfaceSite.size());
        ArrayHandle<int2> bma2(boundaryMoveAssist2);
        for (int ii = 0; ii < surfaceSite.size(); ++ii)
            {
            bma1.data[ii].x =0.0;bma1.data[ii].y =0.0;
            bma2.data[ii].x =0.0;bma2.data[ii].y =0.0;
            }
        }
    };

/*!
This function moves an object, shifting the type of lattice sites over on the lattice. Surface sites are also moved.
Sites that change from being part of the boundary to a surface site adopt the average state of the neighboring
sites that weren't formerly part of the boundary object. This function makes use of the fact that primitive
translations will not change the total number of boundary or surface sites associated with each object
\param objectIndex The position in the vector of boundarySites and surfaceSites that the desired object to move is in
\param motionDirection Which way to move the object. Follows same convention as lattice neighbors for stencilType=0
*/
void cubicLattice::displaceBoundaryObject(int objectIndex, int motionDirection)
    {
    if(!useGPU)
        {
        int negativeMotionIndex;
        switch(motionDirection)
            {
            case 0:
                negativeMotionIndex = 1;
                break;
            case 1:
                negativeMotionIndex = 0;
                break;
            case 2:
                negativeMotionIndex = 3;
                break;
            case 3:
                negativeMotionIndex = 2;
                break;
            case 4:
                negativeMotionIndex = 5;
                break;
            case 5:
                negativeMotionIndex = 4;
                break;
            default:
                printf("Illegal motion direction\n");
                return;
            };
        ArrayHandle>dVec> pos(positions);
        ArrayHandle<int> t(types);
        ArrayHandle<int> bSites(boundarySites[objectIndex]);
        ArrayHandle<int> sSites(surfaceSites[objectIndex]);
        ArrayHandle<int2> bma1(boundaryMoveAssist1);
        ArrayHandle<int2> bma2(boundaryMoveAssist2);
        ArrayHandle<int> neighbors(neighboringSites,access_location::host,access_mode::read);

        for(int ii = 0; ii < surfaceSites[objectIndex].getNumElements();++ii)
            {
            int site = sSites.data[ii];
            int motionSite = neighbors.data[neighborIndex(motionDirection,site)];
            int negativeMotionSite = neighbors.data[neighborIndex(negativeMotionIndex,site)];
            if(t.data[motionSite] >0) //that boundary site is about to become a surface site
                {
                }
            if(t.data[negativeMotionSite] >0)//the surface site is about to become a boundary site
                {
                }

            }


        for(int ii =0; ii < boundarySites[objectIndex].getNumElements();++ii)
            {
            int site = bSites.data[ii];
            int motionSite = neighbors.data[neighborIndex(motionDirection,site)];
            int negativeMotionSite = neighbors.data[neighborIndex(negativeMotionIndex,site)];
            if(t.data[motionSite] < 1)
                {

                }
            }

        }
    else
        {
        };
    };
