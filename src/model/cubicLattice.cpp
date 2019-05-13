#include "cubicLattice.h"
#include "cubicLattice.cuh"
#include "functions.h"
/*! \file cubicLattice.cpp */

cubicLattice::cubicLattice(int l, bool _slice, bool _useGPU, bool _neverGPU)
    {
    useGPU=_useGPU;
    neverGPU = _neverGPU;
    sliceSites = _slice;
    N = l*l*l;
    L=l;
    Box = make_shared<periodicBoundaryConditions>(L);
    selfForceCompute = false;
    initializeNSites();
    normalizeSpins = true;
    latticeIndex = Index3D(l);
    };

cubicLattice::cubicLattice(int lx, int ly, int lz, bool _slice, bool _useGPU, bool _neverGPU)
    {
    useGPU=_useGPU;
    neverGPU = _neverGPU;
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
    if(neverGPU)
        {
        neighboringSites.noGPU = true;
        boundaries.noGPU = true;
        boundaryMoveAssist1.noGPU = true;
        boundaryMoveAssist2.noGPU = true;
        }
    initializeSimpleModel(N);

    moveParticlesTuner = make_shared<kernelTuner>(512,1024,128,10,200000);
    };

void cubicLattice::moveParticles(GPUArray<dVec> &dofs,GPUArray<dVec> &displacements,scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(dofs);
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
void cubicLattice::fillNeighborLists(int stencilType)
    {

    vector<int> neighs;
    int nNeighs;
    int temp = getNeighbors(0, neighs,nNeighs,stencilType);

    neighborIndex = Index2D(nNeighs,N);
    neighboringSites.resize(nNeighs*N);

    //if(!useGPU)
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
    GPUArray<int> newSurfaceSites;
    if(neverGPU)
        {
        newBoundarySites.noGPU = true;
        newSurfaceSites.noGPU = true;
        }
    fillGPUArrayWithVector(latticeSites, newBoundarySites);
    fillGPUArrayWithVector(surfaceSite, newSurfaceSites);

    boundarySites.push_back(newBoundarySites);
    surfaceSites.push_back(newSurfaceSites);
    boundaryState.push_back(0);
    scalar3 zero; zero.x = 0.0;zero.y = 0.0;zero.z = 0.0;
    boundaryForce.push_back(zero);
    printf("there are now %i boundary objects known to the configuration...",boundaries.getNumElements());
    printf(" last object had %lu sites and %lu surface sites \n",latticeSites.size(),surfaceSite.size());

    if(latticeSites.size()>boundaryMoveAssist1.getNumElements())
        boundaryMoveAssist1.resize(latticeSites.size());
    //ArrayHandle<pair<int,dVec> > bma1(boundaryMoveAssist1,access_location::host,access_mode::overwrite);
    if(surfaceSite.size()>boundaryMoveAssist2.getNumElements())
        boundaryMoveAssist2.resize(surfaceSite.size());
    //ArrayHandle<pair<int,dVec> > bma2(boundaryMoveAssist2,access_location::host,access_mode::overwrite);
    };

/*!
This function moves an object, shifting the type of lattice sites over on the lattice. Surface sites are also moved.
Sites that change from being part of the boundary to a surface site adopt the average state of the neighboring
sites that weren't formerly part of the boundary object. This function makes use of the fact that primitive
translations will not change the total number of boundary or surface sites associated with each object
\param objectIndex The position in the vector of boundarySites and surfaceSites that the desired object to move is in
\param motionDirection Which way to move the object. Follows same convention as lattice neighbors for stencilType=0
\param magnitude number of lattice sites to move in the given direction
*/
void cubicLattice::displaceBoundaryObject(int objectIndex, int motionDirection, int magnitude)
    {
    for (int mm = 0; mm < magnitude; ++mm)
    {
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions,access_location::host,access_mode::readwrite);
        ArrayHandle<int> t(types,access_location::host,access_mode::readwrite);
        ArrayHandle<int> bSites(boundarySites[objectIndex],access_location::host,access_mode::readwrite);
        ArrayHandle<int> sSites(surfaceSites[objectIndex],access_location::host,access_mode::readwrite);
        ArrayHandle<pair<int,dVec> > bma1(boundaryMoveAssist1,access_location::host,access_mode::overwrite);
        ArrayHandle<pair<int,dVec> > bma2(boundaryMoveAssist2,access_location::host,access_mode::overwrite);
        ArrayHandle<int> neighbors(neighboringSites,access_location::host,access_mode::read);

        //first, copy the Q-tensors for parallel transport, and set all surface sites to type 0 (will be overwritten in second step)
        for(int bb = 0; bb < boundarySites[objectIndex].getNumElements();++bb)
            {
            int site = bSites.data[bb];
            int motionSite = neighbors.data[neighborIndex(motionDirection,site)];
            bma1.data[bb].first = motionSite;
            bma1.data[bb].second = pos.data[site];
            }
        for (int ss = 0; ss < surfaceSites[objectIndex].getNumElements();++ss)
            {
            int site = sSites.data[ss];
            int motionSite = neighbors.data[neighborIndex(motionDirection,site)];
            bma2.data[ss].first = motionSite;
            bma2.data[ss].second = pos.data[site];
            t.data[site] = 0;
            }
        for(int bb = 0; bb < boundarySites[objectIndex].getNumElements();++bb)
            {
            int site = bma1.data[bb].first;
            bSites.data[bb] = site;
            pos.data[site] = bma1.data[bb].second;
            t.data[site] = objectIndex+1;
            }
        for (int ss = 0; ss < surfaceSites[objectIndex].getNumElements();++ss)
            {
            int site = bma2.data[ss].first;
            sSites.data[ss] = site;
            pos.data[site] = bma2.data[ss].second;
            t.data[site] = -1;
            }
        }
    else
        {
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::readwrite);
        ArrayHandle<int> t(types,access_location::device,access_mode::readwrite);
        ArrayHandle<int> bSites(boundarySites[objectIndex],access_location::device,access_mode::readwrite);
        ArrayHandle<int> sSites(surfaceSites[objectIndex],access_location::device,access_mode::readwrite);
        ArrayHandle<pair<int,dVec> > bma1(boundaryMoveAssist1,access_location::device,access_mode::overwrite);
        ArrayHandle<pair<int,dVec> > bma2(boundaryMoveAssist2,access_location::device,access_mode::overwrite);
        ArrayHandle<int> neighbors(neighboringSites,access_location::device,access_mode::read);

        //copy boundary...don't reset lattice type
        gpu_copy_boundary_object(pos.data,bSites.data,neighbors.data,bma1.data,t.data,neighborIndex,
                                 motionDirection,false,boundarySites[objectIndex].getNumElements());
        //copy suface...reset lattice type
        gpu_copy_boundary_object(pos.data,sSites.data,neighbors.data,bma2.data,t.data,neighborIndex,
                                 motionDirection,true,surfaceSites[objectIndex].getNumElements());

        //move both...
        gpu_move_boundary_object(pos.data,bSites.data,bma1.data,t.data,objectIndex+1,
                                 boundarySites[objectIndex].getNumElements());
        gpu_move_boundary_object(pos.data,sSites.data,bma2.data,t.data,-1,
                                 surfaceSites[objectIndex].getNumElements());
        };
    }//end loop over steps.... this should be (easily) optimized away at some point.
    };
