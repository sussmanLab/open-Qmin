#include "qTensorLatticeModel.h"
#include "cubicLattice.cuh"
/*! \file qTensorLatticeModel.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */
qTensorLatticeModel::qTensorLatticeModel(int l, bool _useGPU)
    : cubicLattice(l,false,_useGPU)
    {
    normalizeSpins = false;
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

qTensorLatticeModel::qTensorLatticeModel(int lx,int ly,int lz, bool _useGPU)
    : cubicLattice(lx,ly,lz,false,_useGPU)
    {
    normalizeSpins = false;
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

void qTensorLatticeModel::setNematicQTensorRandomly(noiseSource &noise,scalar S0)
    {
    scalar amplitude =  3./2.*S0;
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar theta = acos(2.0*noise.getRealUniform()-1);
            scalar phi = 2.0*PI*noise.getRealUniform();
            pos.data[pp][0] = amplitude*(sin(theta)*sin(theta)*cos(phi)*cos(phi)-1.0/3.0);
            pos.data[pp][1] = amplitude*sin(theta)*sin(theta)*cos(phi)*sin(phi);
            pos.data[pp][2] = amplitude*sin(theta)*cos(theta)*cos(phi);
            pos.data[pp][3] = amplitude*(sin(theta)*sin(theta)*sin(phi)*sin(phi)-1.0/3.0);
            pos.data[pp][4] = amplitude*sin(theta)*cos(theta)*sin(phi);
            };
        }
    else
        {
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::overwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(N);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_nematic_qTensors(pos.data,d_curandRNGs.data, amplitude, blockSize,nBlocks,N);
        }
    };

void qTensorLatticeModel::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
    cubicLattice::moveParticles(displacements,scale);
    };

void qTensorLatticeModel::createSimpleSpherialColloid(scalar3 center, scalar radius, boundaryObject &bObj)
    {
    dVec Qtensor(0.);
    scalar S0 = bObj.P2;
    vector<int> boundSites;
    ArrayHandle<dVec> pos(positions);
    for (int xx = ceil(center.x-radius); xx < floor(center.x+radius); ++xx)
        for (int yy = ceil(center.y-radius); yy < floor(center.y+radius); ++yy)
            for (int zz = ceil(center.z-radius); zz < floor(center.z+radius); ++zz)
            {
            scalar3 disp;
            disp.x = xx - center.x;
            disp.y = yy - center.y;
            disp.z = zz - center.z;

            if((disp.x*disp.x+disp.y*disp.y+disp.z*disp.z) < radius*radius)
                {
                int3 sitePos;
                sitePos.x = wrap(xx,latticeIndex.sizes.x);
                sitePos.y = wrap(yy,latticeIndex.sizes.y);
                sitePos.z = wrap(zz,latticeIndex.sizes.z);
                int currentSite = latticeIndex(sitePos);
                boundSites.push_back(currentSite);
                switch(bObj.boundary)
                    {
                    case boundaryType::homeotropic:
                        {
                        qTensorFromDirector(disp, S0, Qtensor);
                        break;
                        }
                    case boundaryType::degeneratePlanar:
                        {
                        Qtensor[0]=disp.x; Qtensor[1] = disp.y; Qtensor[2] = disp.z;
                        break;
                        }
                    default:
                        UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
                    };
                pos.data[currentSite] = Qtensor;
                };
            }
    printf("sphere with %lu sites created\n",boundSites.size());
    createBoundaryObject(boundSites,bObj.boundary,bObj.P1,bObj.P2);
    };

void qTensorLatticeModel::createSimpleFlatWallZNormal(int zPlane, boundaryObject &bObj)
    {
    dVec Qtensor(0.0);
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            Qtensor[0] = -0.5*bObj.P2; Qtensor[3] = -0.5*bObj.P2;
            break;
            }
        case boundaryType::degeneratePlanar:
            {
            Qtensor[0]=0.0; Qtensor[1] = 0.0; Qtensor[2] = 1.0;
            break;
            }
        default:
            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
        };

    vector<int> boundSites;
    ArrayHandle<dVec> pos(positions);
    for (int xx = 0; xx < latticeIndex.sizes.x; ++xx)
        for (int yy = 0; yy < latticeIndex.sizes.y; ++yy)
            {
            int currentSite = latticeIndex(xx,yy,zPlane);
            boundSites.push_back(currentSite);
            pos.data[currentSite] = Qtensor;
            }
    createBoundaryObject(boundSites,bObj.boundary,bObj.P1,bObj.P2);
    };

void qTensorLatticeModel::createSimpleFlatWallNormal(int plane, int xyz, boundaryObject &bObj)
    {
    if (xyz <0 || xyz >2)
        UNWRITTENCODE("NOT AN OPTION FOR A FLAT SIMPLE WALL");
    dVec Qtensor(0.0);
    scalar s0 = bObj.P2;
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            if(xyz ==0)
                {Qtensor[0] = s0; Qtensor[3] = -0.5*s0;}
            else if (xyz==1)
                {Qtensor[0] = -0.5*s0; Qtensor[3] = s0;}
            else
                {Qtensor[0] = -0.5*s0; Qtensor[3] = -0.5*s0;}
            break;
            }
        case boundaryType::degeneratePlanar:
            {
            Qtensor[0]=0.0; Qtensor[1] = 0.0; Qtensor[2] = 0.0;
            Qtensor[xyz]=1.0;
            break;
            }
        default:
            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
        };

    vector<int> boundSites;
    ArrayHandle<dVec> pos(positions);
    int currentSite;
    int size1,size2;
    if(xyz ==0)
        {size1=latticeIndex.sizes.y;size2=latticeIndex.sizes.z;}
    else if (xyz==1)
        {size1=latticeIndex.sizes.x;size2=latticeIndex.sizes.z;}
    else
        {size1=latticeIndex.sizes.x;size2=latticeIndex.sizes.y;}
    for (int xx = 0; xx < size1; ++xx)
        for (int yy = 0; yy < size2; ++yy)
            {
            if(xyz ==0)
                currentSite = latticeIndex(plane,xx,yy);
            else if (xyz==1)
                currentSite = latticeIndex(xx,plane,yy);
            else
                currentSite = latticeIndex(xx,yy,plane);

            boundSites.push_back(currentSite);
            pos.data[currentSite] = Qtensor;
            }
    createBoundaryObject(boundSites,bObj.boundary,bObj.P1,bObj.P2);
    };
