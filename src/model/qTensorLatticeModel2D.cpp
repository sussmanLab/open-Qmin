#include "qTensorLatticeModel2D.h"
#include "cubicLattice.cuh" // note that cubicLattice.cuh actually has all of the default "spin updating" gpu codes...keep this here, and refactor name later
#include "qTensorLatticeModel2D.cuh"



/*! \file qTensorLatticeModel2D.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */
qTensorLatticeModel2D::qTensorLatticeModel2D(int l, bool _useGPU, bool _neverGPU)
    : squareLattice(l,false,_useGPU, _neverGPU)
    {
    normalizeSpins = false;
    if(neverGPU)
        defectMeasures.noGPU = true;
    defectMeasures.resize(N);
    if(DIMENSION <2 )
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension at least 5 and recompile\n");
        throw std::exception();
        }
    };

qTensorLatticeModel2D::qTensorLatticeModel2D(int lx,int ly, bool _useGPU, bool _neverGPU)
    : squareLattice(lx,ly,false,_useGPU,_neverGPU)
    {
    normalizeSpins = false;
    if(neverGPU)
        defectMeasures.noGPU = true;
    defectMeasures.resize(N);
    if(DIMENSION <2 )
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension at least 5 and recompile\n");
        throw std::exception();
        }
    };

void qTensorLatticeModel2D::getAverageMaximalEigenvector(vector<scalar> &averageN)
    {
    ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    vector<scalar> eigenValues(2);
    //eigensystemOfQ orders the eignevectors from smallest to largest...
    vector<scalar> eigenVector1(2);
    vector<scalar> eigenVector2(2);//Hence, this is the one we care about.

    averageN.resize(2); averageN[0]=0.0;averageN[1]=0.0;
    int n = 0;
    for (int pp = 0; pp < N; ++pp)
        {
        if(t.data[pp] <=0)
            {
            n += 1;
            eigensystemOfQ2D(Q.data[pp],eigenValues,eigenVector1,eigenVector2);
            averageN[0] += eigenVector2[0];
            averageN[1] += eigenVector2[1];
            }
        }
    averageN[0] /=n;
    averageN[1] /=n;
    }

void qTensorLatticeModel2D::getAverageEigenvalues(bool verbose)
    {
    ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    scalar a ,b;
    a = b = 0.;
    int n = 0;
    for (int pp = 0; pp < N; ++pp)
        {
        scalar a1,b1;
        if(t.data[pp] <=0)
            {
            eigenvaluesOfQ2D(Q.data[pp],a1,b1);
            a += a1;
            b += b1;
            n += 1;
            }
        }
    if(verbose)    printf("average eigenvalues: %f\t%f\n",a/n,b/n);
    }

/*!
defectType==0 stores the largest eigenvalue of Q at each site
defectType==1 stores the determinant of Q each site
*/
void qTensorLatticeModel2D::computeDefectMeasures(int defectType)
    {
    if(!useGPU)
        {
        ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
        ArrayHandle<int> t(types,access_location::host,access_mode::read);
        ArrayHandle<scalar> defects(defectMeasures,access_location::host,access_mode::overwrite);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar a,b;
            if(t.data[pp] >0)
                continue;
            if(defectType==0)
                {
                eigenvaluesOfQ2D(Q.data[pp],a,b);
                defects.data[pp] = max(a,b);
                }
            if(defectType==1)
                {
                defects.data[pp] = determinantOf2DQ(Q.data[pp]);
                }
            }
        }//end CPU
    else
        {
            ArrayHandle<int> t(types,access_location::device,access_mode::read);
            ArrayHandle<dVec> pos(positions,access_location::device,access_mode::read);
            ArrayHandle<scalar> defects(defectMeasures,access_location::device,access_mode::overwrite);
//            UNWRITTENCODE("NEED TO WRITE 2Dqtensor defect measures computation on the gpu");
            gpu_get_2DqTensor_DefectMeasures(pos.data,defects.data,t.data,defectType,N);
        }
    }

void qTensorLatticeModel2D::setNematicQTensorRandomly(noiseSource &noise,scalar S0, bool globallyAligned)
    {
    //cout << "setting randomly aligned nematic Q tensors of strength " << S0 << endl;
    scalar globalPhi = 2.0*PI*noise.getRealUniform();
    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        ArrayHandle<int> t(types,access_location::host,access_mode::read);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar phi = 2.0*PI*noise.getRealUniform();
            if(globallyAligned)
                {
                phi = globalPhi;
                }
            if(t.data[pp] <=0)
                {
                scalar2 n;
                n.x = cos(phi);
                n.y = sin(phi);
                qTensorFromDirector2D(n, S0, pos.data[pp]);
                };
            };
        }
    else
        {
//        UNWRITTENCODE("set random 2dqtensor code on the gpu");
        
        ArrayHandle<int> t(types,access_location::device,access_mode::read);
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::readwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(N);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_nematic_2DqTensors(pos.data,t.data,d_curandRNGs.data, S0, blockSize,nBlocks,globallyAligned,globalPhi,N);
        
        }
    };

void qTensorLatticeModel2D::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
    if(!useGPU)
        {//cpu branch
        ArrayHandle<dVec> h_disp(displacements, access_location::host,access_mode::read);
        ArrayHandle<dVec> h_pos(positions);
        if(scale == 1.)
            {
            for(int pp = 0; pp < N; ++pp)
                {
                h_pos.data[pp] += h_disp.data[pp];
                }
            }
        else
            {
            for(int pp = 0; pp < N; ++pp)
                {
                h_pos.data[pp] += scale*h_disp.data[pp];
                }
            }
        }
    else
        {//gpu branch
//        UNWRITTENCODE("GPU move particles");
        
        moveParticlesTuner->begin();
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        if(scale == 1.0)
            gpu_update_2DqTensor(d_disp.data,d_pos.data,N,moveParticlesTuner->getParameter());
        else
            gpu_update_2DqTensor(d_disp.data,d_pos.data,scale,N,moveParticlesTuner->getParameter());

        moveParticlesTuner->end();
        
        };
    };

void qTensorLatticeModel2D::createBoundaryObject(vector<int> &latticeSites, boundaryType _type, scalar Param1, scalar Param2)
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
        int currentIndex = getNeighbors(latticeSites[ii],neighbors,neighNum,1);
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
    printf("there are now %i boundary objects known to the configuration...",boundaries.getNumElements());
    printf(" last object had %lu sites and %lu surface sites \n",latticeSites.size(),surfaceSite.size());
    };

//put a boundary with x-normal on lattice plane xyplane
void qTensorLatticeModel2D::createSimpleFlatWall(int xyPlane, boundaryObject &bObj)
    {
    if (xyPlane <0 || xyPlane >=Lx)
        UNWRITTENCODE("NOT AN OPTION FOR A FLAT SIMPLE WALL");
    dVec Qtensor(0.0);
    scalar s0 = bObj.P2;
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            if(xyPlane ==0)
                {Qtensor[0] = s0; Qtensor[1] = 0;}
            else 
                {Qtensor[0] = -s0; Qtensor[1] = 0;}
            break;
            }
        case boundaryType::degeneratePlanar:
            {
            Qtensor[0]=0.0; Qtensor[1] = 0.0;
            Qtensor[xyPlane]=1.0;
            break;
            }
        default:
            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
        };

    vector<int> boundSites;
    ArrayHandle<dVec> pos(positions);
    int currentSite;
    int size1,size2;
    for (int yy = 0; yy < Ly; ++yy)
        {
        currentSite = latticeIndex(xyPlane,yy);
        boundSites.push_back(currentSite);
        pos.data[currentSite] = Qtensor;
        };

    createBoundaryObject(boundSites,bObj.boundary,bObj.P1,bObj.P2);
    };

void qTensorLatticeModel2D::createBoundaryFromFile(string fname, bool verbose)
    {
    UNWRITTENCODE("2D create boundary from file");
    /*
    */
    };
