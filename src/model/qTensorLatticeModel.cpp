#include "qTensorLatticeModel.h"
#include "cubicLattice.cuh"
/*! \file qTensorLatticeModel.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */
qTensorLatticeModel::qTensorLatticeModel(int l, bool _useGPU, bool _neverGPU)
    : cubicLattice(l,false,_useGPU, neverGPU)
    {
    normalizeSpins = false;
    if(neverGPU)
        defectMeasures.noGPU = true;
    defectMeasures.resize(N);
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

qTensorLatticeModel::qTensorLatticeModel(int lx,int ly,int lz, bool _useGPU, bool _neverGPU)
    : cubicLattice(lx,ly,lz,false,_useGPU,_neverGPU)
    {
    normalizeSpins = false;
    if(neverGPU)
        defectMeasures.noGPU = true;
    defectMeasures.resize(N);
    if(DIMENSION !=5)
        {
        printf("\nAttempting to run a simulation with incorrectly set dimension... change the root CMakeLists.txt file to have dimension 5 and recompile\n");
        throw std::exception();
        }
    };

void qTensorLatticeModel::getAverageMaximalEigenvector(vector<scalar> &averageN)
    {
    ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    vector<scalar> eigenValues(3);
    //eigensystemOfQ orders the eignevectors from smallest to largest...
    vector<scalar> eigenVector1(3);
    vector<scalar> eigenVector2(3);
    vector<scalar> eigenVector3(3);//Hence, this is the one we care about.

    averageN.resize(3); averageN[0]=0.0;averageN[1]=0.0;averageN[2]=0.0;
    int n = 0;
    for (int pp = 0; pp < N; ++pp)
        {
        if(t.data[pp] <=0)
            {
            n += 1;
            eigensystemOfQ(Q.data[pp],eigenValues,eigenVector1,eigenVector2,eigenVector3);
            averageN[0] += eigenVector3[0];
            averageN[1] += eigenVector3[1];
            averageN[2] += eigenVector3[2];
            }
        }
    averageN[0] = averageN[0]/n;
    averageN[1] = averageN[1]/n;
    averageN[2] = averageN[2]/n;
    }

void qTensorLatticeModel::getAverageEigenvalues(bool verbose)
    {
    ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
    ArrayHandle<int> t(types,access_location::host,access_mode::read);
    scalar a ,b,c;
    a = b = c = 0.;
    int n = 0;
    for (int pp = 0; pp < N; ++pp)
        {
        scalar a1,b1,c1;
        if(t.data[pp] <=0)
            {
            eigenvaluesOfQ(Q.data[pp],a1,b1,c1);
            a += a1;
            b += b1;
            c += c1;
            n += 1;
            }
        }
    if(verbose)    printf("average eigenvalues: %f\t%f\t%f\n",a/n,b/n,c/n);
    }

/*!
defectType==0 stores the largest eigenvalue of Q at each site
defectType==1 stores the determinant of Q each site
defectType==2 stores the (Tr(Q^2))^3-54 det(Q)^2 at each site
*/
void qTensorLatticeModel::computeDefectMeasures(int defectType)
    {
    if(!useGPU)
        {
        ArrayHandle<dVec> Q(positions,access_location::host,access_mode::read);
        ArrayHandle<int> t(types,access_location::host,access_mode::read);
        ArrayHandle<scalar> defects(defectMeasures,access_location::host,access_mode::overwrite);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar a,b,c;
            if(t.data[pp] >0)
                continue;
            if(defectType==0)
                {
                eigenvaluesOfQ(Q.data[pp],a,b,c);
                defects.data[pp] = max(max(a,b),c);
                }
            if(defectType==1)
                {
                defects.data[pp] = determinantOfQ(Q.data[pp]);
                }
            if(defectType==2)
                {
                scalar trQ2 = TrQ2(Q.data[pp]);
                scalar det = determinantOfQ(Q.data[pp]);
                defects.data[pp] = trQ2*trQ2*trQ2 - 54.0*det*det;
                }
            }
        }//end CPU
    else
        {
            ArrayHandle<int> t(types,access_location::device,access_mode::read);
            ArrayHandle<dVec> pos(positions,access_location::device,access_mode::read);
            ArrayHandle<scalar> defects(defectMeasures,access_location::device,access_mode::overwrite);
            gpu_get_qtensor_DefectMeasures(pos.data,defects.data,t.data,defectType,N);
        }
    }
void qTensorLatticeModel::setNematicQTensorRandomly(noiseSource &noise,scalar S0, bool globallyAligned)
    {
    //cout << "setting randomly aligned nematic Q tensors of strength " << S0 << endl;
    scalar amplitude =  3./2.*S0;
    scalar globalTheta = acos(2.0*noise.getRealUniform()-1);
    scalar globalPhi = 2.0*PI*noise.getRealUniform();

    if(!useGPU)
        {
        ArrayHandle<dVec> pos(positions);
        ArrayHandle<int> t(types,access_location::host,access_mode::read);
        for(int pp = 0; pp < N; ++pp)
            {
            scalar theta = acos(2.0*noise.getRealUniform()-1);
            scalar phi = 2.0*PI*noise.getRealUniform();
            if(globallyAligned)
                {
                theta = globalTheta;
                phi = globalPhi;
                }
            if(t.data[pp] <=0)
                {
                scalar3 n;
                n.x = cos(phi)*sin(theta);
                n.y = sin(phi)*sin(theta);
                n.z = cos(theta);
                qTensorFromDirector(n, S0, pos.data[pp]);
                };
            };
        }
    else
        {
        ArrayHandle<int> t(types,access_location::device,access_mode::read);
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::readwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(N);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_nematic_qTensors(pos.data,t.data,d_curandRNGs.data, S0, blockSize,nBlocks,globallyAligned,globalTheta,globalPhi,N);
        }
    };

void qTensorLatticeModel::moveParticles(GPUArray<dVec> &displacements,scalar scale)
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
                //approximately restrict Q-tensor values
                /*
                scalar otherMin = -2.0/3.0;
                scalar otherMax = 5./6.;
                for (int dd = 0; dd < DIMENSION; ++dd)
                    {
                    scalar max = (dd >2 ) ? .5 : otherMax;
                    scalar min = (dd >2 ) ? -.75 : otherMin;
                    if(h_pos.data[pp][dd] > max) h_pos.data[pp][dd] = max;
                    if(h_pos.data[pp][dd] < min) h_pos.data[pp][dd] = min;
                    }
                */
                }
            }
        else
            {
            for(int pp = 0; pp < N; ++pp)
                {
                h_pos.data[pp] += scale*h_disp.data[pp];
                //approximately restrict Q-tensor values
                /*
                scalar otherMin = -2.0/3.0;
                scalar otherMax = 5./6.;
                for (int dd = 0; dd < DIMENSION; ++dd)
                    {
                    scalar max = (dd >2 ) ? .5 : otherMax;
                    scalar min = (dd >2 ) ? -.75 : otherMin;
                    if(h_pos.data[pp][dd] > max) h_pos.data[pp][dd] = max;
                    if(h_pos.data[pp][dd] < min) h_pos.data[pp][dd] = min;
                    }
                */
                }
            }
        }
    else
        {//gpu branch
        moveParticlesTuner->begin();
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        if(scale == 1.0)
            gpu_update_qTensor(d_disp.data,d_pos.data,N,moveParticlesTuner->getParameter());
        else
            gpu_update_qTensor(d_disp.data,d_pos.data,scale,N,moveParticlesTuner->getParameter());

        moveParticlesTuner->end();
        };
    };

/*!
Reads a carefully prepared text file to create a new boundary object...
The first line must be a single integer specifying the number of objects to be read in.

Each subsequent block must be formatted as follows (with no additional lines between the blocks):
The first line MUST be formatted as
a b c d
where a=0 means homeotropic, a=1 means degeneratePlanar
b is a scalar setting the anchoring strength
c is the preferred value of S0
d is an integer specifying the number of sites.

Every subsequent line MUST be
x y z Qxx Qxy Qxz Qyy Qyz,
where x y z are the integer lattice sites,
and the Q-tensor components correspond to the desired anchoring conditions.
For homeotropic boundaries, Q^B = 3 S_0/2*(\nu^s \nu^s - \delta_{ab}/3), where \nu^s is the
locally preferred director.
For degenerate planar anchoring the boundary site should be,
Q^B[0] = \hat{nu}_x
Q^B[1] = \hat{nu}_y
Q^B[2] = \hat{nu}_z
where \nu^s = {Cos[\[Phi]] Sin[\[theta]], Sin[\[Phi]] Sin[\[theta]], Cos[\[theta]]}
 is the direction to which the LC should try to be orthogonal
*/
void qTensorLatticeModel::createBoundaryFromFile(string fname, bool verbose)
    {
    ifstream inFile(fname);
    string line,name;
    scalar sVar1,sVar2,sVar3,sVar4,sVar5;
    int iVar1,iVar2,iVar3;
    int nObjects;

    getline(inFile,line);
    istringstream ss(line);
    ss >> nObjects;

    for (int ii = 0; ii < nObjects;++ii)
        {
        getline(inFile,line);
        istringstream ss(line);
        ss >>iVar1 >> sVar1 >> sVar2 >>iVar2;

        scalar Wb = sVar1;
        scalar s0 = sVar2;
        int nEntries = iVar2;
        int bType = iVar1;
        boundaryType bound;
        if(bType == 0)
            bound = boundaryType::homeotropic;
        else
            bound = boundaryType::degeneratePlanar;
        if(verbose)
            printf("reading boudary type %i with %f %f and %i entries\n",iVar1,sVar1,sVar2,iVar2);

        dVec Qtensor;
        vector<int> boundSites;
        int3 sitePos;
        ArrayHandle<dVec> pos(positions);
        int entriesRead = 0;
        while (entriesRead < nEntries && getline(inFile,line) )
            {
            istringstream linestream(line);
            linestream >> iVar1 >> iVar2 >> iVar3 >> Qtensor[0] >> Qtensor[1] >> Qtensor[2] >>Qtensor[3] >> Qtensor[4];
            sitePos.x = wrap(iVar1,latticeIndex.sizes.x);
            sitePos.y = wrap(iVar2,latticeIndex.sizes.y);
            sitePos.z = wrap(iVar3,latticeIndex.sizes.z);
            int currentSite = latticeIndex(sitePos);
            boundSites.push_back(currentSite);
            pos.data[currentSite] = Qtensor;
            /*
            if(verbose)
           printf("(%i,%i,%i) %f %f %f %f %f\n",sitePos.x,sitePos.y,sitePos.z,
                                                Qtensor[0],Qtensor[1],Qtensor[2],Qtensor[3],Qtensor[4]);
            */
            entriesRead += 1;
            };
        if(verbose)
            printf("object with %lu sites created\n",boundSites.size());
        createBoundaryObject(boundSites,bound,Wb,s0);
        };
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
                        scalar nrm = norm(disp);
                        disp.x /=nrm;
                        disp.y /=nrm;
                        disp.z /=nrm;
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
