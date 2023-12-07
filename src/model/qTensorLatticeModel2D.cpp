#include "qTensorLatticeModel2D.h"
#include "cubicLattice.cuh" // note that cubicLattice.cuh actually has all of the default "spin updating" gpu codes...keep this here, and refactor name later

/*! \file qTensorLatticeModel2D.cpp" */

/*!
This simply calls the cubic lattice constructor (without slicing optimization, since that is not yet
operational).
Additionally, throws an exception if the dimensionality is incorrect.
 */
qTensorLatticeModel2D::qTensorLatticeModel2D(int l, bool _useGPU, bool _neverGPU)
    : squareLattice(l,false,_useGPU, neverGPU)
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

qTensorLatticeModel2D::qTensorLatticeModel2D(int lx,int ly,int lz, bool _useGPU, bool _neverGPU)
    : squareLattice(lx,ly,false,_useGPU,_neverGPU)
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
            averageN[2] += eigenVector2[2];
            }
        }
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
            UNWRITTENCODE("NEED TO WRITE 2Dqtensor defect measures computation on the gpu");
//            gpu_get_2Dqtensor_DefectMeasures(pos.data,defects.data,t.data,defectType,N);
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
        UNWRITTENCODE("set random 2dqtensor code on the gpu");
        /*
        ArrayHandle<int> t(types,access_location::device,access_mode::read);
        ArrayHandle<dVec> pos(positions,access_location::device,access_mode::readwrite);
        int blockSize = 128;
        int nBlocks = N/blockSize+1;
        noise.initialize(N);
        noise.initializeGPURNGs();
        ArrayHandle<curandState> d_curandRNGs(noise.RNGs,access_location::device,access_mode::readwrite);
        gpu_set_random_nematic_2DqTensors(pos.data,t.data,d_curandRNGs.data, S0, blockSize,nBlocks,globallyAligned,globalTheta,globalPhi,N);
        */
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
        UNWRITTENCODE("GPU move particles");
        /*
        moveParticlesTuner->begin();
        ArrayHandle<dVec> d_disp(displacements,access_location::device,access_mode::read);
        ArrayHandle<dVec> d_pos(positions,access_location::device,access_mode::readwrite);
        if(scale == 1.0)
            gpu_update_2dqTensor(d_disp.data,d_pos.data,N,moveParticlesTuner->getParameter());
        else
            gpu_update_2dqTensor(d_disp.data,d_pos.data,scale,N,moveParticlesTuner->getParameter());

        moveParticlesTuner->end();
        */
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
x y Qxx Qxy,
where x y are the integer lattice sites,
and the Q-tensor components correspond to the desired anchoring conditions.
For homeotropic boundaries, Q^B = S_0*(\nu^s \nu^s - \delta_{ab}/2), where \nu^s is the
locally preferred director.
For degenerate planar anchoring the boundary site should be,
Q^B[0] = \hat{nu}_x
Q^B[1] = \hat{nu}_y
where \nu^s = {Cos[\[Phi]] Sin[\[theta]], Sin[\[Phi]] Sin[\[theta]], Cos[\[theta]]}
 is the direction to which the LC should try to be orthogonal
*/
void qTensorLatticeModel2D::createBoundaryFromFile(string fname, bool verbose)
    {
    UNWRITTENCODE("2D create boundary from file");
    /*
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
            printf("reading boundary type %i with %f %f and %i entries\n",iVar1,sVar1,sVar2,iVar2);

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
            entriesRead += 1;
            };
        if(verbose)
            printf("object with %lu sites created\n",boundSites.size());
        createBoundaryObject(boundSites,bound,Wb,s0);
        };
    */
    };
