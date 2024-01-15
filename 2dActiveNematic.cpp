#include "functions.h"
//#include "multirankSimulation.h"
//#include "multirankQTensorLatticeModel.h"
#include "landauDeGennesLC2D.h"
#include "energyMinimizerFIRE.h"
#include "energyMinimizerNesterovAG.h"
#include "noiseSource.h"
#include "indexer.h"
#include "profiler.h"
#include <tclap/CmdLine.h>
#include "logSpacedIntegers.h"
#include "simulation.h"
#include "vectorValueDatabase.h"

#include "activeQTensorModel2D.h"
#include "activeBerisEdwards2D.h"

void getVecToSave(shared_ptr<activeQTensorModel2D> Conf, int N, vector<double> &vec)
    {
//printf("%i\n",    (Conf->returnPositions()).getNumElements());
    vec.resize(7*N);
    ArrayHandle<dVec> p(Conf->returnPositions());
    ArrayHandle<dVec> v(Conf->returnVelocities());
    ArrayHandle<scalar> pressure(Conf->pressure);
    for (int nn = 0; nn < N; ++nn)
        {
        int2 invIdx= Conf->latticeIndex.inverseIndex(nn);
        vec[7*nn+0] = invIdx.x;
        vec[7*nn+1] = invIdx.y;
        vec[7*nn+2] = p.data[nn].x[0];
        vec[7*nn+3] = p.data[nn].x[1];
        vec[7*nn+4] = v.data[nn].x[0];
        vec[7*nn+5] = v.data[nn].x[1];
        vec[7*nn+6] = pressure.data[nn];
        };
    }


//programBranch = 0 --> active flow, branch=1 -->minimization
using namespace TCLAP;
int main(int argc, char*argv[])
    {

    //First, we set up a basic command line parser with some message and version
    CmdLine cmd("2D active nematic simulation!",' ',"V1.0");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programBranchSwitchArg("z","programBranchSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);

    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);
    SwitchArg verboseSwitch("v","verbose","output more things to screen ", cmd, false);
    SwitchArg saveSwitch("s","save","save files", cmd, false);

    ValueArg<int> lSwitchArg("l","boxL","number of lattice sites for cubic box",false,50,"int",cmd);
    ValueArg<int> lxSwitchArg("","Lx","number of lattice sites in x direction",false,50,"int",cmd);
    ValueArg<int> lySwitchArg("","Ly","number of lattice sites in y direction",false,50,"int",cmd);

    ValueArg<scalar> L1SwitchArg("L","L1","LdG constant for the first distortion term",false,65536.,"scalar",cmd);
    ValueArg<scalar> nclSwitchArg("n","nematicCoherenceLength","nematic coherence length",false,0.5,"scalar",cmd);
    ValueArg<scalar> alSwitchArg("a","activeLengthscale","active length scale ",false,2.,"scalar",cmd);
    ValueArg<scalar> ReSwitchArg("R","Re","Reynolds number",false,0.1,"scalar",cmd);
    ValueArg<scalar> flowAlignmentSwitchArg("F","flowAlignment","flow alignment parameter lambda",false,0.1,"scalar",cmd);
    ValueArg<scalar> rotationalViscositySwitchArg("q","rv","rotational viscosity parameter",false,2560.,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for equation of motion",false,0.0002,"scalar",cmd);
    ValueArg<scalar> pdtSwitchArg("w","pseudodeltaT","step size for pressure poisson relaxation",false,0.0002,"scalar",cmd);
    ValueArg<scalar> forceToleranceSwitchArg("f","fTarget","target minimization threshold for norm of residual forces",false,0.000000000001,"scalar",cmd);

    ValueArg<int> iterationsSwitchArg("i","iterations","maximum number of minimization steps",false,100,"int",cmd);
    ValueArg<int> randomSeedSwitch("","randomSeed","seed for reproducible random number generation", false, -1, "int",cmd);

    //parse the arguments
    cmd.parse( argc, argv );
    //define variables that correspond to the command line parameters
    int boxL = lSwitchArg.getValue();
    int boxLx = lxSwitchArg.getValue();
    int boxLy = lySwitchArg.getValue();
    int randomSeed = randomSeedSwitch.getValue();
    bool reproducible = reproducibleSwitch.getValue();
    bool saveFiles = saveSwitch.getValue();
    if(randomSeed != -1)
        reproducible = true;

    bool verbose= verboseSwitch.getValue();
    int gpu = gpuSwitchArg.getValue();
    int programBranch = programBranchSwitchArg.getValue();
    int nDev=0;
    if(gpu>0)
        {
        cudaGetDeviceCount(&nDev);
        if(nDev == 0)
            gpu = -1;
        };

    if(boxL!=50)
        {
        boxLx = boxL;
        boxLy = boxL;
        }
    scalar dt = dtSwitchArg.getValue();
    scalar forceCutoff = forceToleranceSwitchArg.getValue();
    int maximumIterations = iterationsSwitchArg.getValue();

    bool GPU = false;
    if (gpu >=0)
            GPU = chooseGPU(gpu);

    noiseSource noise(reproducible);
    if(randomSeed == -1)
        noise.setReproducibleSeed(13371);
    else
        noise.setReproducibleSeed(randomSeed);

    if(verbose) printf("setting a rectilinear lattice of size (%i,%i)\n",boxLx,boxLy);

    scalar L1 =L1SwitchArg.getValue();
    bool slice = false;
    scalar c = L1 / (nclSwitchArg.getValue()*nclSwitchArg.getValue());
    scalar a = -c;
    scalar S0 = sqrt(-1.0*a/(4.0*c));

    scalar nCoherence = nclSwitchArg.getValue();
    scalar activeLengthScale = alSwitchArg.getValue();
    scalar ReynoldsNumber = ReSwitchArg.getValue();
    scalar flowAlignmentParameter = flowAlignmentSwitchArg.getValue();
    scalar rotationalViscosity = rotationalViscositySwitchArg.getValue();

    shared_ptr<activeQTensorModel2D> Configuration = make_shared<activeQTensorModel2D>(boxLx,boxLy,GPU, GPU);
    Configuration->setNematicQTensorRandomly(noise,S0,false);

    shared_ptr<landauDeGennesLC2D> landauLCForce = make_shared<landauDeGennesLC2D>(a,c,L1, GPU);
    landauLCForce->setModel(Configuration);

    scalar pseudoTimestep = pdtSwitchArg.getValue();



    if(programBranch ==0)//active flow simulations
        {
        scalar dpTarget = 0.0001;
        shared_ptr<activeBerisEdwards2D> activeBE2D = make_shared<activeBerisEdwards2D>(L1,rotationalViscosity,flowAlignmentParameter,ReynoldsNumber,activeLengthScale,dt,pseudoTimestep, dpTarget);
        shared_ptr<Simulation> sim = make_shared<Simulation>();
        sim->setConfiguration(Configuration);
        sim->addForce(landauLCForce);
        sim->addUpdater(activeBE2D,Configuration);


        char dataname[256];
        sprintf(dataname,"./activeFlow.nc");
        vector<double> saveVec(7*boxLx*boxLy);
        //store database: [x,y,qxx,qxy,vx,vy,p]
        vectorValueDatabase vvdat(7*boxLx*boxLy,dataname,NcFile::Replace);

        profiler timestepProf("timestep cost");
        //For testing purposes, save in log-spaced time
        logSpacedIntegers lsi(0,.15);
        lsi.update();
        int currentIteration = lsi.nextSave;

        for (int ii = 0; ii < maximumIterations; ++ii)
            {
            if(ii == currentIteration)
                {
                cout << ii << endl;
                getVecToSave(Configuration, boxLx*boxLy,saveVec);
                if(saveFiles)
                    vvdat.writeState(saveVec,(ii)*dt);
                lsi.update();
                currentIteration = lsi.nextSave;
                }
            timestepProf.start();
            sim->performTimestep();
            timestepProf.end();
            }
        printf("Mean number of pressure relaxation iterations = %f\n",activeBE2D->reportMeanPressureIterations());
        timestepProf.print();
        }
    if(programBranch == 1)//non-active minimization
        {
        shared_ptr<energyMinimizerNesterovAG> minimizer =  make_shared<energyMinimizerNesterovAG>(Configuration);
        minimizer->setNesterovAGParameters(dt, 0.01,forceCutoff);
        minimizer->setMaximumIterations(maximumIterations);

        shared_ptr<Simulation> sim = make_shared<Simulation>();
        sim->setConfiguration(Configuration);
        sim->addForce(landauLCForce);
        sim->addUpdater(minimizer,Configuration);
    
        char dataname[256];
        sprintf(dataname,"./minimize.nc");
        //store database: [x,y,qxx,qxy,vx,vy,p]
        vectorValueDatabase vvdat(7*boxLx*boxLy,dataname,NcFile::Replace);
        vector<double> saveVec(7*boxLx*boxLy);
        getVecToSave(Configuration, 7*boxLx*boxLy,saveVec);
        vvdat.writeState(saveVec,0.0);
        sim->performTimestep();
        getVecToSave(Configuration, 7*boxLx*boxLy,saveVec);
        vvdat.writeState(saveVec,0.0);
        };
    
        profiler p2("vnr cost");
        profiler p1("vr cost");
        vector<int> tester1;
        vector<int> tester2;
        p2.start();
        for(int jj = 0; jj < 2000;++jj)
            {
            tester1.push_back(jj);
            }
        p2.end();
        p1.start();
        tester2.reserve(2000);
        for(int jj = 0; jj < 2000;++jj)
            {
            tester2.push_back(jj);
            }
        p1.end();
        p2.print();
        p1.print();
/*
    vector<scalar> maxEvec;
    Configuration->getAverageMaximalEigenvector(maxEvec);
    printf("(%f,%f)\n",maxEvec[0],maxEvec[1]);
    sim->performTimestep();
    Configuration->getAverageMaximalEigenvector(maxEvec);
    printf("(%f,%f)\n",maxEvec[0],maxEvec[1]);

    ArrayHandle<dVec> fp(Configuration->returnPositions());
    printdVec(fp.data[0]);
    printdVec(fp.data[10]);
    printdVec(fp.data[20]);
    shared_ptr<energyMinimizerNesterovAG> minimizer =  make_shared<energyMinimizerNesterovAG>(Configuration);
    minimizer->setNesterovAGParameters(dt, 0.01,forceCutoff);
    minimizer->setMaximumIterations(maximumIterations);
    sim->addUpdater(minimizer,Configuration);
*/
/*
*/
    return 0;
    };
