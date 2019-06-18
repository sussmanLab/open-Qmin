#include "functions.h"
#include "gpuarray.h"
#include "multirankSimulation.h"
#include "simulation.h"
#include "qTensorLatticeModel.h"
#include "landauDeGennesLC.h"
#include "energyMinimizerFIRE.h"
#include "energyMinimizerGradientDescent.h"
#include "noiseSource.h"
#include "indexer.h"
#include "qTensorFunctions.h"
#include "latticeBoundaries.h"
#include "profiler.h"
#include "logSpacedIntegers.h"
#include <tclap/CmdLine.h>
#include <mpi.h>
#include "cuda_profiler_api.h"

/*!
This file has been used to make timing information about finding minima in the presence of various objects and boundary conditions
 */
int3 partitionProcessors(int numberOfProcesses)
    {
    int3 ans;
    int cubeTest = round(pow(numberOfProcesses,1./3.));
    if(cubeTest*cubeTest*cubeTest == numberOfProcesses)
        {
        ans.x = cubeTest;
        ans.y = cubeTest;
        ans.z = cubeTest;
        }
    else
        {
        ans.z = floor(pow(numberOfProcesses,1./3.));
        int nLeft = floor(numberOfProcesses/ans.z);
        ans.y = floor(sqrt(nLeft));
        ans.x = floor(nLeft / ans.y);
        }
    return ans;
    }

using namespace TCLAP;
int main(int argc, char*argv[])
{
    int myRank,worldSize;
    int tag=99;
    char message[20];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD,&myRank);

    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(processorName, &nameLen);
    int myLocalRank;
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,MPI_INFO_NULL, &shmcomm);
    MPI_Comm_rank(shmcomm, &myLocalRank);
    printf("processes rank %i, local rank %i\n",myRank,myLocalRank);

//    printf("Hello world from processor %s, rank %d out of %d processors\n",
//           processorName, myRank, worldSize);


    //First, we set up a basic command line parser with some message and version
    CmdLine cmd("dDimSim applied to a lattice of XY-spins", ' ', "V0.5");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> minimizerSwitchArg("m","minimizerSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> savestateSwitchArg("s","savestateSwitch","an integer controlling the level of configurational output",false,-1,"int",cmd);
    ValueArg<int> fileidxSwitchArg("f","fileidxSwitch","an integer controlling part of the RNG seed used",false,0,"int",cmd);


    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);
    ValueArg<scalar> aSwitchArg("a","phaseConstantA","value of phase constant A",false,0.172,"scalar",cmd);
    ValueArg<scalar> bSwitchArg("b","phaseConstantB","value of phase constant B",false,2.12,"scalar",cmd);
    ValueArg<scalar> cSwitchArg("c","phaseConstantC","value of phase constant C",false,1.73,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for minimizer",false,0.0005,"scalar",cmd);

    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);
    ValueArg<int> iterationsSwitchArg("i","iterations","number of minimization steps",false,20,"int",cmd);
    ValueArg<int> kSwitchArg("k","nConstants","approximation for distortion term",false,1,"int",cmd);


    ValueArg<scalar> l1SwitchArg("","L1","value of L1 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l2SwitchArg("","L2","value of L2 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l3SwitchArg("","L3","value of L3 term",false,4.64,"scalar",cmd);

    ValueArg<int> lSwitchArg("l","boxL","number of lattice sites for cubic box",false,50,"int",cmd);
    ValueArg<int> lxSwitchArg("","Lx","number of lattice sites in x direction",false,50,"int",cmd);
    ValueArg<int> lySwitchArg("","Ly","number of lattice sites in y direction",false,50,"int",cmd);
    ValueArg<int> lzSwitchArg("","Lz","number of lattice sites in z direction",false,50,"int",cmd);

    ValueArg<scalar> q0SwitchArg("q","q0","value of desired q0",false,.05,"scalar",cmd);

    ValueArg<int> threadsSwitchArg("t","threads","number of threads to request",false,1,"int",cmd);

    //parse the arguments
    cmd.parse( argc, argv );
    //define variables that correspond to the command line parameters
    bool reproducible = reproducibleSwitch.getValue();
    int gpu = gpuSwitchArg.getValue();
    int programSwitch = programSwitchArg.getValue();
    int minimizerSwitch = minimizerSwitchArg.getValue();
    int savestate = savestateSwitchArg.getValue();
    int fileidx = fileidxSwitchArg.getValue();
    int nDev;
    cudaGetDeviceCount(&nDev);
    if(nDev == 0)
        gpu = -1;
    scalar phaseA = aSwitchArg.getValue();
    scalar phaseB = bSwitchArg.getValue();
    scalar phaseC = cSwitchArg.getValue();


    int boxL = lSwitchArg.getValue();
    int boxLx = lxSwitchArg.getValue();
    int boxLy = lySwitchArg.getValue();
    int boxLz = lzSwitchArg.getValue();
    if(boxL != 50)
        {
        boxLx = boxL;
        boxLy = boxL;
        boxLz = boxL;
        }

    int nConstants = kSwitchArg.getValue();
    scalar L1 = l1SwitchArg.getValue();
    scalar L2 = l2SwitchArg.getValue();
    scalar L3 = l3SwitchArg.getValue();
    scalar q0 = q0SwitchArg.getValue();

    //int nThreads = threadsSwitchArg.getValue();

    scalar dt = dtSwitchArg.getValue();
    int maximumIterations = iterationsSwitchArg.getValue();
    bool GPU = false;
    if(myRank >= 0 && gpu >=0 && worldSize > 1)
        GPU = chooseGPU(myLocalRank);
    else if (gpu >=0)
        GPU = chooseGPU(gpu);

    logSpacedIntegers logIntTest(0,0.05);
    for (int tt = 0 ;tt < maximumIterations; ++tt)
        logIntTest.update();
    if(myRank ==0)
        cout << "max iterations set to " << logIntTest.nextSave << endl;



    int3 rankTopology = partitionProcessors(worldSize);
    if(myRank ==0)
        printf("lattice divisions: {%i, %i, %i}\n",rankTopology.x,rankTopology.y,rankTopology.z);

    scalar a = -1;
    scalar b = -phaseB/phaseA;
    scalar c = phaseC/phaseA;
    noiseSource noise(reproducible);
    noise.setReproducibleSeed(13371+myRank+fileidx);

    if(myRank ==0)
        printf("setting a rectilinear lattice of size (%i,%i,%i)\n",boxLx,boxLy,boxLz);
    profiler pInit("initialization");

    pInit.start();
    bool xH = (rankTopology.x >1) ? true : false;
    bool yH = (rankTopology.y >1) ? true : false;
    bool zH = (rankTopology.z >1) ? true : false;
    bool edges = nConstants > 1 ? true : false;
    bool corners = nConstants > 1 ? true : false;
    bool neverGPU = !GPU;
    shared_ptr<multirankQTensorLatticeModel> Configuration = make_shared<multirankQTensorLatticeModel>(boxLx,boxLy,boxLz,xH,yH,zH,false,neverGPU);
    shared_ptr<multirankSimulation> sim = make_shared<multirankSimulation>(myRank,rankTopology.x,rankTopology.y,rankTopology.z,edges,corners);
    shared_ptr<landauDeGennesLC> landauLCForce = make_shared<landauDeGennesLC>(neverGPU);
    sim->setConfiguration(Configuration);
    pInit.end();

    if(nConstants ==1)
        {
        if(myRank ==0)
            printf("using 1-constant approximation: %f \n",L1);
        landauLCForce->setElasticConstants(L1,0,0);
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        }
    if(nConstants ==2)
        {
        printf("using 2-constant approximation: %f\t%f\t%f \n",L1,L2,q0);
        landauLCForce->setElasticConstants(L1,L2,q0);
        landauLCForce->setNumberOfConstants(distortionEnergyType::twoConstant);
        }
    if(nConstants ==3)
        {
        printf("using 3-constant approximation: %f\t%f\t%f \n",L1,L2,L3);
        landauLCForce->setElasticConstants(L1,L2,L3);
        landauLCForce->setNumberOfConstants(distortionEnergyType::threeConstant);
        }
    landauLCForce->setModel(Configuration);
    sim->addForce(landauLCForce);

    scalar forceCutoff=1e-16;
    int iterationsPerStep = 100;


//    shared_ptr<energyMinimizerGradientDescent> GDminimizer = make_shared<energyMinimizerGradientDescent>(Configuration);
//    GDminimizer->setMaximumIterations(1);
    shared_ptr<energyMinimizerFIRE> Fminimizer =  make_shared<energyMinimizerFIRE>(Configuration);
    Fminimizer->setMaximumIterations(1);
    if(minimizerSwitch ==1)
        {
//        GDminimizer->setGradientDescentParameters(dt,forceCutoff);
//        sim->addUpdater(GDminimizer,Configuration);
        printf("gradient descent minimizer added\n");
        }
    else
        {
        scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.5;
        scalar alphaDec=0.9; int nMin=4;scalar alphaMin = .0;
        Fminimizer->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
        Fminimizer->setCurrentIterations(0);
        sim->addUpdater(Fminimizer,Configuration);
        if(myRank ==0)
            printf("FIRE minimizer added\n");
        }

    sim->setCPUOperation(true);//have cpu and gpu initialized the same...for debugging
    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
    S0=0.53;
    //FIX S0=0.53
    c = (2.0-b*S0)/(3.0*S0*S0);
    S0 = (-b+sqrt(b*b-24*a*c))/(6*c);

    landauLCForce->setPhaseConstants(a,b,c);

    Configuration->setNematicQTensorRandomly(noise,S0);
    sim->setCPUOperation(!GPU);
    if(myRank ==0)
        {
        printf("relative phase constants: %f\t%f\t%f\n",a,b,c);
        printf("setting random configuration with S0 = %f\n",S0);
        printf("initialization done\n");
        };

    boundaryObject homeotropicBoundary(boundaryType::homeotropic,0.58,S0);
    boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,0.58,S0);
    scalar3 left,center, right,down,up,direction;
    left.x = 0.0*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
    center.x = 0.5*boxLx;center.y = 0.5*boxLy;center.z = 0.5*boxLz;
    right.x = 1.*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
    //use program switches to define what objects are in the simulation
    scalar newRadius = 0.35*boxLx;
    switch(programSwitch)
        {
        case 1:
            sim->createWall(2, 0, homeotropicBoundary); // z-normal wall on plane 0
            left.x = 0.3*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
            right.x = .7*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
            sim->createSphericalColloid(left,0.12*boxLx,homeotropicBoundary);
            sim->createSphericalColloid(right,0.12*boxLx,homeotropicBoundary);
            //sim->createSphericalColloid(center,0.25*boxLx,homeotropicBoundary);
            break;
        case 2:
            left.x = 0.4*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
            right.x = .6*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
            sim->createWall(2, 0, planarDegenerateBoundary); // z-normal wall on plane 0
            sim->createSpherocylinder(left,right,0.1*boxLx,homeotropicBoundary);
            break;
        case 3:
            sim->createSphericalCavity(center,0.5*boxLx-1.5,homeotropicBoundary);
            break;
        case 4:
            down.x = 0.5*boxLx;down.y = 0.5*boxLy;down.z = 0.0*boxLz-1;
            up.x =   0.5*boxLx;  up.y = 0.5*boxLy;  up.z = 1.0*boxLz+1;
            sim->createCylindricalObject(down, up,0.5*boxLx-1.5, false, homeotropicBoundary);
            break;
        case 5:
            sim->createWall(2, 0, homeotropicBoundary); // z-normal wall on plane 0
            sim->createSphericalColloid(center,newRadius,homeotropicBoundary);
            direction.x=0;direction.y=0;direction.z=1.;
            sim->setDipolarField(center,direction, newRadius, 1.0*boxLx,S0);
//            sim->saveState("../data/dipoleTest");
            //sim->createSphericalColloid(center,0.25*boxLx,homeotropicBoundary);
            break;
        case 6:
            sim->createWall(2, 0, homeotropicBoundary); // z-normal wall on plane 0
            sim->createSphericalColloid(center,newRadius,homeotropicBoundary);
//            sim->saveState("../data/dipoleTest");
            //sim->createSphericalColloid(center,0.25*boxLx,homeotropicBoundary);
            break;
        default:
            break;
        }

    sim->finalizeObjects();

    char filename[256];
    sprintf(filename,"../data/minimizationTiming_L%i_g%i_z%i_m%i_dt%.5f_fidx%i_rank%i_worldSize%i.txt",boxLx,gpu,programSwitch,minimizerSwitch,dt,fileidx,myRank,worldSize);
    ofstream myfile;
    if(myRank ==0)
        {
        myfile.open(filename);
        myfile.setf(ios_base::scientific);
        myfile << setprecision(10);
        };

    profiler pMinimize("minimization");
    pMinimize.start();

    /* for linear spaced data saving
    int totalMinimizerCalls = maximumIterations / iterationsPerStep;
    vector<int> minimizationIntervals;
    minimizationIntervals.push_back(2);
    minimizationIntervals.push_back(7);
    minimizationIntervals.push_back(20);
    minimizationIntervals.push_back(30);
    minimizationIntervals.push_back(40);
    for (int tt = 0;tt < totalMinimizerCalls; ++tt)
        minimizationIntervals.push_back(iterationsPerStep);
    */

    logSpacedIntegers logInts(0,0.05);
    int currentIterationMax = 1;
    logInts.update();

    scalar E1;
    scalar maxForce;
    chrono::time_point<chrono::high_resolution_clock>  startTime;
    chrono::time_point<chrono::high_resolution_clock>  endTime;

    chrono::time_point<chrono::high_resolution_clock>  s1,e1;
    scalar remTime = 0.0;

    startTime = chrono::high_resolution_clock::now();
    scalar workingTime;
    //for (int tt = 0; tt < minimizationIntervals.size(); ++tt)
    int iters;
    for (int tt = 0; tt < maximumIterations; ++tt)
        {
        sim->performTimestep();

        //compute the energy, but don't count it towards the minimization time
//        s1 = chrono::high_resolution_clock::now();
//        E1 = sim->computePotentialEnergy(true);
//        e1 = chrono::high_resolution_clock::now();
//        chrono::duration<double> del = e1-s1;
//        remTime += del.count();

        if(minimizerSwitch ==1)
            {
//            maxForce = GDminimizer->getMaxForce();
//            iters = GDminimizer->iterations;
            }
        else
            {
            maxForce = Fminimizer->getMaxForce();
            iters = Fminimizer->iterations;
            }

        endTime = chrono::high_resolution_clock::now();
        chrono::duration<double> difference = endTime-startTime;
        workingTime = difference.count() - remTime;
        if(myRank ==0)
            {
            printf("\n%i \t %g\t %g\t %g\t\t %g \n",iters,E1,maxForce,workingTime, remTime);
            myfile << iters <<"\t" << workingTime <<"\t" << E1 <<"\t"<< maxForce <<"\n";
            }

        currentIterationMax = logInts.nextSave;
        logInts.update();
        //currentIterationMax += minimizationIntervals[tt];
        if(minimizerSwitch ==1)
            {
//            GDminimizer->setMaximumIterations(currentIterationMax);
            }
        else
            Fminimizer->setMaximumIterations(currentIterationMax);
        }
    iters = Fminimizer->iterations;
    pMinimize.end();
    if(savestate >0 || (savestate >=0 &&fileidx == 0))
        {
        char savename[256];
        sprintf(savename,"../data/finalConfiguration_g%i_z%i_m%i_dt%.5f_fidx%i",gpu,programSwitch,minimizerSwitch,dt,fileidx);
        sim->p1.print();
        sim->saveState(savename);
        }
    scalar totalMinTime = workingTime;
    scalar communicationTime = sim->p1.timeTaken;
    if(myRank==0)
        {
        printf("rank: %i\n min  time %f\n energy time %f\n comm time %f\n percent comm: %f\n",myRank,totalMinTime,remTime,communicationTime,communicationTime/totalMinTime);
        pMinimize.print();
        myfile << "final info\n";
        myfile << myRank <<"\t"<<worldSize <<"\t"<<iters<<"\t"<<totalMinTime<<"\t"<<remTime<<"\t"<<communicationTime<<"\n";
        myfile.close();
        }

    /*
    cout << "size of configuration " << Configuration->getClassSize() << endl;
    cout << "size of force computer" << landauLCForce->getClassSize() << endl;
    cout << "size of fire updater " << Fminimizer->getClassSize() << endl;
    cout << "size of gd updater " << GDminimizer->getClassSize() << endl;
    */
    MPI_Finalize();
        return 0;
};

