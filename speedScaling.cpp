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

int3 partitionProcessors(int numberOfProcesses)
    {
    int3 ans;
    ans.z = floor(pow(numberOfProcesses,1./3.));
    int nLeft = floor(numberOfProcesses/ans.z);
    ans.y = floor(sqrt(nLeft));
    ans.x = floor(nLeft / ans.y);
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

    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);

    ValueArg<scalar> aSwitchArg("a","phaseConstantA","value of phase constant A",false,0.172,"scalar",cmd);
    ValueArg<scalar> bSwitchArg("b","phaseConstantB","value of phase constant B",false,2.12,"scalar",cmd);
    ValueArg<scalar> cSwitchArg("c","phaseConstantC","value of phase constant C",false,1.73,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for minimizer",false,0.0005,"scalar",cmd);

    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);
    ValueArg<int> iterationsSwitchArg("i","iterations","number of minimization steps",false,100,"int",cmd);
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
    int nDev;
    cudaGetDeviceCount(&nDev);
    if(nDev == 0)
        gpu = -1;
    bool GPU = false;
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
    if(myRank >= 0 && gpu >=0 && worldSize > 1)
        GPU = chooseGPU(myLocalRank);
    else if (gpu >=0)
        GPU = chooseGPU(gpu);

    int3 rankTopology = partitionProcessors(worldSize);
    if(myRank ==0)
        printf("lattice divisions: {%i, %i, %i}\n",rankTopology.x,rankTopology.y,rankTopology.z);

    scalar a = -1;
    scalar b = -phaseB/phaseA;
    scalar c = phaseC/phaseA;
    noiseSource noise(reproducible);
    noise.setReproducibleSeed(13371+myRank);
    printf("setting a rectilinear lattice of size (%i,%i,%i)\n",boxLx,boxLy,boxLz);
    profiler pInit("initialization");

    pInit.start();
    bool xH = (rankTopology.x >1) ? true : false;
    bool yH = (rankTopology.y >1) ? true : false;
    bool zH = (rankTopology.z >1) ? true : false;
    //xH=yH=zH=true;
    bool edges = nConstants > 1 ? true : false;
    bool corners = false;


    vector<int> boxSizes;
    boxSizes.push_back(10);
    boxSizes.push_back(20);
    boxSizes.push_back(30);
    boxSizes.push_back(50);
    boxSizes.push_back(70);
    boxSizes.push_back(100);
    boxSizes.push_back(150);
    boxSizes.push_back(200);
    boxSizes.push_back(250);
    if(gpu==1 || gpu <0)
        {
        boxSizes.push_back(300);
        boxSizes.push_back(325);
        }
    char filename[256];
    sprintf(filename,"../data/speedScaling_g%i_z%i_m%i_rank%i.txt",gpu,programSwitch,minimizerSwitch,myRank);
    ofstream myfile;
    myfile.open(filename);
    myfile.setf(ios_base::scientific);
    myfile << setprecision(10);
    for(int ll = 0; ll < boxSizes.size(); ++ll)
    {
    boxLx = boxLy=boxLz=boxL = boxSizes[ll];
    shared_ptr<multirankQTensorLatticeModel> Configuration = make_shared<multirankQTensorLatticeModel>(boxLx,boxLy,boxLz,xH,yH,zH);
    shared_ptr<multirankSimulation> sim = make_shared<multirankSimulation>(myRank,rankTopology.x,rankTopology.y,rankTopology.z,edges,corners);
    shared_ptr<landauDeGennesLC> landauLCForce = make_shared<landauDeGennesLC>();
    sim->setConfiguration(Configuration);
    pInit.end();

    landauLCForce->setPhaseConstants(a,b,c);
    printf("relative phase constants: %f\t%f\t%f\n",a,b,c);
    if(nConstants ==1)
        {
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

    scalar forceCutoff=1e-17;
    int iterationsPerStep = 100;


    shared_ptr<energyMinimizerGradientDescent> GDminimizer = make_shared<energyMinimizerGradientDescent>(Configuration);
    GDminimizer->setMaximumIterations(maximumIterations);
    shared_ptr<energyMinimizerFIRE> Fminimizer =  make_shared<energyMinimizerFIRE>(Configuration);
    Fminimizer->setMaximumIterations(maximumIterations);
    if(minimizerSwitch ==1)
        {
        GDminimizer->setGradientDescentParameters(dt,forceCutoff);
        sim->addUpdater(GDminimizer,Configuration);
        }
    else
        {
        scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.5;
        scalar alphaDec=0.9; int nMin=4;scalar alphaMin = .0;
        Fminimizer->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
        sim->addUpdater(Fminimizer,Configuration);
        }

    sim->setCPUOperation(true);//have cpu and gpu initialized the same...for debugging
    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
    printf("setting random configuration with S0 = %f\n",S0);
    Configuration->setNematicQTensorRandomly(noise,S0);
    sim->setCPUOperation(!GPU);
    printf("initialization done\n");

    sim->finalizeObjects();


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

    scalar E1;
    scalar maxForce;
    chrono::time_point<chrono::high_resolution_clock>  startTime;
    chrono::time_point<chrono::high_resolution_clock>  endTime;

    chrono::time_point<chrono::high_resolution_clock>  s1,e1;
    scalar remTime = 0.0;

    scalar workingTime;
    startTime = chrono::high_resolution_clock::now();
    sim->performTimestep();

    endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> difference = endTime-startTime;
    workingTime = difference.count() / (scalar) maximumIterations;
    myfile << boxL*boxL*boxL <<"\t" << workingTime <<"\n";

    pMinimize.end();

    pMinimize.print();
    sim->p1.print();
    }
    myfile.close();
    MPI_Finalize();
        return 0;
};

