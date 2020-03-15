#include "functions.h"
#include "multirankSimulation.h"
#include "multirankQTensorLatticeModel.h"
#include "landauDeGennesLC.h"
#include "energyMinimizerFIRE.h"
#include "energyMinimizerNesterovAG.h"
#include "energyMinimizerLoLBFGS.h"
#include "energyMinimizerAdam.h"
#include "energyMinimizerGradientDescent.h"
#include "noiseSource.h"
#include "indexer.h"
#include "qTensorFunctions.h"
#include "latticeBoundaries.h"
#include "profiler.h"
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
    //printf("processes rank %i, local rank %i\n",myRank,myLocalRank);

    //First, we set up a basic command line parser with some message and version
    CmdLine cmd("openQmin simulation!",' ',"V0.8");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);

    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);

    ValueArg<scalar> aSwitchArg("a","phaseConstantA","value of phase constant A",false,0.172,"scalar",cmd);
    ValueArg<scalar> bSwitchArg("b","phaseConstantB","value of phase constant B",false,2.12,"scalar",cmd);
    ValueArg<scalar> cSwitchArg("c","phaseConstantC","value of phase constant C",false,1.73,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for minimizer",false,0.0005,"scalar",cmd);
    ValueArg<scalar> forceToleranceSwitchArg("f","fTarget","target minimization threshold for norm of residual forces",false,0.000000000001,"scalar",cmd);

    ValueArg<int> iterationsSwitchArg("i","iterations","number of minimization steps",false,100,"int",cmd);
    ValueArg<int> kSwitchArg("k","nConstants","approximation for distortion term",false,1,"int",cmd);


    ValueArg<scalar> l1SwitchArg("","L1","value of L1 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l2SwitchArg("","L2","value of L2 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l3SwitchArg("","L3","value of L3 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l4SwitchArg("","L4","value of L4 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l6SwitchArg("","L6","value of L6 term",false,4.64,"scalar",cmd);

    ValueArg<int> lSwitchArg("l","boxL","number of lattice sites for cubic box",false,50,"int",cmd);
    ValueArg<int> lxSwitchArg("","Lx","number of lattice sites in x direction",false,50,"int",cmd);
    ValueArg<int> lySwitchArg("","Ly","number of lattice sites in y direction",false,50,"int",cmd);
    ValueArg<int> lzSwitchArg("","Lz","number of lattice sites in z direction",false,50,"int",cmd);

    ValueArg<string> boundaryFileSwitchArg("","boundaryFile", "carefully prepared file of boundary sites" ,false, "NONE", "string",cmd);
    ValueArg<string> saveFileSwitchArg("","saveFile", "the base name to save the post-minimization configuration" ,false, "NONE", "string",cmd);
    ValueArg<int> saveStrideSwitchArg("","stride","stride of the saved lattice sites",false,1,"int",cmd);

    ValueArg<scalar> setHFieldXSwitchArg("","hFieldX", "x component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldYSwitchArg("","hFieldY", "y component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldZSwitchArg("","hFieldZ", "z component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldMu0SwitchArg("","hFieldMu0", "mu0 for external magenetic field",false,1,"scalar",cmd);
    ValueArg<scalar> setHFieldChiSwitchArg("","hFieldChi", "Chi for external magenetic field",false,1,"scalar",cmd);
    ValueArg<scalar> setHFieldDeltaChiSwitchArg("","hFieldDeltaChi", "mu0 for external magenetic field",false,0.5,"scalar",cmd);

    //parse the arguments
    cmd.parse( argc, argv );
    //define variables that correspond to the command line parameters
    string boundaryFile = boundaryFileSwitchArg.getValue();
    string saveFile = saveFileSwitchArg.getValue();
    int saveStride = saveStrideSwitchArg.getValue();

    bool reproducible = reproducibleSwitch.getValue();
    int gpu = gpuSwitchArg.getValue();
    int programSwitch = programSwitchArg.getValue();
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
    scalar L4 = l4SwitchArg.getValue();
    scalar L6 = l6SwitchArg.getValue();

    scalar dt = dtSwitchArg.getValue();
    scalar forceCutoff = forceToleranceSwitchArg.getValue();
    int maximumIterations = iterationsSwitchArg.getValue();

    bool GPU = false;
    if(myRank >= 0 && gpu >=0 && worldSize > 1)
            GPU = chooseGPU(myLocalRank);
    else if (gpu >=0)
            GPU = chooseGPU(gpu);

    int3 rankTopology = partitionProcessors(worldSize);
    if(myRank ==0 && worldSize > 1)
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
    bool edges = ((rankTopology.y >1) && nConstants > 1) ? true : false;
    bool corners = ((rankTopology.z >1) && nConstants > 1) ? true : false;
    bool neverGPU = !GPU;
    shared_ptr<multirankQTensorLatticeModel> Configuration = make_shared<multirankQTensorLatticeModel>(boxLx,boxLy,boxLz,xH,yH,zH,false,neverGPU);
    shared_ptr<multirankSimulation> sim = make_shared<multirankSimulation>(myRank,rankTopology.x,rankTopology.y,rankTopology.z,edges,corners);
    shared_ptr<landauDeGennesLC> landauLCForce = make_shared<landauDeGennesLC>(neverGPU);
    sim->setConfiguration(Configuration);
    pInit.end();

    landauLCForce->setPhaseConstants(a,b,c);
    printf("relative phase constants: %f\t%f\t%f\n",a,b,c);
    if(nConstants ==1)
        {
        printf("using 1-constant approximation: %f \n",L1);
        landauLCForce->setElasticConstants(L1);
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        }
    else
        {
        landauLCForce->setElasticConstants(L1,L2,L3,L4,L6);
        landauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);
        }

    scalar3 field; //direction and magnitude
    field.x = setHFieldXSwitchArg.getValue();
    field.y = setHFieldYSwitchArg.getValue();
    field.z = setHFieldZSwitchArg.getValue();
    scalar mu0 = setHFieldMu0SwitchArg.getValue();
    scalar chi = setHFieldChiSwitchArg.getValue();
    scalar deltaChi = setHFieldDeltaChiSwitchArg.getValue();
    bool applyField = false;
    if(field.x != 0 || field.y != 0 || field.z != 0)
        applyField = true;
    if(applyField)
        {
        landauLCForce->setHField(field,chi,mu0,deltaChi);
        printf("applying H-field (%f, %f, %f) with (mu0, chi, deltaChi) = (%f, %f, %f)\n",field.x,field.y,field.z,mu0,chi,deltaChi);
        }

    landauLCForce->setModel(Configuration);
    sim->addForce(landauLCForce);

    shared_ptr<energyMinimizerFIRE> Fminimizer =  make_shared<energyMinimizerFIRE>(Configuration);
    Fminimizer->setMaximumIterations(maximumIterations);
    scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.95;
    scalar alphaDec=0.9; int nMin=4;scalar alphaMin = .0;
    Fminimizer->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
    sim->addUpdater(Fminimizer,Configuration);

    sim->setCPUOperation(true);//have cpu and gpu initialized the same...for debugging
    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
    printf("setting random configuration with S0 = %f\n",S0);
    Configuration->setNematicQTensorRandomly(noise,S0);
    sim->setCPUOperation(!GPU);
    printf("initialization done\n");

    if(boundaryFile == "NONE")
        {
        if(myRank ==0)
            cout << "not using any custom boundary conditions" << endl;
        }
    else
        {
        sim->createBoundaryFromFile(boundaryFile,true);
        }

    /*
    If you would like to add certain types of pre-defined objects to the simulation, you can insert them here
    (before the "finalizeObjects()" call below. You can also change the indicated header file and recompile
    the code, if you prefer. That header file contains (possibly) helpful examples to follow
    */
#include "addObjectsToOpenQmin.h"
    sim->finalizeObjects();

    profiler pMinimize("minimization");
    pMinimize.start();
    //note that this single "performTimestep()" call performs some number of iterations of the FIRE algorithm, with that number set from the command line
    sim->performTimestep();
    pMinimize.end();

    scalar E1 = sim->computePotentialEnergy(true);
    scalar maxForce;
    maxForce = Fminimizer->getMaxForce();

    printf("minimized to %g\t E=%f\t\n",maxForce,E1);

    pMinimize.print();
    sim->p1.print();
    if(saveFile != "NONE")
        sim->saveState(saveFile,saveStride);
    scalar totalMinTime = pMinimize.timeTaken;
    scalar communicationTime = sim->p1.timeTaken;
    if(myRank == 0)
        printf("min  time %f\n comm time %f\n percent comm: %f\n",totalMinTime,communicationTime,communicationTime/totalMinTime);

    cout << "size of configuration " << Configuration->getClassSize() << endl;
    cout << "size of force computer" << landauLCForce->getClassSize() << endl;
    cout << "size of fire updater " << Fminimizer->getClassSize() << endl;
    MPI_Finalize();
    return 0;
    };
