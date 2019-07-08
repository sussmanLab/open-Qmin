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

    //parse the arguments
    cmd.parse( argc, argv );
    //define variables that correspond to the command line parameters
    string boundaryFile = boundaryFileSwitchArg.getValue();
    string saveFile = saveFileSwitchArg.getValue();

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
	shared_ptr<energyMinimizerFIRE> fire;
	shared_ptr<energyMinimizerNesterovAG> nesterov;
	scalar globalLx = boxLx*rankTopology.x;
	scalar globalLy = boxLy*rankTopology.y;
	scalar globalLz = boxLz*rankTopology.z;
	noise.Reproducible = true;
	landauLCForce->setPhaseConstants(0.000000,0.000000,0.000000);
	landauLCForce->setModel(Configuration);
	sim->addForce(landauLCForce);
	sim->clearUpdaters();
	fire = make_shared<energyMinimizerFIRE>(Configuration);
	sim->addUpdater(fire,Configuration);
	sim->setCPUOperation(!sim->useGPU);
	fire->setCurrentIterations(0);
	fire->setFIREParameters(0.0005000000,0.990000,0.050000,1.100000,0.950000,0.900000,4,0.0000000000010000,0.000000);
	fire->setMaximumIterations(1000);
	sim->setCPUOperation(!GPU);
	Configuration->setNematicQTensorRandomly(noise,0.580383);
	landauLCForce->setPhaseConstants(-1.000000,-12.325581,9.058140);
	landauLCForce->setElasticConstants(4.640000,0,0,0,0);
	landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
	landauLCForce->setModel(Configuration);
	{auto upd = sim->updaters[0].lock();
	upd->setCurrentIterations(0);
	upd->setMaximumIterations(1000);}
	sim->performTimestep();
{scalar3 spherePos;
	spherePos.x =0.500000*globalLx;
	spherePos.y =0.500000*globalLy;
	spherePos.z =0.500000*globalLz;
	scalar rad = 0.250000*globalLx;
	boundaryObject homeotropicBoundary(boundaryType::homeotropic,5.800000,0.530000);
sim->createSphericalColloid(spherePos,rad,homeotropicBoundary);}
	{boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,5.800000,0.530000);
	sim->createWall(0,0,planarDegenerateBoundary);}
	sim->finalizeObjects();
	sim->clearUpdaters();
	nesterov = make_shared<energyMinimizerNesterovAG>(Configuration);
	sim->addUpdater(nesterov,Configuration);
	sim->setCPUOperation(!sim->useGPU);
	nesterov->scheduledMomentum = false;
	nesterov->setCurrentIterations(0);
	nesterov->setNesterovAGParameters(0.0050000000,0.010000,0.0000000000010000);
	nesterov->setMaximumIterations(1000);
	{auto upd = sim->updaters[0].lock();
	upd->setCurrentIterations(0);
	upd->setMaximumIterations(1000);}
	sim->performTimestep();
	{auto upd = sim->updaters[0].lock();
	upd->setCurrentIterations(0);
	upd->setMaximumIterations(1000);}
	sim->performTimestep();
	{scalar3 center,direction;
	direction.x=0;direction.y=0;direction.z=1;
	center.x = 0.5000000000*globalLx;
	center.y = 0.5000000000*globalLx;
	center.z = 0.5000000000*globalLx;
	scalar radius = 0.2500000000*globalLx;
	scalar range = 2.0000000000*globalLx;
	scalar thetaD = 1.0000000000*PI;
	sim->setDipolarField(center,3.141593,5.000000,40.000000,0.580383);}
	sim->clearUpdaters();
	fire = make_shared<energyMinimizerFIRE>(Configuration);
	sim->addUpdater(fire,Configuration);
	sim->setCPUOperation(!sim->useGPU);
	fire->setCurrentIterations(0);
	fire->setFIREParameters(0.0005000000,0.990000,0.050000,1.100000,0.950000,0.900000,4,0.0000000000010000,0.000000);
	fire->setMaximumIterations(1000);
	{auto upd = sim->updaters[0].lock();
	upd->setCurrentIterations(0);
	upd->setMaximumIterations(1000);}
	sim->performTimestep();



	MPI_Finalize();
	return 0;
	};
