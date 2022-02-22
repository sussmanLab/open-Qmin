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
#include "logSpacedIntegers.h"

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
    ValueArg<int> initializationSwitchArg("z","initializationSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);

    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);
    SwitchArg verboseSwitch("v","verbose","output more things to screen ", cmd, false);


    ValueArg<scalar> aSwitchArg("a","phaseConstantA","value of phase constant A",false,-0.172,"scalar",cmd);
    ValueArg<scalar> bSwitchArg("b","phaseConstantB","value of phase constant B",false,-2.12,"scalar",cmd);
    ValueArg<scalar> cSwitchArg("c","phaseConstantC","value of phase constant C",false,1.73,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for minimizer",false,0.0005,"scalar",cmd);
    ValueArg<scalar> forceToleranceSwitchArg("f","fTarget","target minimization threshold for norm of residual forces",false,0.000000000001,"scalar",cmd);

    ValueArg<int> iterationsSwitchArg("i","iterations","maximum number of minimization steps",false,100,"int",cmd);
    ValueArg<int> randomSeedSwitch("","randomSeed","seed for reproducible random number generation", false, -1, "int",cmd);

    scalar defaultL=4.64;
    ValueArg<scalar> l1SwitchArg("","L1","value of L1 term",false,defaultL,"scalar",cmd);
    ValueArg<scalar> l2SwitchArg("","L2","value of L2 term",false,defaultL,"scalar",cmd);
    ValueArg<scalar> l3SwitchArg("","L3","value of L3 term",false,defaultL,"scalar",cmd);
    ValueArg<scalar> l4SwitchArg("","L4","value of L4 term",false,defaultL,"scalar",cmd);
    ValueArg<scalar> l6SwitchArg("","L6","value of L6 term",false,defaultL,"scalar",cmd);

    ValueArg<int> lSwitchArg("l","boxL","number of lattice sites for cubic box",false,50,"int",cmd);
    ValueArg<int> lxSwitchArg("","Lx","number of lattice sites in x direction",false,50,"int",cmd);
    ValueArg<int> lySwitchArg("","Ly","number of lattice sites in y direction",false,50,"int",cmd);
    ValueArg<int> lzSwitchArg("","Lz","number of lattice sites in z direction",false,50,"int",cmd);

    ValueArg<string> initialConfigurationFileSwitchArg("","initialConfigurationFile", "carefully prepared file of the initial state of all lattice sites" ,false, "NONE", "string",cmd);
    ValueArg<string> fieldFileSwitchArg("","spatiallyVaryingFieldFile", "carefully prepared file containing information on a spatially varying external H field" ,false, "NONE", "string",cmd);
    ValueArg<string> boundaryFileSwitchArg("","boundaryFile", "carefully prepared file of boundary sites" ,false, "NONE", "string",cmd);
    ValueArg<string> saveFileSwitchArg("","saveFile", "the base name to save the post-minimization configuration" ,false, "NONE", "string",cmd);
    ValueArg<int> linearSaveSwitchArg("","linearSpacedSaving","save a file every x minimization steps",false,-1,"int",cmd);
    ValueArg<scalar> logSaveSwitchArg("","logSpacedSaving","save a file every x^j for integer j",false,-1,"scalar",cmd);
    ValueArg<int> saveStrideSwitchArg("","stride","stride of the saved lattice sites",false,1,"int",cmd);

    ValueArg<scalar> setHFieldXSwitchArg("","hFieldX", "x component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldYSwitchArg("","hFieldY", "y component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldZSwitchArg("","hFieldZ", "z component of external H field",false,0,"scalar",cmd);
    ValueArg<scalar> setHFieldMu0SwitchArg("","hFieldMu0", "mu0 for external magenetic field",false,1,"scalar",cmd);
    ValueArg<scalar> setHFieldChiSwitchArg("","hFieldChi", "Chi for external magenetic field",false,1,"scalar",cmd);
    ValueArg<scalar> setHFieldDeltaChiSwitchArg("","hFieldDeltaChi", "Delta Chi for external magenetic field",false,0.5,"scalar",cmd);

    ValueArg<scalar> setEFieldXSwitchArg("","eFieldX", "x component of external E field",false,0,"scalar",cmd);
    ValueArg<scalar> setEFieldYSwitchArg("","eFieldY", "y component of external E field",false,0,"scalar",cmd);
    ValueArg<scalar> setEFieldZSwitchArg("","eFieldZ", "z component of external E field",false,0,"scalar",cmd);
    ValueArg<scalar> setEFieldEps0SwitchArg("","eFieldEpsilon0", "epsilon0 for external electric field",false,1,"scalar",cmd);
    ValueArg<scalar> setEFieldEpsSwitchArg("","eFieldEpsilon", "Epsilon for external electric field",false,1,"scalar",cmd);
    ValueArg<scalar> setEFieldDeltaEpsSwitchArg("","eFieldDeltaEpsilon", "DeltaEpsilon for external electric field",false,0.5,"scalar",cmd);

    //parse the arguments
    cmd.parse( argc, argv );
    //define variables that correspond to the command line parameters
    string initialConfigurationFile = initialConfigurationFileSwitchArg.getValue();
    string fieldFile = fieldFileSwitchArg.getValue();
    string boundaryFile = boundaryFileSwitchArg.getValue();
    string saveFile = saveFileSwitchArg.getValue();
    int saveStride = saveStrideSwitchArg.getValue();
    int linearSave = linearSaveSwitchArg.getValue();
    scalar logSave = logSaveSwitchArg.getValue();

    int randomSeed = randomSeedSwitch.getValue();
    bool reproducible = reproducibleSwitch.getValue();
    if(randomSeed != -1)
        reproducible = true;

    bool verbose= verboseSwitch.getValue();
    int gpu = gpuSwitchArg.getValue();
    int initializationSwitch = initializationSwitchArg.getValue();
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
    scalar c = -phaseC/phaseA;
    noiseSource noise(reproducible);
    if(randomSeed == -1)
        noise.setReproducibleSeed(13371+myRank);
    else
        noise.setReproducibleSeed(randomSeed+myRank);

    if(verbose) printf("setting a rectilinear lattice of size (%i,%i,%i)\n",boxLx,boxLy,boxLz);
    profiler pInit("initialization");
    bool useOneConstantApprox = true;
    if(L2 != defaultL || L3 != defaultL || L4 != defaultL || L6 != defaultL)
        useOneConstantApprox = false;
    pInit.start();
    bool xH = (rankTopology.x >1) ? true : false;
    bool yH = (rankTopology.y >1) ? true : false;
    bool zH = (rankTopology.z >1) ? true : false;
    bool edges = ((rankTopology.y >1) && !useOneConstantApprox) ? true : false;
    bool corners = ((rankTopology.z >1) && !useOneConstantApprox) ? true : false;
    bool neverGPU = !GPU;

    shared_ptr<multirankQTensorLatticeModel> Configuration = make_shared<multirankQTensorLatticeModel>(boxLx,boxLy,boxLz,xH,yH,zH,false,neverGPU);
    shared_ptr<multirankSimulation> sim = make_shared<multirankSimulation>(myRank,rankTopology.x,rankTopology.y,rankTopology.z,edges,corners);
    shared_ptr<landauDeGennesLC> landauLCForce = make_shared<landauDeGennesLC>(neverGPU);
    sim->setConfiguration(Configuration);
    pInit.end();

    landauLCForce->setPhaseConstants(a,b,c);
    if(verbose) printf("relative phase constants: %f\t%f\t%f\n",a,b,c);
    if(useOneConstantApprox)
        {
        if(verbose) printf("using 1-constant approximation: %f \n",L1);
        landauLCForce->setElasticConstants(L1);
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        }
    else
        {
        if(verbose) printf("using a multi-constant approximation with (L1,L2,L3,L4,L6) = : (%f, %f, %f, %f, %f) \n",L1,L2,L3,L4,L6);
        landauLCForce->setElasticConstants(L1,L2,L3,L4,L6);
        landauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);
        }

    scalar3 fieldH,fieldE; //direction and magnitude
    fieldH.x = setHFieldXSwitchArg.getValue();
    fieldH.y = setHFieldYSwitchArg.getValue();
    fieldH.z = setHFieldZSwitchArg.getValue();
    scalar mu0 = setHFieldMu0SwitchArg.getValue();
    scalar chi = setHFieldChiSwitchArg.getValue();
    scalar deltaChi = setHFieldDeltaChiSwitchArg.getValue();
    fieldE.x = setEFieldXSwitchArg.getValue();
    fieldE.y = setEFieldYSwitchArg.getValue();
    fieldE.z = setEFieldZSwitchArg.getValue();
    scalar eps0 = setEFieldEps0SwitchArg.getValue();
    scalar eps = setEFieldEpsSwitchArg.getValue();
    scalar deltaEps = setEFieldDeltaEpsSwitchArg.getValue();
    bool applyFieldH = false;
    bool applyFieldE = false;
    bool applyVaryingField = fieldFile == "NONE" ? false : true;
    if(fieldH.x != 0 || fieldH.y != 0 || fieldH.z != 0)
        applyFieldH = true;
    if(fieldE.x != 0 || fieldE.y != 0 || fieldE.z != 0)
        applyFieldE = true;
    if(applyFieldH)
        {
        landauLCForce->setHField(fieldH,chi,mu0,deltaChi);
        if(verbose) printf("applying H-field (%f, %f, %f) with (mu0, chi, deltaChi) = (%f, %f, %f)\n",fieldH.x,fieldH.y,fieldH.z,mu0,chi,deltaChi);
        }
    if(applyFieldE)
        {
        landauLCForce->setEField(fieldE,eps,eps0,deltaEps);
        if(verbose) printf("applying E-field (%f, %f, %f) with (eps0, eps, deltaEps) = (%f, %f, %f)\n",fieldE.x,fieldE.y,fieldE.z,eps0,eps,deltaEps);
        }

    landauLCForce->setModel(Configuration);
    sim->addForce(landauLCForce);
    if(applyVaryingField)
        {
        landauLCForce->setSpatiallyVaryingField(fieldFile,chi, mu0,deltaChi,sim->rankParity);
        }


    shared_ptr<energyMinimizerFIRE> Fminimizer =  make_shared<energyMinimizerFIRE>(Configuration);
    Fminimizer->setMaximumIterations(maximumIterations);
    scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.95;
    scalar alphaDec=0.9; int nMin=4;scalar alphaMin = .0;
    Fminimizer->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
    sim->addUpdater(Fminimizer,Configuration);

    sim->setCPUOperation(true);//have cpu and gpu initialized the same...for debugging
    /*
    The following header file includes various common ways you might want to set the inital state of the lattice of Qtensors. 
    It is controlled by the "initializationSwitch" command line option (-z integer); by default (-z 0) the lattice will be set to a different random Q-tensor at every lattice site (with uniform s0)
    */
    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
#include "setInitialConditions.h"
    sim->setCPUOperation(!GPU);
    if(verbose) printf("initialization done\n");

    if(boundaryFile == "NONE")
        {
        if(myRank ==0 && verbose )
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

    if(linearSave < 0 && logSave < 0)   //minimize and save end result
        {
        //note that this single "performTimestep()" call performs some number of iterations of the FIRE algorithm, with that number set from the command line
        sim->performTimestep();
        }
    else if (logSave<0)                 //save every linearSave steps
        {
        int currentIteration = 0;
        string saveFileAppend="_t";
        while(currentIteration +linearSave < maximumIterations && Fminimizer->getMaxForce() > forceCutoff)
            {
            //save the current state, then minimize more
            string newSaveFile = saveFile+saveFileAppend+std::to_string(currentIteration);
            if(saveFile != "NONE")
                sim->saveState(newSaveFile,saveStride);
            currentIteration += linearSave;
            Fminimizer->setMaximumIterations(currentIteration);
            sim->performTimestep();
            if(verbose) printf("saving to %s (plus _xX_yY_zZ.txt)\n",newSaveFile.c_str());
            };
        }
    else                                //save logarithmically
        {
        int currentIteration = 0;
        string saveFileAppend="_t";
        logSpacedIntegers lsi(0,logSave);

        while(currentIteration < maximumIterations && Fminimizer->getMaxForce() > forceCutoff)
            {
            //save the current state, then minimize more
            string newSaveFile = saveFile+saveFileAppend+std::to_string(currentIteration);
            if(saveFile != "NONE")
                sim->saveState(newSaveFile,saveStride);
            lsi.update();
            currentIteration =lsi.nextSave;
            Fminimizer->setMaximumIterations(currentIteration);
            sim->performTimestep();
            if(verbose) printf("saving to %s (plus _xX_yY_zZ.txt)\n",newSaveFile.c_str());
            };
        }

    pMinimize.end();


    scalar E1 = sim->computePotentialEnergy(true);
    scalar maxForce;
    maxForce = Fminimizer->getMaxForce();

    printf("minimized to %g\t E=%f\t\n",maxForce,E1);

    if(verbose) pMinimize.print();
    if(verbose) sim->p1.print();
    if(saveFile != "NONE")
        sim->saveState(saveFile,saveStride);
    scalar totalMinTime = pMinimize.timeTaken;
    scalar communicationTime = sim->p1.timeTaken;
    if(myRank == 0 && verbose)
        printf("min  time %f\n comm time %f\n percent comm: %f\n",totalMinTime,communicationTime,communicationTime/totalMinTime);

    if(verbose) cout << "size of configuration " << Configuration->getClassSize() << endl;
    if(verbose) cout << "size of force computer" << landauLCForce->getClassSize() << endl;
    if(verbose) cout << "size of fire updater " << Fminimizer->getClassSize() << endl;


    /*
    //save average maximal eigenvector information -- occasionally used for debugging
    vector<scalar> averageN(3);
    Configuration->getAverageMaximalEigenvector(averageN);
    ofstream tempF("averageN.txt", ios::app);
    tempF <<averageN[0]<<", "<<averageN[1]<<", "<<averageN[2]<<"\n";
    tempF.close();
    */
    MPI_Finalize();
    return 0;
    };
