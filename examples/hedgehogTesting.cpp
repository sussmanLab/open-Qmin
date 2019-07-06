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

scalar distanceFromDipolarConfiguration(shared_ptr<multirankQTensorLatticeModel> conf, scalar3 center, scalar radius, scalar thetaD,scalar S0)
    {
    scalar answer = 0.0;
    ArrayHandle<dVec> pos(conf->returnPositions());
    ArrayHandle<int> t(conf->returnTypes());
    int N  = conf->getNumberOfParticles();
    dVec qTensorTarget;
    scalar k = 0.32;
    scalar rd = 1.22;

    if (thetaD < 0.75*PI)
        rd = 1.1;

    scalar td = thetaD;
    for (int ii = 0; ii < N; ++ii)
        {
        if(t.data[ii] >0)
            continue;
        int3 globalSitePosition = conf->indexToPosition(ii);
        scalar3 relativePosition;
        relativePosition.x = globalSitePosition.x - center.x;
        relativePosition.y = globalSitePosition.y - center.y;
        relativePosition.z = globalSitePosition.z - center.z;
        scalar r ,t, p, Theta;
        scalar3 director;
        r= norm(relativePosition);
        if(r >radius)
            {
            t = acos(relativePosition.z/r);
            p = atan2(relativePosition.y,relativePosition.x);
            r = r/radius; //normalize properly

            scalar ri = 1./r;
            scalar ri2 = ri/r;
            scalar ri3= ri2/r;
            scalar rdi = 1./rd;
            scalar rdi2 = rdi/rd;
            scalar rdi3= rdi2/rd;
            Theta = 2*t;
            Theta -= 0.5*( atan2(r*Sin(t) - rd*Sin(td),r*Cos(t) - rd*Cos(td))
                          +atan2(r*Sin(t) + rd*Sin(td),r*Cos(t) - rd*Cos(td))
                          +atan2(r*rd*Sin(t) - Sin(td),r*rd*Cos(t) - Cos(td))
                          +atan2(r*rd*Sin(t) + Sin(td),r*rd*Cos(t) - Cos(td)) );
            Theta += exp(-k*ri3)*
                            ( (rd+rdi)*(ri-ri2)*Cos(td)*Sin(t)
                             +(rd*rd+rdi2)*(ri2-ri3)*(2.*Cos(td)*Cos(td)-1.)*Sin(t)*Cos(t)
                             +1./3.*(rd*rd*rd+rdi3)*(ri3-ri3/r)*Cos(td)*(4.*Cos(td)*Cos(td)-3.)*Sin(t)*(4.*Cos(t)*Cos(t) - 1.) );
            director.x = Sin(Theta)*Cos(p);
            director.y = Sin(Theta)*Sin(p);
            director.z = Cos(Theta);
            if(Theta == 0 && p ==0)
                {
                director.x=0;director.y=0;director.z=0;
                }
            qTensorFromDirector(director, S0, qTensorTarget);
            dVec difference = pos.data[ii]-qTensorTarget;
            answer += TrQ2(difference);
            };//end of loop over particles in range

        }

    return answer;
    };

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
    ValueArg<int> savestateSwitchArg("s","savestateSwitch","an integer controlling the level of configurational output",false,0,"int",cmd);
    ValueArg<int> fileidxSwitchArg("f","fileidxSwitch","an integer controlling part of the RNG seed used",false,0,"int",cmd);


    SwitchArg reproducibleSwitch("r","reproducible","reproducible random number generation", cmd, true);
    ValueArg<scalar> aSwitchArg("a","phaseConstantA","value of phase constant A",false,0.172,"scalar",cmd);
    ValueArg<scalar> bSwitchArg("b","phaseConstantB","value of phase constant B",false,2.12,"scalar",cmd);
    ValueArg<scalar> cSwitchArg("c","phaseConstantC","value of phase constant C",false,1.73,"scalar",cmd);

    ValueArg<scalar> dtSwitchArg("e","deltaT","step size for minimizer",false,0.0005,"scalar",cmd);

    ValueArg<int> gpuSwitchArg("g","GPU","which gpu to use",false,-1,"int",cmd);
    ValueArg<int> iterationsSwitchArg("i","iterations","number of minimization steps",false,100,"int",cmd);
    ValueArg<int> kSwitchArg("k","nConstants","approximation for distortion term",false,1,"int",cmd);


    ValueArg<scalar> l1SwitchArg("","L1","value of L1 term",false,2.32,"scalar",cmd);
    ValueArg<scalar> l2SwitchArg("","L2","value of L2 term",false,4.64,"scalar",cmd);
    ValueArg<scalar> l3SwitchArg("","L3","value of L3 term",false,4.64,"scalar",cmd);

    ValueArg<int> lSwitchArg("l","boxL","number of lattice sites for cubic box",false,50,"int",cmd);
    ValueArg<int> lxSwitchArg("","Lx","number of lattice sites in x direction",false,50,"int",cmd);
    ValueArg<int> lySwitchArg("","Ly","number of lattice sites in y direction",false,50,"int",cmd);
    ValueArg<int> lzSwitchArg("","Lz","number of lattice sites in z direction",false,50,"int",cmd);

    ValueArg<scalar> q0SwitchArg("q","q0","value of desired q0",false,.05,"scalar",cmd);
    ValueArg<scalar> pSwitchArg("p","radius","radius of sphere in units of Lx",false,.25,"scalar",cmd);

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
    scalar radiusFactor = pSwitchArg.getValue();
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
    dt = dt*pow(2/phaseB,.25);
    int maximumIterations = iterationsSwitchArg.getValue();
    bool GPU = false;
    if(myRank >= 0 && gpu >=0 && worldSize > 1)
        GPU = chooseGPU(myLocalRank);
    else if (gpu >=0)
        GPU = chooseGPU(gpu);

    logSpacedIntegers logIntTest(0,0.05);
    for (int tt = 0 ;tt < maximumIterations; ++tt)
        logIntTest.update();
    cout << "max iterations set to " << logIntTest.nextSave << endl;



    int3 rankTopology = partitionProcessors(worldSize);
    if(myRank ==0)
        printf("lattice divisions: {%i, %i, %i}\n",rankTopology.x,rankTopology.y,rankTopology.z);

    scalar a = -1;
    scalar b = -phaseB/phaseA;
    scalar c = phaseC/phaseA;
    noiseSource noise(reproducible);
    noise.setReproducibleSeed(13371+myRank+fileidx);
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
        printf("using 1-constant approximation: %f \n",L1);
        landauLCForce->setElasticConstants(L1,0,0);
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        }
    landauLCForce->setModel(Configuration);
    sim->addForce(landauLCForce);

    scalar forceCutoff=1e-12;
    int iterationsPerStep = 100;


    //shared_ptr<energyMinimizerGradientDescent> GDminimizer = make_shared<energyMinimizerGradientDescent>(Configuration);
    //GDminimizer->setMaximumIterations(1);
    shared_ptr<energyMinimizerFIRE> Fminimizer =  make_shared<energyMinimizerFIRE>(Configuration);
    Fminimizer->setMaximumIterations(1);
    if(minimizerSwitch ==1)
        {
        //GDminimizer->setGradientDescentParameters(dt,forceCutoff);
        //sim->addUpdater(GDminimizer,Configuration);
        //printf("gradient descent minimizer added\n");
        }
    else
        {
        scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.5;
        scalar alphaDec=0.9; int nMin=4;scalar alphaMin = .0;
        Fminimizer->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
        Fminimizer->setCurrentIterations(0);
        sim->addUpdater(Fminimizer,Configuration);
        //printf("FIRE minimizer added\n");
        }

    sim->setCPUOperation(!GPU);
    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
    S0=0.53;
    //FIX S0=0.53
    c = (2.0-b*S0)/(3.0*S0*S0);
    S0 = (-b+sqrt(b*b-24*a*c))/(6*c);

    landauLCForce->setPhaseConstants(a,b,c);

    printf("relative phase constants: %.3f  %.3f  %.3f  \t setting random configuration with S0 = %f\n",a,b,c,S0);
    Configuration->setNematicQTensorRandomly(noise,S0);
    sim->setCPUOperation(!GPU);
    printf("initialization done\n");

    boundaryObject homeotropicBoundary(boundaryType::homeotropic,5.8,S0);
    boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,0.58,S0);
    scalar3 left,center, right,down,up,direction;
    left.x = 0.0*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
    center.x = 0.5*boxLx;center.y = 0.5*boxLy;center.z = 0.5*boxLz;
    right.x = 1.*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
    //use program switches to define what objects are in the simulation
    scalar newRadius = floor(radiusFactor*boxLx);
    sim->createSphericalColloid(center,newRadius,homeotropicBoundary);
    direction.x=0;direction.y=0;direction.z=1.;
    if(programSwitch ==0)
        {
        sim->setDipolarField(center,PI, newRadius, 1.0*boxLx,S0);
        cout << "setting hedgehog initial conditions" << endl;
        }
    else
        {
        sim->setDipolarField(center,0.5*PI, newRadius, 1.0*boxLx,S0);
        cout << "setting quadrupolar initial conditions" << endl;
        }

/*
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
*/

    sim->finalizeObjects();
    char filename[256];
    sprintf(filename,"../data/hedgehogTesting_z%i_L%i_g%i_b%.3g_r%i.txt",programSwitch,boxLx,gpu,phaseB,(int)newRadius);
    ofstream myfile;
    myfile.open(filename);
    myfile.setf(ios_base::scientific);
    myfile << setprecision(10);
    profiler pMinimize("minimization");
    pMinimize.start();

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
    for (int tt = 0; tt < maximumIterations; ++tt)
        {
        sim->performTimestep();
        int iters;
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

        if(iters > 10)
            {
            //compute the energy, but don't count it towards the minimization time
            s1 = chrono::high_resolution_clock::now();
            E1 = sim->computePotentialEnergy();

            scalar hedgehogDistance =distanceFromDipolarConfiguration(Configuration,center,newRadius,PI,S0);
            scalar quadrupolarDistance =distanceFromDipolarConfiguration(Configuration,center,newRadius,0.5*PI,S0);
            e1 = chrono::high_resolution_clock::now();
            chrono::duration<double> del = e1-s1;
            remTime += del.count();

            endTime = chrono::high_resolution_clock::now();
            chrono::duration<double> difference = endTime-startTime;
            workingTime = difference.count() - remTime;
            printf("(iterations, E1, maxForce, workingTime, remTime)%i \t %g\t %g\t %g\t\t %g \n",iters,E1,maxForce,workingTime, remTime);
            printf("hedgehogDistance = %g\t saturn ring distance = %g\t \n",hedgehogDistance,quadrupolarDistance);
           myfile << iters <<"\t" << E1 <<"\t"<< maxForce <<"\t" << hedgehogDistance<< "\t" << quadrupolarDistance <<"\n";
            }
        currentIterationMax = logInts.nextSave;
        logInts.update();
        //currentIterationMax += minimizationIntervals[tt];
//        if(minimizerSwitch ==1)
//            GDminimizer->setMaximumIterations(currentIterationMax);
//        else
        Fminimizer->setMaximumIterations(currentIterationMax);
        }
    pMinimize.end();
    pMinimize.print();
    myfile.close();


    if(programSwitch ==0)
        {
        char savename[256];
        sprintf(savename,"../data/hhConfiguration_L%i_b%.3g_r%i",boxLx,phaseB,(int)newRadius);
        sim->p1.print();
        sim->saveState(savename);
        }

    /*
    scalar totalMinTime = pMinimize.timeTaken;
    scalar communicationTime = sim->p1.timeTaken;
    if(myRank != 0)
        printf("min  time %f\n comm time %f\n percent comm: %f\n",totalMinTime,communicationTime,communicationTime/totalMinTime);
    */
    /*
    cout << "size of configuration " << Configuration->getClassSize() << endl;
    cout << "size of force computer" << landauLCForce->getClassSize() << endl;
    cout << "size of fire updater " << Fminimizer->getClassSize() << endl;
    cout << "size of gd updater " << GDminimizer->getClassSize() << endl;
    */
    MPI_Finalize();
        return 0;
};
