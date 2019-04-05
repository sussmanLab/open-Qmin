#include <QApplication>
#include <QMainWindow>
#include <QSplashScreen>
#include <QDesktopWidget>
#include <QTimer>
#include <QGuiApplication>

//#include <Qt3DCore/QEntity>
//#include <Qt3DRender/QCamera>
//#include <Qt3DRender/QCameraLens>
//#include <Qt3DCore/QTransform>
//#include <Qt3DCore/QAspectEngine>

//#include <Qt3DInput/QInputAspect>

//#include <Qt3DRender/QRenderAspect>
//#include <Qt3DExtras/Qt3DWindow>
//#include <Qt3DExtras/QForwardRenderer>
//#include <Qt3DExtras/QPhongMaterial>
//#include <Qt3DExtras/QCylinderMesh>
//#include <Qt3DExtras/QSphereMesh>
//#include <Qt3DExtras/QTorusMesh>

#include <QPropertyAnimation>
#include "mainwindow.h"
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
    printf("processes rank %i, local rank %i\n",myRank,myLocalRank);

//    printf("Hello world from processor %s, rank %d out of %d processors\n",
//           processorName, myRank, worldSize);


    //First, we set up a basic command line parser with some message and version
    CmdLine cmd("dDimSim applied to a lattice of XY-spins", ' ', "V0.5");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    SwitchArg visualSwitch("v","visualMode","run with the GUI", cmd, false);
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
    bool visualMode = visualSwitch.getValue();
    bool reproducible = reproducibleSwitch.getValue();
    int gpu = gpuSwitchArg.getValue();
    int programSwitch = programSwitchArg.getValue();
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

    if(visualMode) //activate the gui
        {
        if(myRank ==0)
            {
            QApplication a(argc, argv);
            QSplashScreen *splash = new QSplashScreen;
            splash->setPixmap(QPixmap("../landauDeGUI/examples/splashWithText.jpeg"));
            splash->show();
            MainWindow w;
            QRect screenGeometry = QApplication::desktop()->screenGeometry();
            int x = (screenGeometry.width()-w.width())/2;
            int y = (screenGeometry.height()-w.height())/2;
            w.move(x,y);
            QTimer::singleShot(750,splash,SLOT(close()));
            QTimer::singleShot(750,&w,SLOT(show()));
            MPI_Finalize();
            return a.exec();
            }
        MPI_Finalize();
        }
    else // otherwise, run off of the command line parameters
        {
        if(true)
            {
            cout << "non-visual mode activated on rank " << myRank << endl;
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

            /*
            scalar cValue = .0001; int mStorage = 8; scalar tau = 1000;
            shared_ptr<energyMinimizerLoLBFGS> lolbfgs = make_shared<energyMinimizerLoLBFGS>(Configuration);
            if(programSwitch == 1)
                {
                lolbfgs->setMaximumIterations(maximumIterations);
                lolbfgs->setLoLBFGSParameters(mStorage,dt,cValue,forceCutoff,tau);
                sim->addUpdater(lolbfgs,Configuration);
                }
            else
            */
            shared_ptr<energyMinimizerFIRE> fire =  make_shared<energyMinimizerFIRE>(Configuration);
            scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.95;
            scalar alphaDec=0.9; int nMin=4; scalar forceCutoff=1e-12; scalar alphaMin = 0.;
            fire->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
            fire->setMaximumIterations(maximumIterations);

            sim->addUpdater(fire,Configuration);

            sim->setCPUOperation(true);//have cpu and gpu initialized the same...for debugging
            //sim->setNThreads(nThreads);
            scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
            printf("setting random configuration with S0 = %f\n",S0);
            Configuration->setNematicQTensorRandomly(noise,S0);
            sim->setCPUOperation(!GPU);
            printf("initialization done\n");

        boundaryObject homeotropicBoundary(boundaryType::homeotropic,1.0,S0);
        scalar3 left,center, right;
        left.x = 0.0*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
        center.x = 1.0*boxLx;center.y = 0.5*boxLy;center.z = 0.5*boxLz;
        right.x = 1.5*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
        //sim->createSphericalColloid(left,4,homeotropicBoundary);
        sim->createSphericalColloid(center,5,homeotropicBoundary);
        //sim->createSphericalColloid(right,4,homeotropicBoundary);

        //sim->createWall(0, 5, homeotropicBoundary);
        sim->createWall(0, 0, homeotropicBoundary);

        sim->finalizeObjects();
        /*
        boundaryObject homeotropicBoundary(boundaryType::homeotropic,1.0,S0);
        scalar3 left;
        left.x = 0.3*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
        scalar3 center;
        left.x = 0.5*boxLx;left.y = 0.5*boxLy;left.z = 0.5*boxLz;
        scalar3 right;
        right.x = 0.7*boxLx;right.y = 0.5*boxLy;right.z = 0.5*boxLz;
        Configuration->createSimpleFlatWallNormal(0,1, homeotropicBoundary);
        */
            profiler pMinimize("minimization");
            pMinimize.start();
            sim->performTimestep();
            pMinimize.end();

            scalar E1 = sim->computePotentialEnergy(true);
            scalar maxForce = fire->getMaxForce();
            printf("minimized to %g\t E=%f\t\n",maxForce,E1);

            pMinimize.print();
            sim->p1.print();
            sim->saveState("../data/test");
            scalar totalMinTime = pMinimize.timeTaken;
            scalar communicationTime = sim->p1.timeTaken;
            if(myRank != 0)
                printf("min  time %f\n comm time %f\n percent comm: %f\n",totalMinTime,communicationTime,communicationTime/totalMinTime);

            /*
            cout << "size of configuration " << Configuration->getClassSize() << endl;
            cout << "size of force computer" << landauLCForce->getClassSize() << endl;
            cout << "size of fire updater " << fire->getClassSize() << endl;
            */
        //    cout << "size of (unused) lolbfgs updater " << lolbfgs->getClassSize() << endl;
          /*

        if(myRank%2 == 0) //send and receive
            {
            cout << "primary rank " << endl;
            ArrayHandle<dVec> p(Configuration->returnPositions());
            printdVec(p.data[Configuration->indexInExpandedDataArray(-1,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(-1,0,0)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(0,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(10,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(19,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(20,0,0)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(20,5,1)]);
            }
        else
            {
            cout << "other rank,... size " << Configuration->getNumberOfParticles() << " ... total size " << Configuration->returnPositions().getNumElements() <<  endl;
            ArrayHandle<int> ne(Configuration->neighboringSites);
            printf("neighbors of 0: %i %i %i %i %i %i\n",
                    ne.data[Configuration->neighborIndex(0,0)],
                    ne.data[Configuration->neighborIndex(1,0)],
                    ne.data[Configuration->neighborIndex(2,0)],
                    ne.data[Configuration->neighborIndex(3,0)],
                    ne.data[Configuration->neighborIndex(4,0)],
                    ne.data[Configuration->neighborIndex(5,0)]);
            ArrayHandle<dVec> p(Configuration->returnPositions());
            printf("idxs %i %i %i\n",
                    Configuration->indexInExpandedDataArray(-1,0,0),
                    Configuration->indexInExpandedDataArray(20,0,0),
                    Configuration->indexInExpandedDataArray(20,5,1));
            printdVec(p.data[Configuration->indexInExpandedDataArray(-1,0,0)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(-1,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(10,5,1)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(19,0,0)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(20,0,0)]);
            printdVec(p.data[Configuration->indexInExpandedDataArray(20,5,1)]);
            }
            */
        }
        /*
        landauLCForce->computeObjectForces(0);
        //landauLCForce->computeObjectForces(1);
        Configuration->displaceBoundaryObject(0,5,1);

        fire->setMaximumIterations(2*maximumIterations);
        sim->performTimestep();
        scalar E2 = sim->computePotentialEnergy(true);
        printf("e1 %f E2 %f\t force %f\n",E1,E2,E2-E1);
        landauLCForce->computeObjectForces(0);
        */


        /*
        int nn = Configuration->surfaceSites[0].getNumElements();
        ArrayHandle<int> surf1(Configuration->surfaceSites[0]);
        for (int ii = 0; ii < nn;++ii)
            printf("%i\t",surf1.data[ii]);
        printf("\n");
        */

        MPI_Finalize();
        return 0;
        }
    /*

    scalar a = -1;
    scalar b = -2.12/0.172;
    scalar c = 1.73/0.172;
    scalar l = 2.32;
    scalar l2 = 1.32;
    scalar l3 = 1.82;
    scalar q0 =0.01;

    scalar S0 = (-b+sqrt(b*b-24*a*c))/(6*c);
    cout << "S0 set at " << S0 << endl;
    noiseSource noise(true);
    scalar Lz=0.5*L;
    shared_ptr<qTensorLatticeModel> Configuration = make_shared<qTensorLatticeModel>(L,L,Lz);
    Configuration->setNematicQTensorRandomly(noise,S0);

    shared_ptr<Simulation> sim = make_shared<Simulation>();
    sim->setConfiguration(Configuration);

    shared_ptr<landauDeGennesLC> landauLCForceOneConstant = make_shared<landauDeGennesLC>(a,b,c,l);
    shared_ptr<landauDeGennesLC> landauLCForceTwoConstant = make_shared<landauDeGennesLC>
                                                (a,b,c,l,l2,q0,distortionEnergyType::twoConstant);
    shared_ptr<landauDeGennesLC> landauLCForceThreeConstant = make_shared<landauDeGennesLC>
                                                (a,b,c,l,l2,l3,distortionEnergyType::threeConstant);
    switch(Nconstants)
        {
        case 1 :
            landauLCForceOneConstant->setModel(Configuration);
            sim->addForce(landauLCForceOneConstant);
            break;
        case 2 :
            landauLCForceTwoConstant->setModel(Configuration);
            sim->addForce(landauLCForceTwoConstant);
            break;
        case 3 :
            landauLCForceThreeConstant->setModel(Configuration);
            sim->addForce(landauLCForceThreeConstant);
            break;
        default:
            cout << " you have asked for a force calculation ("<<Nconstants<<") which has not been coded" << endl;
            break;
        }
    sim->setNThreads(nThreads);

    boundaryObject homeotropicBoundary(boundaryType::homeotropic,1.0,S0);
    boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,.582,S0);

    scalar3 left;
    left.x = 0.3*L;left.y = 0.5*L;left.z = 0.5*Lz;
    scalar3 center;
    left.x = 0.5*L;left.y = 0.5*L;left.z = 0.5*Lz;
    scalar3 right;
    right.x = 0.7*L;right.y = 0.5*L;right.z = 0.5*Lz;
    if(Nconstants!= 2)
        {
        Configuration->createSimpleFlatWallNormal(0,1, homeotropicBoundary);
        Configuration->createSimpleFlatWallNormal(0,0, homeotropicBoundary);
        Configuration->createSimpleFlatWallNormal(0,2, homeotropicBoundary);
        Configuration->createSimpleSpherialColloid(left,0.18*L, homeotropicBoundary);
        Configuration->createSimpleSpherialColloid(right, 0.18*L, homeotropicBoundary);
        //Configuration->createSimpleSpherialColloid(center, 0.18*L, homeotropicBoundary);
        };


    //after the simulation box has been set, we can set particle positions... putting this here ensures that the random
    //spins are the same for gpu and cpu testing
    if(gpuSwitch >=0)
        {
        sim->setCPUOperation(false);
        };

    shared_ptr<energyMinimizerFIRE> fire = make_shared<energyMinimizerFIRE>(Configuration);
    scalar alphaStart=.99; scalar deltaTMax=100*dt; scalar deltaTInc=1.1; scalar deltaTDec=0.95;
    scalar alphaDec=0.9; int nMin=4; scalar forceCutoff=1e-12; scalar alphaMin = 0.7;
    fire->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
    fire->setMaximumIterations(maximumIterations);
       //if(programSwitch ==0)
        sim->addUpdater(fire,Configuration);
    //else //adam parameters not tuned yet... avoid this for now
    //    sim->addUpdater(adam,Configuration);

    if(gpuSwitch >=0)
        {
        sim->setCPUOperation(false);
        };
    scalar E = sim->computePotentialEnergy();
    printf("simulation energy at %f\n",E);

    Configuration->getAverageEigenvalues();
    printdVec(Configuration->averagePosition());

    sim->computeForces();

    int curMaxIt = maximumIterations;

    clock_t t1 = clock();
    cudaProfilerStart();
    sim->performTimestep();
    clock_t t2 = clock();
    cudaProfilerStop();

    Configuration->getAverageEigenvalues();
    printdVec(Configuration->averagePosition());

    cout << endl << "minimization (system size, time per iteration):"  << endl;
    printf("{%f,%f},\n",L,(t2-t1)/(scalar)CLOCKS_PER_SEC/maximumIterations);
    sim->setCPUOperation(true);
    E = sim->computePotentialEnergy();
    printf("simulation energy per site at %f\n",E);

    if (programSwitch >0)
        {
        ArrayHandle<dVec> pp(Configuration->returnPositions());
        ArrayHandle<int> tt(Configuration->returnTypes());
        char dataname[256];
        sprintf(dataname,"../data/twoWallsTest.txt");
        ofstream myfile;
        myfile.open (dataname);
        for (int ii = 0; ii < Configuration->getNumberOfParticles();++ii)
            {
            int3 pos = Configuration->latticeIndex.inverseIndex(ii);
            myfile << pos.x <<"\t"<<pos.y<<"\t"<<pos.z<<"\t"<<pp.data[ii][0]<<"\t"<<pp.data[ii][1]<<"\t"<<
                    pp.data[ii][2]<<"\t"<<pp.data[ii][3]<<"\t"<<pp.data[ii][4]<<"\t"<<tt.data[ii]<<"\n";
            };

        myfile.close();
        }
    string fname="../data/boundaryInput.txt";
    Configuration->createBoundaryFromFile(fname,true);
    */
    //Configuration->computeDefectMeasures(0);
    //Configuration->computeDefectMeasures(1);
    //Configuration->computeDefectMeasures(2);
    /*
    ArrayHandle<dVec> pp(Configuration->returnPositions());
    vector<scalar> eVals(3);
    vector<scalar> eVec1(3);
    vector<scalar> eVec2(3);
    vector<scalar> eVec3(3);
    printdVec(pp.data[10]);
    eigensystemOfQ(pp.data[10],eVals,eVec1,eVec2,eVec3);

    cout << endl << endl;
    printdVec(Configuration->averagePosition());
    dVec avePos=Configuration->averagePosition();
    eigensystemOfQ(avePos,eVals,eVec1,eVec2,eVec3);
    */

    /*
    if(GPU && Nconstants == 1)
        landauLCForceOneConstant->printTuners();
    if(GPU && Nconstants == 2)
        landauLCForceTwoConstant->printTuners();
    if(GPU && Nconstants == 3)
        landauLCForceThreeConstant->printTuners();
    */
};
