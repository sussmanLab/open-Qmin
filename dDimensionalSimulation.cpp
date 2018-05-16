#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>
#include "cuda_profiler_api.h"

#include "functions.h"
#include "gpuarray.h"
#include "periodicBoundaryConditions.h"
#include "simulation.h"
#include "simpleModel.h"
#include "baseUpdater.h"
#include "energyMinimizerFIRE.h"
#include "velocityVerlet.h"
#include "noseHooverNVT.h"
#include "noiseSource.h"
#include "harmonicRepulsion.h"
#include "lennardJones6_12.h"
#include "indexer.h"
#include "hyperrectangularCellList.h"
#include "neighborList.h"
#include "poissonDiskSampling.h"

using namespace std;
using namespace TCLAP;

scalar sphereVolume(scalar radius, int dimension)
    {
    if(dimension == 1)
        return 2*radius;
    else
        if(dimension == 2)
            return PI*radius*radius;
        else
            return (2.*PI*radius*radius)/((scalar) dimension)*sphereVolume(radius,dimension-2);
    };

/*!
command line parameters help identify a data directory and a filename... the output is a text file
(in the data/ directory rooted here) containing easy-to-read fourier transforms of the height-map
representation of the extremal interfaces for each point in time
*/
int main(int argc, char*argv[])
{
    // wrap tclap in a try block
    try
    {
    //First, we set up a basic command line parser...
    //      cmd("command description message", delimiter, version string)
    CmdLine cmd("basic testing of dDimSim", ' ', "V0.1");

    //define the various command line strings that can be passed in...
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,"value type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,-1,"int",cmd);
    ValueArg<int> nSwitchArg("n","Number","number of particles in the simulation",false,100,"int",cmd);
    ValueArg<int> maxIterationsSwitchArg("i","iterations","number of timestep iterations",false,100,"int",cmd);
    ValueArg<scalar> lengthSwitchArg("l","sideLength","size of simulation domain",false,10.0,"double",cmd);
    ValueArg<scalar> temperatureSwitchArg("t","temperature","temperature of simulation",false,.001,"double",cmd);

    //allow setting of system size by either volume fraction or density (assuming N has been set)
    ValueArg<scalar> phiSwitchArg("p","phi","volume fraction",false,-1.0,"double",cmd);
    ValueArg<scalar> rhoSwitchArg("r","rho","density",false,-1.0,"double",cmd);
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    scalar L = lengthSwitchArg.getValue();
    scalar Temperature = temperatureSwitchArg.getValue();
    scalar phi = phiSwitchArg.getValue();
    scalar rho = rhoSwitchArg.getValue();

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    if(phi >0)
        {
        L = pow(N*sphereVolume(.5,DIMENSION) / phi,(1.0/(scalar) DIMENSION));
        rho = N/pow(L,(scalar)DIMENSION);
        }
    else
        phi = N*sphereVolume(.5,DIMENSION) / pow(L,(scalar)DIMENSION);

    if(rho >0)
        {
        L = pow(((scalar)N/rho),(1.0/(scalar) DIMENSION));
        phi = rho * sphereVolume(.5,DIMENSION);
        }
    else
        rho = N/pow(L,(scalar)DIMENSION);


    int dim =DIMENSION;
    cout << "running a simulation in "<<dim << " dimensions with box sizes " << L << endl;
    cout << "density = " << rho << "\tvolume fracation = "<<phi<<endl;
    shared_ptr<simpleModel> Configuration = make_shared<simpleModel>(N);
    shared_ptr<periodicBoundaryConditions> PBC = make_shared<periodicBoundaryConditions>(L);

    shared_ptr<Simulation> sim = make_shared<Simulation>();
    sim->setConfiguration(Configuration);
    sim->setBox(PBC);

    //after the simulation box has been set, we can set particle positions...do so via poisson disk sampling?
    noiseSource noise(true);
    vector<dVec> poissonPoints;
    scalar diameter = .75;
    clock_t tt1=clock();
    int loopCount = 0;
    while(poissonPoints.size() != N)
        {
        poissonDiskSampling(N,diameter,poissonPoints,noise,PBC);
        loopCount +=1;
         diameter *= 0.95;
        }
    clock_t tt2=clock();
    scalar seedingTimeTaken = (tt2-tt1)/(scalar)CLOCKS_PER_SEC;
    cout << "disk sampling took "<< loopCount << " diameter attempts and took " << seedingTimeTaken << " total seconds" <<endl;

    Configuration->setParticlePositions(poissonPoints);
//    Configuration->setParticlePositionsRandomly(noise);
    scalar ke = Configuration->setVelocitiesMaxwellBoltzmann(Temperature,noise);
    printf("temperature input %f \t temperature calculated %f\n",Temperature,Configuration->computeInstantaneousTemperature());

    shared_ptr<neighborList> neighList = make_shared<neighborList>(1.,PBC);
    neighList->cellList->computeAdjacentCells();
     //monodisperse harmonic spheres
    shared_ptr<harmonicRepulsion> softSpheres = make_shared<harmonicRepulsion>();
    softSpheres->setMonodisperse();
    softSpheres->setNeighborList(neighList);
    vector<scalar> stiffnessParameters(1,1.0);
    softSpheres->setForceParameters(stiffnessParameters);
    sim->addForce(softSpheres,Configuration);
    /*
    //kob-anderson 80:20 mixture
    {
    ArrayHandle<int> h_t(Configuration->returnTypes());
    for (int ii = 0; ii < N; ++ii)
        if(ii < 0.8*N)
            h_t.data[ii] = 0;
        else
            h_t.data[ii] = 1;
    }
    shared_ptr<lennardJones6_12> lj = make_shared<lennardJones6_12>();
    lj->setNeighborList(neighList);
    vector<scalar> ljParams(8);
    ljParams[0]=1.0;ljParams[1]=1.5;ljParams[2]=1.5;ljParams[3]=0.5;
    ljParams[4]=1.0;ljParams[5]=0.8;ljParams[6]=0.8;ljParams[7]=0.88;
    lj->setForceParameters(ljParams);
    sim->addForce(lj,Configuration);
    */

    shared_ptr<noseHooverNVT> nvt = make_shared<noseHooverNVT>(Configuration,Temperature);
    nvt->setDeltaT(1e-8);
    sim->addUpdater(nvt,Configuration);

    if(gpuSwitch >=0)
        {
        sim->setCPUOperation(false);
//        Configuration->setGPU();
//        softSpheres->setGPU();
//        fire->setGPU();
//        neighList->setGPU();
        };
sim->performTimestep();
cout << "simulation set-up finished" << endl;cout.flush();

    clock_t t1 = clock();
    cudaProfilerStart();

    scalar dt=-12;
    for (int timestep = 0; timestep < maximumIterations; ++timestep)
        {
        if(timestep %1000 ==0 && dt < -3)
            {
            dt += 1;
            scalar newdt = pow(10,dt);
            nvt->setDeltaT(newdt);
            cout << "setting new timestep size of " <<newdt << endl;
            }
        sim->performTimestep();
        if(timestep%100 == 0)
            printf("timestep %i: target T = %f\t instantaneous T = %g\t PE = %g\t nlist max = %i\n",timestep,Temperature,Configuration->computeInstantaneousTemperature(),sim->computePotentialEnergy(),neighList->Nmax);
        };
    cudaProfilerStop();
    clock_t t2 = clock();
    sim->setCPUOperation(true);
    scalar E = sim->computePotentialEnergy();
    printf("simulation potential energy at %f\n",E);

    scalar timeTaken = (t2-t1)/(scalar)CLOCKS_PER_SEC/maximumIterations;
    cout << endl << "simulations took " << timeTaken << " per time step" << endl;

    ofstream ofs;
    char dataname[256];
    sprintf(dataname,"../data/timing_Phi%f_d%i_g%i.txt",phi,DIMENSION,gpuSwitch);
    ofs.open(dataname,ofstream::app);
    ofs << N <<"\t" << timeTaken << "\n";
    ofs.close();
/*
    t1 = clock();
    neighList->computeNeighborLists(Configuration->returnPositions());
    t2 = clock();
    scalar ntime = (t2-t1)/(scalar)CLOCKS_PER_SEC;
    cout << endl << "nlists take " << ntime << endl;
    t1 = clock();
    softSpheres->computeForces(Configuration->returnForces());
    t2 = clock();
    scalar ftime = (t2-t1)/(scalar)CLOCKS_PER_SEC - ntime;
    cout << endl << "forces take " << ftime << endl;
    t1 = clock();
    nve->performUpdate();
    t2 = clock();
    scalar stime = (t2-t1)/(scalar)CLOCKS_PER_SEC - ntime - ftime;
    cout << endl << "timestep takes" << stime << endl;

*/

//
//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
