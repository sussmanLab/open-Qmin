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
#include "noiseSource.h"
#include "harmonicRepulsion.h"
#include "indexer.h"
#include "hyperrectangularCellList.h"
#include "neighborList.h"

using namespace std;
using namespace TCLAP;

/*!
testing a basic NVE simulation
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
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int N = nSwitchArg.getValue();
    int maximumIterations = maxIterationsSwitchArg.getValue();
    scalar L = lengthSwitchArg.getValue();
    scalar Temperature = temperatureSwitchArg.getValue();
    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    int dim =DIMENSION;
    cout << "running a simulation in "<<dim << " dimensions" << endl;
    shared_ptr<simpleModel> Configuration = make_shared<simpleModel>(N);
    shared_ptr<periodicBoundaryConditions> PBC = make_shared<periodicBoundaryConditions>(L);

    shared_ptr<Simulation> sim = make_shared<Simulation>();
    sim->setConfiguration(Configuration);
    sim->setBox(PBC);

    //after the simulation box has been set, we can set particle positions
    noiseSource noise(true);
    Configuration->setParticlePositionsRandomly(noise);
    scalar ke = Configuration->setVelocitiesMaxwellBoltzmann(Temperature,noise);
    printf("temperature input %f \t temperature calculated %f\n",Temperature,Configuration->computeInstantaneousTemperature());

    shared_ptr<neighborList> neighList = make_shared<neighborList>(1.,PBC);
    shared_ptr<harmonicRepulsion> softSpheres = make_shared<harmonicRepulsion>();
    softSpheres->setNeighborList(neighList);
    vector<scalar> stiffnessParameters(1,1.0);
    softSpheres->setForceParameters(stiffnessParameters);
    sim->addForce(softSpheres,Configuration);

    neighList->cellList->computeAdjacentCells();

    shared_ptr<velocityVerlet> nve = make_shared<velocityVerlet>();
    nve->setDeltaT(0.002);
    sim->addUpdater(nve,Configuration);

    if(gpuSwitch >=0)
        {
        sim->setCPUOperation(false);
//        Configuration->setGPU();
//        softSpheres->setGPU();
//        fire->setGPU();
//        neighList->setGPU();
        };


    cudaProfilerStart();
    clock_t t1 = clock();
    for (int timestep = 0; timestep < maximumIterations; ++timestep)
        sim->performTimestep();
    clock_t t2 = clock();
    cudaProfilerStop();

    sim->setCPUOperation(true);
    scalar E = sim->computePotentialEnergy();
    printf("simulation potential energy at %f\n",E);
    /*
    //how did FIRE do? check by hand
    {
    ArrayHandle<dVec> pos(Configuration->returnPositions());
    for (int pp = 0; pp < N; ++pp)
        printdVec(pos.data[pp]);
    }
    */
    cout << endl << "simulations took " << (t2-t1)/(scalar)CLOCKS_PER_SEC << endl;
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
