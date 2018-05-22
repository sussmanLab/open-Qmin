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
#include "energyMinimizerAdam.h"
#include "noiseSource.h"
#include "harmonicRepulsion.h"
#include "indexer.h"
#include "hyperrectangularCellList.h"
#include "neighborList.h"

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
    scalar phiDest = 1.90225*exp(-(scalar)DIMENSION / 2.51907);
    ValueArg<scalar> phiSwitchArg("p","phi","volume fraction",false,phiDest,"double",cmd);
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

    //after the simulation box has been set, we can set particle positions
    noiseSource noise(true);
    Configuration->setParticlePositionsRandomly(noise);

    shared_ptr<neighborList> neighList = make_shared<neighborList>(1.,PBC);
    shared_ptr<harmonicRepulsion> softSpheres = make_shared<harmonicRepulsion>();
    softSpheres->setNeighborList(neighList);
    vector<scalar> stiffnessParameters(1,1.0);
    softSpheres->setForceParameters(stiffnessParameters);
    sim->addForce(softSpheres,Configuration);

    shared_ptr<energyMinimizerFIRE> fire = make_shared<energyMinimizerFIRE>(Configuration);
    fire->setFIREParameters(0.05,0.99,0.1,1.1,0.95,.9,4,1e-12);
    fire->setMaximumIterations(maximumIterations);

    shared_ptr<energyMinimizerAdam> adam  = make_shared<energyMinimizerAdam>();
    adam->setAdamParameters();
    adam->setMaximumIterations(maximumIterations);

    if(programSwitch == 0)
        sim->addUpdater(fire,Configuration);
    else
        sim->addUpdater(adam,Configuration);

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
    sim->performTimestep();
    clock_t t2 = clock();
    cudaProfilerStop();

    /*
    //how did FIRE do? check by hand
    {
    ArrayHandle<dVec> pos(Configuration->returnPositions());
    for (int pp = 0; pp < N; ++pp)
        printdVec(pos.data[pp]);
    }
    */
    cout << endl << "minimization took " << (t2-t1)/(scalar)CLOCKS_PER_SEC << endl;
    sim->setCPUOperation(true);
    scalar E = sim->computePotentialEnergy();
    printf("simulation potential energy at %f\n",E);
    MatrixDxD Pressure;
    cout << "built" << endl;
    sim->computePressureTensor(Pressure);
    cout << "computedbuilt" << endl;
    scalar p = trace(Pressure) / (1.0*DIMENSION);
    cout << "Pressure = " <<  p << endl;
    /*
    neighList->computeNeighborLists(Configuration->returnPositions());
    ArrayHandle<unsigned int> hnpp(neighList->neighborsPerParticle);
    ArrayHandle<int> hn(neighList->particleIndices);
    for (int ii = 0; ii < N; ++ii)
        {
        int neigh = hnpp.data[ii];
        cout << "particle "<<ii<<"'s neighbors: ";
        for (int nn = 0; nn < neigh; ++nn)
            {
            cout << hn.data[neighList->neighborIndexer(nn,ii)] <<", ";
            };
        cout << endl;
        };
*/
//
//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
