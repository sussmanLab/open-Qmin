#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it, because I'm lazy

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>
#include "functions.h"
#include "gpuarray.h"
#include "periodicBoundaryConditions.h"
#include "simulation.h"
#include "simpleModel.h"
#include "baseUpdater.h"
#include "energyMinimizerFIRE.h"
#include "noiseSource.h"
#include "harmonicBond.h"


using namespace std;
using namespace TCLAP;

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

    // cmd("command description message", delimiter, version string)
    CmdLine cmd("interface parsing and analyzing", ' ', "V0.0");
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,description of the type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,-1,"int",cmd);
    ValueArg<int> nSwitchArg("n","Number","number of particles in the simulation",false,100,"int",cmd);
    ValueArg<scalar> lengthSwitchArg("l","sideLength","size of simulation domain",false,10.0,"double",cmd);
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int N = nSwitchArg.getValue();
    scalar L = lengthSwitchArg.getValue();
    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    int dim =DIMENSION;
    cout << "running a simulation in "<<dim << " dimensions" << endl;
    shared_ptr<simpleModel> Configuration = make_shared<simpleModel>(N);
    shared_ptr<periodicBoundaryConditions> PBC = make_shared<periodicBoundaryConditions>(L);
    shared_ptr<updater> upd = make_shared<updater>(1);
    shared_ptr<energyMinimizerFIRE> fire = make_shared<energyMinimizerFIRE>(Configuration);

    shared_ptr<harmonicBond> bonds = make_shared<harmonicBond>();
    vector<simpleBond> blist;
    simpleBond testBond(0,1,1.7,1.0);
    blist.push_back(testBond);
    testBond.setBondIndices(5,6);
    testBond.setRestLength(1.2);
    blist.push_back(testBond);
    bonds->setBondList(blist);

    shared_ptr<Simulation> sim = make_shared<Simulation>();
    sim->setConfiguration(Configuration);
    sim->setBox(PBC);

    //after the simulation box has been set, we can set particle positions
    noiseSource noise(true);
    Configuration->setParticlePositionsRandomly(noise);
/*
    sim->moveParticles(Configuration->returnPositions());
    {
    ArrayHandle<dVec> pos(Configuration->returnPositions());
    for (int pp = 0; pp < N; ++pp)
        printdVec(pos.data[pp]);
    }
*/
    {
    ArrayHandle<dVec> pos(Configuration->returnPositions());
    for (int pp = 0; pp < N; ++pp)
        printdVec(pos.data[pp]);
    }

    cout << endl << endl;
    sim->addForce(bonds,Configuration);
    sim->addUpdater(fire,Configuration);
    sim->performTimestep();

    {
    ArrayHandle<dVec> pos(Configuration->returnPositions());
    for (int pp = 0; pp < N; ++pp)
        printdVec(pos.data[pp]);
    }


//
//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
