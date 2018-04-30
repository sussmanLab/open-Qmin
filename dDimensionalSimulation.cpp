#include "std_include.h" // std library includes, definition of scalar, etc.. has a "using namespace std" in it.
//#include "Matrix.h"//for when I might need 2x2 matrix manipulations
//#include "box.h"//plausible helpful for dealing with periodic boundary conditions in 2D
//#include "functions.h" // where a lot of the work happens

//we'll use TCLAP as our command line parser
#include <tclap/CmdLine.h>
#include "functions.h"
#include "gpuarray.h"
using namespace std;
using namespace TCLAP;

/*!
command line parameters help identify a data directory and a filename... the output is a text file
(in the data/ directory rooted here) containing easy-to-read fourier transforms of the height-map
representation of the extremal interfaces for each point in time
*/
int main(int argc, char*argv[])
{
    int dim =DIMENSION;
    cout << "running a simulation in "<<dim << " dimensions" << endl;

    // wrap tclap in a try block
    try
    {

    // cmd("command description message", delimiter, version string)
    CmdLine cmd("interface parsing and analyzing", ' ', "V0.0");
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,description of the type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    ValueArg<int> gpuSwitchArg("g","USEGPU","an integer controlling which gpu to use... g < 0 uses the cpu",false,-1,"int",cmd);
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    dVec tester;
    dVec dArrayZero = make_dVec(0.0);
    dVec dArrayOne(1.0);
    for (int dd = 0; dd < DIMENSION; ++dd)
        {
        tester.x[dd] = dd;
        };
    dArrayZero += tester;
    dArrayZero += tester;
    printdVec(dArrayZero);
    dArrayZero = dArrayOne + tester;
    printdVec(dArrayZero);
    cout << dot(dArrayZero,dArrayZero) << "    " << norm(dArrayZero) << endl;


//
//The end of the tclap try
//
    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
