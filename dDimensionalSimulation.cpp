#include "std_include.h" // std library includes, definition of Dscalar, Dscalar2, etc.. has a "using namespace std" in it.
//#include "Matrix.h"//for when I might need 2x2 matrix manipulations
//#include "box.h"//plausible helpful for dealing with periodic boundary conditions in 2D
//#include "functions.h" // where a lot of the work happens

//we'll use TCLAP as our command line parser 
#include <tclap/CmdLine.h>
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
    // wrap tclap in a try block
    try
    {

	// cmd("command description message", delimiter, version string)
	CmdLine cmd("interface parsing and analyzing", ' ', "V0.0");
    //ValueArg<T> variableName("shortflag","longFlag","description",required or not, default value,description of the type",CmdLine object to add to
    ValueArg<int> programSwitchArg("z","programSwitch","an integer controlling program branch",false,0,"int",cmd);
    //parse the arguments
    cmd.parse( argc, argv );

    int programSwitch = programSwitchArg.getValue();
    cout << programSwitch << "    " <<dim +2 << endl;
    cout << programSwitch << "    " <<DIMENSION +2 << endl;

//The end of the tclap try
	} catch (ArgException &e)  // catch any exceptions
	{ cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }
    return 0;
};
