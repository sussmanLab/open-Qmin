#include "fileGenerator.h"
#include <iomanip>
#include <fstream>
#include <sstream>

/*! \file fileGenerator.cpp */


fileGenerator::fileGenerator()
    {
    string dir=DIRECTORY;
    string outName="/customScriptFromGUI.cpp";
    outputName = dir+outName;

    //the constructor first copies the front matter of the main openQmin file
    string inName=dir+"/openQmin.cpp";
    string line;
    ifstream infile(inName);
    for (int ll = 0; ll < 150; ++ll)
        {
        getline(infile,line);
        if(ll != 139 && ll != 138 && ll != 150)
            {
            line=line+"\n";
            lines.push_back(line);
            };
        };
    };

void fileGenerator::save()
    {
    char fn[256];
    sprintf(fn,"%s",outputName.c_str());
    cout << "saving to " <<outputName << endl;
    ofstream myfile;
    myfile.open(fn);
    for (int ii = 0; ii < lines.size(); ++ii)
        {
        myfile << lines[ii];
        }
    myfile << "\tMPI_Finalize();\n\treturn 0;\n\t};\n";
    myfile.close();
};
