#ifndef fileGenerator_H
#define fileGenerator_H


#include <string>
#include <iostream>
#include <vector>
using namespace std;
class fileGenerator
    {
    public:
        fileGenerator(){initialize();};

        void initialize();
        void save();

        void addLine(string _line)
            {
            lines.push_back(_line+"\n");
            };

        vector<string> lines;
        string outputName;
    };
#endif
