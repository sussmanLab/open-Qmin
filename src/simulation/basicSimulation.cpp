#include "basicSimulation.h"
/*! \file basicSimulation.cpp */

/*!
Initialize all of the shared points, set default values of things
*/
basicSimulation::basicSimulation(): integerTimestep(0), Time(0.),integrationTimestep(0.01),spatialSortThisStep(false),
sortPeriod(-1),useGPU(false)
    {
    Box = make_shared<periodicBoundaryConditions>(1.0);
    };

/*!
Set a new Box for the simulation...This is the function that should be called to propagate a change
in the box dimensions throughout the simulation...
*/
void basicSimulation::setBox(BoxPtr _box)
    {
    Box = _box;
    auto Conf = configuration.lock();
    Conf->Box=Box;
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void basicSimulation::setCurrentTime(scalar _cTime)
    {
    Time = _cTime;
    //auto Conf = cellConfiguration.lock();
    //Conf->setTime(Time);
    };


