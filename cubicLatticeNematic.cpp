#include <QApplication>
#include <QMainWindow>
#include "mainwindow.h"

/*!
Eventually: run minimization of a (3D) cubic lattice with a local Q-tensor
living on each lattice site, together with some sites enforcing boundary
conditions.

Will require that the compiled dimension in the CMakeList be 5
*/
int main(int argc, char*argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
    /*

    //define variables that correspond to the command line parameters
    int programSwitch = programSwitchArg.getValue();

    int gpuSwitch = gpuSwitchArg.getValue();
    bool GPU = false;
    if(gpuSwitch >=0)
        GPU = chooseGPU(gpuSwitch);

    int dim =DIMENSION;

    if(DIMENSION != 5)
        {
        cout << "oopsies. you're doing an q-tensor model with improperly defined dimenisonality. rethink your life choices" << endl;
        throw std::exception();
        }

    boundaryObject homeotropicBoundary(boundaryType::homeotropic,1.0,S0);
    boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,.582,S0);

    scalar3 left;
    left.x = 0.3*L;left.y = 0.5*L;left.z = 0.5*Lz;
    scalar3 right;
    right.x = 0.7*L;right.y = 0.5*L;right.z = 0.5*Lz;
    if(Nconstants!= 2)
        {
        Configuration->createSimpleFlatWallZNormal(0, planarDegenerateBoundary);
        Configuration->createSimpleSpherialColloid(left,0.18*L, homeotropicBoundary);
        Configuration->createSimpleSpherialColloid(right, 0.18*L, homeotropicBoundary);
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


    */
};
