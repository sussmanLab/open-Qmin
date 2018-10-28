#include <QApplication>
#include <QMainWindow>
#include <QGuiApplication>

#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QCameraLens>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>

#include <Qt3DInput/QInputAspect>

#include <Qt3DRender/QRenderAspect>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DExtras/QTorusMesh>

#include <QPropertyAnimation>
#include "mainwindow.h"

Qt3DCore::QEntity *createScene()
{
    // Root entity
    Qt3DCore::QEntity *rootEntity = new Qt3DCore::QEntity;

    // Material
    Qt3DRender::QMaterial *material = new Qt3DExtras::QPhongMaterial(rootEntity);

    // Torus
    Qt3DCore::QEntity *torusEntity = new Qt3DCore::QEntity(rootEntity);
    Qt3DExtras::QTorusMesh *torusMesh = new Qt3DExtras::QTorusMesh;
    torusMesh->setRadius(5);
    torusMesh->setMinorRadius(1);
    torusMesh->setRings(100);
    torusMesh->setSlices(20);

    Qt3DCore::QTransform *torusTransform = new Qt3DCore::QTransform;
    torusTransform->setScale3D(QVector3D(1.5, 1, 0.5));
    torusTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 45.0f));

    torusEntity->addComponent(torusMesh);
    torusEntity->addComponent(torusTransform);
    torusEntity->addComponent(material);

    // Sphere
    Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity(rootEntity);
    Qt3DExtras::QSphereMesh *sphereMesh = new Qt3DExtras::QSphereMesh;
    sphereMesh->setRadius(3);


    sphereEntity->addComponent(sphereMesh);
    sphereEntity->addComponent(material);

    return rootEntity;
}


int main(int argc, char*argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    Qt3DExtras::Qt3DWindow view;

       Qt3DCore::QEntity *scene = createScene();

       // Camera
       Qt3DRender::QCamera *camera = view.camera();
       camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 1000.0f);
       camera->setPosition(QVector3D(0, 0, 40.0f));
       camera->setViewCenter(QVector3D(0, 0, 0));

    view.setRootEntity(scene);


    w.show();
  //    view.show();
    return a.exec();
    /*



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
