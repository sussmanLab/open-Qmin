
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
#include "ui_mainwindow.h"

MainWindow::~MainWindow()
{
    delete ui;
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->setPhaseConstants->hide();
    ui->setDistortionConstants1->hide();
    ui->setDistortionConstants2->hide();
    ui->setDistortionConstants3->hide();
    ui->fireParametersWidget->hide();
    ui->addObjectsWidget->hide();
    ui->fileImportWidget->hide();
    ui->fileSaveWidget->hide();
    vector<string> deviceNames;
    getAvailableGPUs(deviceNames);
    deviceNames.push_back("CPU");
    computationalNames.resize(deviceNames.size());
    for(int ii = 0; ii < computationalNames.size(); ++ii)
        {
        computationalNames[ii] = QString::fromStdString(deviceNames[ii]);
        ui->detectedGPUBox->insertItem(ii,computationalNames[ii]);
        }
    if(computationalNames.size()==1)//default to smaller simulations
    {
        ui->boxXLine->setText("20");
        ui->boxYLine->setText("20");
        ui->boxZLine->setText("20");

    }
    hideControls();
    QString printable = QStringLiteral("Welcome to landauDeGUI, a graphical interface to a continuum LdG liquid crystal simulation package!");
    ui->testingBox->setText(printable);
}

void MainWindow::hideControls()
{
    ui->resetQTensorsButton->hide();
    ui->minimizeButton->hide();
    ui->addObjectButton->hide();
    ui->minimizationParametersButton->hide();
    ui->addIterationsButton->hide();
    ui->addIterationsBox->hide();
    ui->displayZone->hide();
    ui->drawStuffButton->hide();
    ui->label_40->hide(); ui->label_42->hide();
    ui->latticeSkipBox->hide();
    ui->label_39->hide();
    ui->directorScaleBox->hide();
    ui->label_41->hide();ui->label_43->hide();ui->label_44->hide();ui->label_45->hide();
    ui->label_12->hide();ui->label_13->hide();ui->label_56->hide();ui->label_57->hide();
    ui->xRotSlider->hide();
    ui->zRotSlider->hide();
    ui->zoomSlider->hide();
    ui->visualProgressCheckBox->hide();
    ui->defectThresholdBox->hide();
    ui->defectDrawCheckBox->hide();
    ui->label_7->hide();
    ui->progressBar->hide();
    ui->reprodicbleRNGBox->hide();
    ui->globalAlignmentCheckBox->hide();
    ui->builtinBoundaryVisualizationBox->hide();
    ui->boundaryFromFileButton->hide();
}
void MainWindow::showControls()
{
    ui->defectDrawCheckBox->show();
    ui->resetQTensorsButton->show();
    ui->minimizeButton->show();
    ui->addObjectButton->show();
    ui->minimizationParametersButton->show();
    ui->addIterationsButton->show();
    ui->addIterationsBox->show();
    ui->displayZone->show();
    ui->drawStuffButton->show();
    ui->label_40->show();
    ui->latticeSkipBox->show();
    ui->label_39->show();ui->label_42->show();
    ui->label_12->show();ui->label_13->show();ui->label_56->show();ui->label_57->show();
    ui->directorScaleBox->show();
    ui->label_41->show();ui->label_43->show();ui->label_44->show();ui->label_45->show();
    ui->xRotSlider->show();
    ui->zRotSlider->show();
    ui->zoomSlider->show();
    ui->visualProgressCheckBox->show();
    ui->defectThresholdBox->show();
    ui->label_7->show();
    ui->progressBar->show();
    ui->reprodicbleRNGBox->show();
    ui->globalAlignmentCheckBox->show();
    ui->builtinBoundaryVisualizationBox->show();
    ui->boundaryFromFileButton->show();
}

void MainWindow::on_initializeButton_released()
{
    BoxX = ui->boxXLine->text().toInt();
    BoxY = ui->boxYLine->text().toInt();
    BoxZ = ui->boxZLine->text().toInt();
    noise.Reproducible= ui->reproducibleButton->isChecked();
    ui->initializationFrame->hide();
    if(noise.Reproducible)
        {
        ui->reprodicbleRNGBox->setChecked(true);
        }
    else
        {
        ui->reprodicbleRNGBox->setChecked(false);
        }

    int compDevice = ui->detectedGPUBox->currentIndex();
    if(compDevice==computationalNames.size()-1)//CPU branch
        {
        GPU = false;
        }
    else//gpu branch
        {
        GPU = chooseGPU(compDevice);
        }

    A=ui->initialPhaseA->text().toDouble();
    B=ui->initialPhaseB->text().toDouble();
    C=ui->initialPhaseC->text().toDouble();
    simulationInitialize();
    sim->setCPUOperation(!GPU);
    ui->progressBar->setValue(50);
    scalar S0 = (-B+sqrt(B*B-24*A*C))/(6*C);
    Configuration->setNematicQTensorRandomly(noise,S0);

    landauLCForce->setPhaseConstants(A,B,C);
    int nC = ui->nConstantsSpinBox->value();
    switch(nC)
    {
        case 1:
            ui->setDistortionConstants1->show();
            break;
        case 2:
            ui->setDistortionConstants2->show();
            break;
        case 3:
            ui->setDistortionConstants3->show();
            break;
    }

    QString printable = QStringLiteral("N %8 Lx %1 Ly %2 Lz %3 gpu %4... A %5 B %6 C %7 ")
                        .arg(BoxX).arg(BoxY).arg(BoxZ).arg(compDevice).arg(A).arg(B).arg(C).arg(Configuration->getNumberOfParticles());
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
    on_drawStuffButton_released();
}

void MainWindow::simulationInitialize()
{
     Configuration = make_shared<qTensorLatticeModel>(BoxX,BoxY,BoxZ);
     sim = make_shared<Simulation>();
     landauLCForce = make_shared<landauDeGennesLC>();

     sim->setConfiguration(Configuration);

     landauLCForce->setPhaseConstants(A,B,C);
     landauLCForce->setModel(Configuration);
     sim->addForce(landauLCForce);
     fire = make_shared<energyMinimizerFIRE>(Configuration);
     sim->addUpdater(fire,Configuration);
     on_fireParamButton_released();
     ui->reproducibleButton->setEnabled(true);
}

void MainWindow::on_setPhaseConstantsButton_released()
{
    A=ui->phaseABox->text().toDouble();
    B=ui->phaseBBox->text().toDouble();
    C=ui->phaseCBox->text().toDouble();
    landauLCForce->setPhaseConstants(A,B,C);
    ui->setPhaseConstants->hide();
    QString printable = QStringLiteral("Phase constant sum is %1").arg((A+B+C));
    ui->testingBox->setText(printable);
    showControls();
}

void MainWindow::on_setOneConstant_released()
{
    scalar _l1=ui->oneConstantL1Box->text().toDouble();
    ui->setDistortionConstants1->hide();
    landauLCForce->setElasticConstants(_l1,0,0);
    landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
    QString printable = QStringLiteral("One-elastic-constant approximation set: L1 %1").arg((L1));
    ui->testingBox->setText(printable);
    landauLCForce->setModel(Configuration);
    showControls();
}

void MainWindow::on_setTwoConstants_released()
{
    scalar _l1=ui->twoConstantL1Box->text().toDouble();
    scalar _l2=ui->twoConstantL2Box->text().toDouble();
    scalar _q0=ui->twoConstantQ0Box->text().toDouble();
    ui->setDistortionConstants2->hide();
    landauLCForce->setElasticConstants(_l1,_l2,_q0);
    landauLCForce->setNumberOfConstants(distortionEnergyType::twoConstant);
    QString printable = QStringLiteral("Two-elastic-constant approximation set: Lx %1 Ly %2 q0 %3 ").arg(_l1).arg(_l2).arg(_q0);
    ui->testingBox->setText(printable);
    landauLCForce->setModel(Configuration);
    showControls();
}

void MainWindow::on_setThreeConstants_released()
{
    scalar _l1=ui->threeConstantL1Box->text().toDouble();
    scalar _l2=ui->threeConstantL2Box->text().toDouble();
    scalar _l3=ui->threeConstantL3Box->text().toDouble();
    ui->setDistortionConstants3->hide();
    landauLCForce->setElasticConstants(_l1,_l2,_l3);
    landauLCForce->setNumberOfConstants(distortionEnergyType::threeConstant);
    QString printable = QStringLiteral("three-elastic-constant approximation set: L1 %1 L2 %2 L3 %3 ").arg(_l1).arg(_l2).arg(_l3);
    ui->testingBox->setText(printable);
    landauLCForce->setModel(Configuration);
    showControls();
}

void MainWindow::on_fireParamButton_released()
{
    ui->fireParametersWidget->hide();
    ui->progressBar->setValue(0);
    scalar dt = ui->dtBox->text().toDouble();
    scalar alphaStart= ui->alphaStartBox->text().toDouble();
    scalar deltaTMax=ui->dtMaxBox->text().toDouble();
    scalar deltaTInc=ui->dtIncBox->text().toDouble();
    scalar deltaTDec=ui->dtDecBox->text().toDouble();
    scalar alphaDec=ui->alphaDecBox->text().toDouble();
    int nMin=ui->nMinBox->text().toInt();
    scalar forceCutoff=ui->forceCutoffBox->text().toDouble();
    scalar alphaMin = ui->alphaMinBox->text().toDouble();
    maximumIterations = ui->maxIterationsBox->text().toInt();

    fire->setCurrentIterations(0);
    fire->setFIREParameters(dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
    fire->setMaximumIterations(maximumIterations);
    QString printable = QStringLiteral("Minimization parameters set, force cutoff of %1 dt between %2 and %3 chosen").arg(forceCutoff).arg(dt).arg(deltaTMax);
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);

}

void MainWindow::on_minimizeButton_released()
{
    bool graphicalProgress = ui->visualProgressCheckBox->isChecked();

    ui->progressBar->setValue(0);
    QString printable1 = QStringLiteral("minimizing");
    ui->testingBox->setText(printable1);
    clock_t t1 = clock();
    int initialIterations = fire->getCurrentIterations();
    if(!graphicalProgress)
        sim->performTimestep();
    else
    {
        int stepsToTake = fire->getMaxIterations();
        for (int ii = 1; ii <= 10; ++ii)
        {
            fire->setMaximumIterations(fire->getCurrentIterations()+stepsToTake/10);
            QString printable2 = QStringLiteral("minimizing");
            ui->testingBox->setText(printable2);
            sim->performTimestep();
            on_drawStuffButton_released();
            ui->progressBar->setValue(10*ii);
        };
    };
    int iterationsTaken = fire->getCurrentIterations() - initialIterations;
    ui->progressBar->setValue(50);
    clock_t t2 = clock();
    ui->progressBar->setValue(75);

    scalar E = sim->computePotentialEnergy();
    ui->progressBar->setValue(80);
    scalar time =1.0*(t2-t1)/(1.0*CLOCKS_PER_SEC)/iterationsTaken;
    scalar maxForce = fire->getMaxForce();
    QString printable = QStringLiteral("simulation energy per site at: %1...this took %2 for %3 steps...<f> = %4 ").arg(E)
                .arg(time).arg(fire->getCurrentIterations()-initialIterations).arg(maxForce);
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
}

void MainWindow::on_resetQTensorsButton_released()
{
    ui->progressBar->setValue(0);
    QString printable1 = QStringLiteral("resetting q tensors ");
    ui->progressBar->setValue(20);
    ui->testingBox->setText(printable1);
    ui->progressBar->setValue(40);
    scalar S0 = (-B+sqrt(B*B-24*A*C))/(6*C);
    ui->progressBar->setValue(60);
    bool globalAlignment = ui->globalAlignmentCheckBox->isChecked();
    Configuration->setNematicQTensorRandomly(noise,S0,globalAlignment);
    ui->progressBar->setValue(70);
    scalar E = sim->computePotentialEnergy();
    ui->progressBar->setValue(80);
    QString printable = QStringLiteral("simulation energy per site at: %1...").arg(E);
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
    if(ui->visualProgressCheckBox->isChecked())
        on_drawStuffButton_released();
}

void MainWindow::on_addIterationsButton_released()
{
    int additionalIterations = ui->addIterationsBox->text().toInt();
    maximumIterations += additionalIterations;
    fire->setMaximumIterations(maximumIterations);
    on_minimizeButton_released();
}

void MainWindow::on_drawStuffButton_released()
{
    ArrayHandle<dVec> Q(Configuration->returnPositions());
    ArrayHandle<int> types(Configuration->returnTypes());
    vector<scalar> eVals(3);
    vector<scalar> eVec1(3);
    vector<scalar> eVec2(3);
    vector<scalar> eVec3(3);

    int skip = ui->latticeSkipBox->text().toInt();
    scalar scale = ui->directorScaleBox->text().toDouble();
    bool defectDraw = ui->defectDrawCheckBox->isChecked();
    scalar defectCutoff = ui->defectThresholdBox->text().toDouble();

    int N = Configuration->getNumberOfParticles();
    int n = (int)floor(N/skip);
    vector<scalar3> lineSegments;
    vector<scalar3> defects;
    scalar3 director;
    QString printable1 = QStringLiteral("finding directors ");
    ui->testingBox->setText(printable1);
    for (int xx = 0; xx < BoxX; xx += skip)
         for (int yy = 0; yy < BoxY; yy += skip)
              for (int zz = 0; zz < BoxZ; zz += skip)
    {
        int3 curIdx; curIdx.x=xx;curIdx.y=yy;curIdx.z=zz;
        int ii = Configuration->latticeSiteToLinearIndex(curIdx);
        if(types.data[ii]>0)
            continue;
        eigensystemOfQ(Q.data[ii],eVals,eVec1,eVec2,eVec3);
        director.x=eVec3[0];
        director.y=eVec3[1];
        director.z=eVec3[2];

        int3 pos = Configuration->latticeIndex.inverseIndex(ii);
        scalar3 lineSegment1;
        scalar3 lineSegment2;

        lineSegment1.x = pos.x-0.5*scale*director.x;
        lineSegment2.x = pos.x+0.5*scale*director.x;
        lineSegment1.y = pos.y-0.5*scale*director.y;
        lineSegment2.y = pos.y+0.5*scale*director.y;
        lineSegment1.z = pos.z-0.5*scale*director.z;
        lineSegment2.z = pos.z+0.5*scale*director.z;

        lineSegments.push_back(lineSegment1);
        lineSegments.push_back(lineSegment2);
    }
    scalar e1,e2,e3;
    if(defectDraw)
    {
        QString printable2 = QStringLiteral("finding defects ");
        ui->testingBox->setText(printable2);
        Configuration->computeDefectMeasures(0);
        ArrayHandle<scalar> defectStrength(Configuration->defectMeasures);

        for (int ii = 0; ii < N; ++ii)
        {
            if(types.data[ii]>0 || defectStrength.data[ii]>defectCutoff)
                continue;
            int3 pos = Configuration->latticeIndex.inverseIndex(ii);
            scalar3 p;p.x=pos.x;p.y=pos.y;p.z=pos.z;
            defects.push_back(p);
        }
    }
    QString printable3 = QStringLiteral("drawing stuff ");
    ui->testingBox->setText(printable3);
    ui->displayZone->setLines(lineSegments,Configuration->latticeIndex.sizes);
    ui->displayZone->setDefects(defects,Configuration->latticeIndex.sizes);
    bool goodVisualization = ui->builtinBoundaryVisualizationBox->isChecked();
    if(goodVisualization)
        {
        ui->displayZone->setSpheres(Configuration->latticeIndex.sizes);
        vector<int3> bsites;
        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = true;
        }
    else
        {
        vector<int3> bsites;
        ArrayHandle<int> type(Configuration->returnTypes());
        for (int ii = 0 ;ii <Configuration->getNumberOfParticles();++ii )
            {
            if (type.data[ii] >0)
                bsites.push_back(Configuration->latticeIndex.inverseIndex(ii));
            }

        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = false;
        //printf("%lu sites\n",bsites.size());cout.flush();
        };
    ui->displayZone->update();
}

void MainWindow::on_xRotSlider_valueChanged(int value)
{
    ui->displayZone->setXRotation(value);
}

void MainWindow::on_zRotSlider_valueChanged(int value)
{
    ui->displayZone->setZRotation(value);
}

void MainWindow::on_zoomSlider_valueChanged(int value)
{
    zoom = value;
    ui->displayZone->zoom = zoom;
    ui->displayZone->setLines(ui->displayZone->lines,Configuration->latticeIndex.sizes);
    on_builtinBoundaryVisualizationBox_released();
    on_drawStuffButton_released();
//    ui->displayZone->update();
}

void MainWindow::on_addSphereButton_released()
{
    scalar3 spherePos;
    spherePos.x = ui->xSpherePosBox->text().toDouble()*BoxX;
    spherePos.y = ui->ySpherePosBox->text().toDouble()*BoxX;
    spherePos.z = ui->zSpherePosBox->text().toDouble()*BoxX;
    scalar rad = ui->sphereRadiusBox->text().toDouble()*BoxX;


    scalar W0 = ui->boundaryEnergyBox->text().toDouble();
    scalar s0b = ui->boundaryS0Box->text().toDouble();

    QString homeotropic ="homeotropic anchoring";
    QString planarDegenerate="planar degenerate anchoring";
    if(ui->anchoringComboBox->currentText() ==homeotropic)
        {
        boundaryObject homeotropicBoundary(boundaryType::homeotropic,W0,s0b);
        Configuration->createSimpleSpherialColloid(spherePos,rad, homeotropicBoundary);
        }
    else if(ui->anchoringComboBox->currentText() ==planarDegenerate)
        {
        boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,W0,s0b);
        Configuration->createSimpleSpherialColloid(spherePos,rad, planarDegenerateBoundary);
        }
    spherePositions.push_back(spherePos);
    sphereRadii.push_back(rad);
    QString printable1 = QStringLiteral("sphere added ");
    ui->testingBox->setText(printable1);
    ui->displayZone->addSphere(spherePos,rad);
}


void MainWindow::on_addWallButton_released()
{
    scalar W0 = ui->boundaryEnergyBox->text().toDouble();
    scalar s0b = ui->boundaryS0Box->text().toDouble();
    QString X ="x";
    QString Y ="y";
    QString Z ="z";
    int xyz=2;
    if(ui->wallNormalBox->currentText()==X)
           xyz=0;
    if(ui->wallNormalBox->currentText()==Y)
           xyz=1;
    if(ui->wallNormalBox->currentText()==Z)
           xyz=2;
    QString homeotropic ="homeotropic anchoring";
    QString planarDegenerate="planar degenerate anchoring";
    int plane = ui->wallPlaneBox->text().toInt();
    int wallType = 0;
    if(ui->anchoringComboBox->currentText() ==homeotropic)
        {
        boundaryObject homeotropicBoundary(boundaryType::homeotropic,W0,s0b);
        Configuration->createSimpleFlatWallNormal(plane,xyz,homeotropicBoundary);
        }
    else if(ui->anchoringComboBox->currentText() ==planarDegenerate)
        {
        boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,W0,s0b);
        Configuration->createSimpleFlatWallNormal(plane,xyz,planarDegenerateBoundary);
        wallType = 1;
        }
    int3 pnt; pnt.x = plane; pnt.y = xyz; pnt.z=wallType;
    ui->displayZone->addWall(pnt);

    QString printable1 = QStringLiteral("flat boundary added in direction %1 on plane %2").arg(xyz).arg(plane);
    ui->testingBox->setText(printable1);
}

void MainWindow::on_finishedWithObjectsButton_released()
{
    on_drawStuffButton_released();
    QString printable1 = QStringLiteral("finished adding objects...for now");
    ui->testingBox->setText(printable1);
}

void MainWindow::on_actionReset_the_system_triggered()
{
    hideControls();
    ui->displayZone->clearObjects();
    ui->addObjectsWidget->hide();
    ui->initializationFrame->show();
    ui->builtinBoundaryVisualizationBox->setChecked(true);
    QString printable1 = QStringLiteral("system reset");
    ui->testingBox->setText(printable1);
}

void MainWindow::on_reprodicbleRNGBox_stateChanged(int arg1)
{
    bool repro = ui->reprodicbleRNGBox->isChecked();
    noise.Reproducible= repro;
    sim->setReproducible(repro);
}

void MainWindow::on_builtinBoundaryVisualizationBox_released()
{
    /*
    if(!ui->builtinBoundaryVisualizationBox->isChecked())
    {
        vector<int3> bsites;
        ArrayHandle<int> type(Configuration->returnTypes());
        for (int ii = 0 ;ii <Configuration->getNumberOfParticles();++ii )
            {
            if (type.data[ii] >0)
                bsites.push_back(Configuration->latticeIndex.inverseIndex(ii));
            }

        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = false;
        printf("%lu sites\n",bsites.size());cout.flush();
    }
    else
        {
        vector<int3> bsites;
        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = true;
        }
        */
    QString printable1 = QStringLiteral("Changing style of boundary visualization");
    ui->testingBox->setText(printable1);
}

void MainWindow::on_importFileNowButton_released()
{
    ui->fileImportWidget->hide();
    QString fname = ui->fileNameBox->text();
    string fn = fname.toStdString();
    //string fn = "/Users/dmsussma/repos/dDimSim/data/boundaryInput.txt";
    Configuration->createBoundaryFromFile(fn,true);cout.flush();
    QString printable1 = QStringLiteral("boundary imported from file");
    ui->testingBox->setText(printable1);
}

void MainWindow::on_saveFileNowButton_released()
{
    QString fname = ui->saveFileNameBox->text();
    string fileName = fname.toStdString();

    ArrayHandle<dVec> pp(Configuration->returnPositions());
    ArrayHandle<int> tt(Configuration->returnTypes());
    ofstream myfile;
    myfile.open (fileName.c_str());
    for (int ii = 0; ii < Configuration->getNumberOfParticles();++ii)
        {
        int3 pos = Configuration->latticeIndex.inverseIndex(ii);
        myfile << pos.x <<"\t"<<pos.y<<"\t"<<pos.z<<"\t"<<pp.data[ii][0]<<"\t"<<pp.data[ii][1]<<"\t"<<
                pp.data[ii][2]<<"\t"<<pp.data[ii][3]<<"\t"<<pp.data[ii][4]<<"\t"<<tt.data[ii]<<"\n";
        };

    myfile.close();

    QString printable1 = QStringLiteral("File saved");
    ui->testingBox->setText(printable1);
}
