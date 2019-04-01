
#include <QMainWindow>
#include <QGuiApplication>

#include <Qt3DCore/QEntity>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QCameraLens>
#include <Qt3DCore/QTransform>
#include <Qt3DCore/QAspectEngine>

#include <Qt3DInput/QInputAspect>

#include <Qt3DRender/QRenderAspect>
//#include <Qt3DExtras/Qt3DWindow>
//#include <Qt3DExtras/QForwardRenderer>
//#include <Qt3DExtras/QPhongMaterial>
//#include <Qt3DExtras/QCylinderMesh>
//#include <Qt3DExtras/QSphereMesh>
//#include <Qt3DExtras/QTorusMesh>

#include <QPropertyAnimation>
#include <chrono>

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
    ui->multithreadingWidget->hide();
    ui->nesterovWidget->hide();
    ui->applyFieldWidget->hide();
    ui->moveObjectWidget->hide();
    ui->colloidalEvolutionWidget->hide();
    ui->colloidalTrajectoryWidget->hide();
    ui->LOLBFGSWidget->hide();

    connect(ui->displayZone,SIGNAL(xRotationChanged(int)),ui->xRotSlider,SLOT(setValue(int)));
    connect(ui->displayZone,SIGNAL(zRotationChanged(int)),ui->zRotSlider,SLOT(setValue(int)));

    vector<string> deviceNames;
    int nDev;
    cudaGetDeviceCount(&nDev);
    if(nDev >0)
        getAvailableGPUs(deviceNames);
    deviceNames.push_back("CPU");
    computationalNames.resize(deviceNames.size());
    for(unsigned int ii = 0; ii < computationalNames.size(); ++ii)
        {
        computationalNames[ii] = QString::fromStdString(deviceNames[ii]);
        ui->detectedGPUBox->insertItem(ii,computationalNames[ii]);
        }
    if(computationalNames.size()==1)//default to smaller simulations if only CPU is available
    {
        ui->boxXLine->setText("20");
        ui->boxYLine->setText("20");
        ui->boxZLine->setText("20");
        ui->boxLSize->setText("20");
    }
    hideControls();
    QString printable = QStringLiteral("Welcome to landauDeGUI, a graphical interface to a continuum LdG liquid crystal simulation package!");
    ui->testingBox->setText(printable);
}

void MainWindow::hideControls()
{
    ui->label_41->hide();ui->label_43->hide();ui->label_44->hide();ui->label_45->hide(); ui->label_40->hide(); ui->label_42->hide();
    ui->label_12->hide();ui->label_13->hide();ui->label_56->hide();ui->label_57->hide(); ui->label_7->hide(); ui->label_39->hide();
    ui->resetQTensorsButton->hide();
    ui->minimizeButton->hide();
    ui->addObjectButton->hide();
    ui->minimizationParametersButton->hide();
    ui->addIterationsButton->hide();
    ui->addIterationsBox->hide();
    ui->displayZone->hide();
    ui->drawStuffButton->hide();
    ui->latticeSkipBox->hide();
    ui->directorScaleBox->hide();
    ui->xRotSlider->hide();
    ui->zRotSlider->hide();
    ui->zoomSlider->hide();
    ui->visualProgressCheckBox->hide();
    ui->defectThresholdBox->hide();
    ui->defectDrawCheckBox->hide();
    ui->progressBar->hide();
    ui->reprodicbleRNGBox->hide();
    ui->globalAlignmentCheckBox->hide();
    ui->builtinBoundaryVisualizationBox->hide();
    ui->boundaryFromFileButton->hide();
    ui->nesterovMinimizationButton->hide();
    ui->computeEnergyButton->hide();
    ui->lolbfgsMinimizationButton->hide();
}
void MainWindow::showControls()
{
    ui->label_41->show();ui->label_43->show();ui->label_44->show();ui->label_45->show();
    ui->label_39->show();ui->label_42->show();ui->label_40->show();ui->label_7->show();
    ui->label_12->show();ui->label_13->show();ui->label_56->show();ui->label_57->show();
    ui->defectDrawCheckBox->show();
    ui->resetQTensorsButton->show();
    ui->minimizeButton->show();
    ui->addObjectButton->show();
    ui->minimizationParametersButton->show();
    ui->addIterationsButton->show();
    ui->addIterationsBox->show();
    ui->displayZone->show();
    ui->drawStuffButton->show();
    ui->latticeSkipBox->show();
    ui->directorScaleBox->show();
    ui->xRotSlider->show();
    ui->zRotSlider->show();
    ui->zoomSlider->show();
    ui->visualProgressCheckBox->show();
    ui->defectThresholdBox->show();
    ui->progressBar->show();
    ui->reprodicbleRNGBox->show();
    ui->globalAlignmentCheckBox->show();
    ui->builtinBoundaryVisualizationBox->show();
    ui->boundaryFromFileButton->show();
    ui->nesterovMinimizationButton->show();
    ui->computeEnergyButton->show();
    //ui->lolbfgsMinimizationButton->show();
}

void MainWindow::on_initializeButton_released()
{
    BoxX = ui->boxXLine->text().toInt();
    BoxY = ui->boxYLine->text().toInt();
    BoxZ = ui->boxZLine->text().toInt();

    QString dScaleAns = QString::number(round(10*0.075*BoxX)*0.1);

    QString lSkipAns = QString::number(floor(1.75 +0.0625*BoxX));
    ui->directorScaleBox->setText(dScaleAns);
    ui->latticeSkipBox->setText(lSkipAns);

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

    simulationInitialize();

    A=ui->initialPhaseA->text().toDouble();
    B=ui->initialPhaseB->text().toDouble();
    C=ui->initialPhaseC->text().toDouble();

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
    QString printable = QStringLiteral("One-elastic-constant approximation set: L1 %1").arg((_l1));
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
    sim->clearUpdaters();
    fire = make_shared<energyMinimizerFIRE>(Configuration);
    sim->addUpdater(fire,Configuration);
    sim->setCPUOperation(!GPU);
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
    auto upd = sim->updaters[0].lock();

    ui->progressBar->setValue(0);
    QString printable1 = QStringLiteral("minimizing");
    ui->testingBox->setText(printable1);
    auto t1 = chrono::system_clock::now();
    int initialIterations = upd->getCurrentIterations();
    if(!graphicalProgress)
        sim->performTimestep();
    else
    {
        int stepsToTake = upd->getMaxIterations();
        for (int ii = 1; ii <= 10; ++ii)
        {
            upd->setMaximumIterations(upd->getCurrentIterations()+stepsToTake/10);
            sim->performTimestep();
            on_drawStuffButton_released();
            ui->progressBar->setValue(10*ii);
            QString printable2 = QStringLiteral("minimizing");
            ui->testingBox->setText(printable2);
        };
    };
    int iterationsTaken = upd->getCurrentIterations() - initialIterations;
    ui->progressBar->setValue(50);
    auto t2 = chrono::system_clock::now();
    chrono::duration<scalar> diff = t2-t1;
    ui->progressBar->setValue(75);

    ui->progressBar->setValue(80);
    scalar maxForce = sim->getMaxForce();
    QString printable = QStringLiteral("minimization iterations took %2 total time for %3 steps...<f> = %4 ")
                .arg(diff.count()).arg(iterationsTaken).arg(maxForce);
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
    if(noise.Reproducible)
        noise.setReproducibleSeed(13377);
    bool globalAlignment = ui->globalAlignmentCheckBox->isChecked();
    Configuration->setNematicQTensorRandomly(noise,S0,globalAlignment);

    ui->progressBar->setValue(80);
    QString printable = QStringLiteral("Qtensor values reset at S0=%1...").arg(S0);
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
    if(ui->visualProgressCheckBox->isChecked())
        on_drawStuffButton_released();
}

void MainWindow::on_addIterationsButton_released()
{
    bool graphicalProgress = ui->visualProgressCheckBox->isChecked();
    ui->progressBar->setValue(0);

    int additionalIterations = ui->addIterationsBox->text().toInt();
    maximumIterations = additionalIterations;
    int subdivisions =  10;

    if (iterationsPerColloidalEvolution >0)
        subdivisions = additionalIterations / iterationsPerColloidalEvolution;

    int stepsPerSubdivision = additionalIterations / subdivisions;
    vector<int3> moveChain;
    for (int ii = 0; ii < subdivisions; ++ii)
        {
        {
        auto upd = sim->updaters[0].lock();
        int curIterations = upd->getCurrentIterations();
        upd->setMaximumIterations(curIterations+stepsPerSubdivision);
        }
        sim->performTimestep();
        if(iterationsPerColloidalEvolution > 0)
            {
            for (int bb = 0; bb < Configuration->boundaryState.size();++bb)
                {
                if(Configuration->boundaryState[bb]==1)
                    landauLCForce->computeObjectForces(bb);
                }
            for (int bb = 0; bb < Configuration->boundaryState.size();++bb)
                {
                if(Configuration->boundaryState[bb]==1)
                    {
                    scalar3 currentForce = Configuration->boundaryForce[bb];
                    int dirx = -10;int diry = -10;int dirz = -10;
                    bool moved = false;
                    if(colloidalEvolutionPrefactor*currentForce.x > 1 || colloidalEvolutionPrefactor*currentForce.x < -1)
                        {
                        dirx = (currentForce.x > 0) ? 1 : 0;
                        currentForce.x = 1;moved = true;
                        Configuration->boundaryForce[bb].x = 0;
                        };
                    if(colloidalEvolutionPrefactor*currentForce.y > 1|| colloidalEvolutionPrefactor*currentForce.y < -1)
                        {
                        diry = (currentForce.y > 0) ? 3 : 2;
                        currentForce.y = 1;moved = true;
                        Configuration->boundaryForce[bb].y = 0;
                        };
                    if(colloidalEvolutionPrefactor*currentForce.z > 1|| colloidalEvolutionPrefactor*currentForce.z < -1)
                        {
                        dirz = (currentForce.z > 0) ? 5 : 4;
                        currentForce.z = 1;moved = true;
                        Configuration->boundaryForce[bb].z = 0;
                        };
                    if(moved)
                        {
                        int xm=0; int ym = 0; int zm = 0;
                        if(dirx >= 0)
                            {
                            Configuration->displaceBoundaryObject(bb, dirx,1);
                            xm = (dirx ==0 ) ? -1 : 1;
                            }
                        if(diry >= 0)
                            {
                            Configuration->displaceBoundaryObject(bb, diry,1);
                            ym = (diry ==2) ? -1 : 1;
                            }
                        if(dirz >= 0)
                            {
                            Configuration->displaceBoundaryObject(bb, dirz,1);
                            zm = (dirz ==4) ? -1 : 1;
                            }
                        int3 thisMove; thisMove.x = xm; thisMove.y=ym;thisMove.z=zm;
                        moveChain.push_back(thisMove);
                        }
                    }
                }
            }//end check of colloidal moves

        if(graphicalProgress) on_drawStuffButton_released();
        int progress = ((1.0*ii/(1.0*subdivisions))*100);
        QString printable2 = QStringLiteral("evolving... %1 percent done").arg(progress);
        ui->testingBox->setText(printable2);
        ui->progressBar->setValue(progress);
        }
    scalar maxForce = sim->getMaxForce();
    QString printable3 = QStringLiteral("system evolved...mean force is %1").arg(maxForce);
    ui->testingBox->setText(printable3);
    ui->progressBar->setValue(100);
    printf("move chain:\n");
    for(int ii = 0; ii < moveChain.size(); ++ii)
        printf("{%i,%i,%i},",moveChain[ii].x,moveChain[ii].y,moveChain[ii].z);
    printf("\nmove chain end :\n");
}

void MainWindow::on_drawStuffButton_released()
{
    ArrayHandle<dVec> Q(Configuration->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<int> types(Configuration->returnTypes(),access_location::host,access_mode::read);
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
    if(defectDraw)
    {
        QString printable2 = QStringLiteral("finding defects ");
        ui->testingBox->setText(printable2);
        Configuration->computeDefectMeasures(0);
        ArrayHandle<scalar> defectStrength(Configuration->defectMeasures,access_location::host,access_mode::read);

        for (int ii = 0; ii < N; ++ii)
        {
            if(types.data[ii]>0 || defectStrength.data[ii]>defectCutoff)
                continue;
            int3 pos = Configuration->latticeIndex.inverseIndex(ii);
            scalar3 p;p.x=pos.x;p.y=pos.y;p.z=pos.z;
            defects.push_back(p);
        }
    }
    ui->displayZone->setLines(lineSegments,Configuration->latticeIndex.sizes);
    ui->displayZone->setDefects(defects,Configuration->latticeIndex.sizes);
    bool goodVisualization = ui->builtinBoundaryVisualizationBox->isChecked();
    if(goodVisualization)
        {
        ui->displayZone->setSpheres(Configuration->latticeIndex.sizes);
        ui->displayZone->drawBoundaries = true;
        }
    else
        {
        on_builtinBoundaryVisualizationBox_released();
        };
    QString printable3 = QStringLiteral("drawing stuff ");
    ui->testingBox->setText(printable3);
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
    if(pnt.x ==0)
    {
        int3 PntPeriodicCopy;PntPeriodicCopy.y=pnt.y;PntPeriodicCopy.z=pnt.z;
        if(xyz==0)
            PntPeriodicCopy.x = Configuration->latticeIndex.sizes.x;
        if(xyz==1)
            PntPeriodicCopy.x = Configuration->latticeIndex.sizes.y;
        if(xyz==2)
            PntPeriodicCopy.x = Configuration->latticeIndex.sizes.z;
        ui->displayZone->addWall(PntPeriodicCopy);
    }

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
    if(repro)
        noise.setReproducibleSeed(13377);
    sim->setReproducible(repro);
}

void MainWindow::on_builtinBoundaryVisualizationBox_released()
{
    QString printable1 = QStringLiteral("Changing style of boundary visualization");
    ui->testingBox->setText(printable1);
    if(!ui->builtinBoundaryVisualizationBox->isChecked())
    {
        int totalSize = 0;
        for (int bb = 0; bb < Configuration->boundarySites.size();++bb)
            totalSize += Configuration->boundarySites[bb].getNumElements();
        vector<int3> bsites;bsites.reserve(totalSize);

        for (int bb = 0; bb < Configuration->boundarySites.size();++bb)
            {
            ArrayHandle<int> site(Configuration->boundarySites[bb],access_location::host,access_mode::read);
            for(int ii = 0; ii < Configuration->boundarySites[bb].getNumElements();++ii)
                bsites.push_back(Configuration->latticeIndex.inverseIndex(site.data[ii]));
            }
        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = false;
    }
    else
        {
        vector<int3> bsites;
        ui->displayZone->setAllBoundarySites(bsites);
        ui->displayZone->drawBoundaries = true;
        }
}

void MainWindow::on_importFileNowButton_released()
{
    ui->fileImportWidget->hide();
    QString fname = ui->fileNameBox->text();
    string fn = fname.toStdString();
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
        myfile << pos.x <<"\t"<<pos.y<<"\t"<<pos.z;
        for (int dd = 0; dd <DIMENSION; ++dd)
            myfile <<"\t"<<pp.data[ii][dd];
        myfile << "\t"<<tt.data[ii]<<"\n";
        };

    myfile.close();
    QString printable1 = QStringLiteral("File saved");
    ui->testingBox->setText(printable1);
    ui->fileSaveWidget->hide();
}

void MainWindow::on_boxLSize_textEdited(const QString &arg1)
{
    ui->boxXLine->setText(arg1);
    ui->boxYLine->setText(arg1);
    ui->boxZLine->setText(arg1);
};

void MainWindow::on_multithreadingButton_released()
{
    ui->multithreadingWidget->hide();
    int nThreads = ui->multithreadingBox->text().toInt();
    sim->setNThreads(nThreads);
    QString printable1;
    if(nThreads ==1 )
        printable1 = QStringLiteral("requesting single-threaded operation");
    else
        printable1 = QStringLiteral("requesting %1 threads").arg(nThreads);
    ui->testingBox->setText(printable1);
}

void MainWindow::on_lolbfgsParamButton_released()
{
    ui->LOLBFGSWidget->hide();
    sim->clearUpdaters();
    lolbfgs = make_shared<energyMinimizerLoLBFGS>(Configuration);
    sim->addUpdater(lolbfgs,Configuration);
    sim->setCPUOperation(!GPU);
    ui->progressBar->setValue(0);
    scalar dt = ui->lolbfgsDtBox->text().toDouble();
    int M = ui->lolbfgsMBox->text().toInt();
    scalar forceCutoff=ui->lolbfgsForceCutoffBox->text().toDouble();
    maximumIterations = ui->lolbfgsMaxIterationsBox->text().toInt();
    lolbfgs->setCurrentIterations(0);
    lolbfgs->setLoLBFGSParameters(M, dt,1.0, forceCutoff);

    lolbfgs->setMaximumIterations(maximumIterations);

    QString printable = QStringLiteral("simple online LBFGS minimization parameters set, force cutoff of %1 dt of %2 and M %3 chosen for %4 steps %5").arg(forceCutoff).arg(dt).arg(M)
                                    .arg(maximumIterations).arg(lolbfgs->getMaxIterations());
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
}

void MainWindow::on_nesterovParamButton_released()
{
    ui->nesterovWidget->hide();
    sim->clearUpdaters();
    nesterov = make_shared<energyMinimizerNesterovAG>(Configuration);
    sim->addUpdater(nesterov,Configuration);
    sim->setCPUOperation(!GPU);
    ui->progressBar->setValue(0);
    scalar dt = ui->nesterovDtBox->text().toDouble();
    scalar mu = ui->nesterovMomentumBox->text().toDouble();
    nesterov->scheduledMomentum = ui->scheduleMomentumCheckbox->isChecked();
    scalar forceCutoff=ui->nesterovForceCutoffBox->text().toDouble();
    maximumIterations = ui->nesterovMaxIterationsBox->text().toInt();
    nesterov->setCurrentIterations(0);
    nesterov->setNesterovAGParameters(dt,mu,forceCutoff);
    nesterov->setMaximumIterations(maximumIterations);

    QString printable = QStringLiteral("nesterov minimization parameters set, force cutoff of %1 dt of %2 and momentum %3 chosen for %4 steps %5").arg(forceCutoff).arg(dt).arg(mu)
                                    .arg(maximumIterations).arg(nesterov->getMaxIterations());
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
}

void MainWindow::on_scheduleMomentumCheckbox_released()
{
    bool checkBoxChecked = ui->scheduleMomentumCheckbox->isChecked();
    ui->nesterovMomentumBox->setEnabled(!checkBoxChecked);
}

void MainWindow::on_cancelFieldButton_released()
{
    ui->applyFieldWidget->hide();
}

void MainWindow::on_setFieldButton_released()
{
    QString Ef ="E field";
    QString Hf ="H field";
    scalar3 field;
    field.x = ui->fieldXBox->text().toDouble();
    field.y = ui->fieldYBox->text().toDouble();
    field.z = ui->fieldZBox->text().toDouble();
    scalar epsilon = ui->eBox->text().toDouble();
    scalar epsilon0 = ui->e0Box->text().toDouble();
    scalar deltaEpsilon = ui->deBox->text().toDouble();

    QString printable;
    if(ui->fieldTypeComboBox->currentText()==Ef)
        {
        landauLCForce->setEField(field,epsilon,epsilon0,deltaEpsilon);
        printable=QStringLiteral("E field set (%1 %2 %3) %4 %5 %6").arg(field.x).arg(field.y).arg(field.z).arg(epsilon).arg(epsilon0).arg(deltaEpsilon);
        }
    if(ui->fieldTypeComboBox->currentText()==Hf)
        {
        landauLCForce->setHField(field,epsilon,epsilon0,deltaEpsilon);
        printable=QStringLiteral("H field set (%1 %2 %3) %4 %5 %6").arg(field.x).arg(field.y).arg(field.z).arg(epsilon).arg(epsilon0).arg(deltaEpsilon);
        }
    ui->applyFieldWidget->hide();
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
}

void MainWindow::on_fieldTypeComboBox_currentTextChanged(const QString &arg1)
{
    QString Ef ="E field";
    QString Hf ="H field";
    if(arg1==Ef)
        {
        ui->fieldXLabel->setText("Ex");
        ui->fieldYLabel->setText("Ey");
        ui->fieldZLabel->setText("Ez");
        ui->epsilonLabel->setText("epsilon");
        ui->epsilon0Label->setText("epsilon0");
        ui->deltaEpsilonLabel->setText("Delta epsilon");
        }
    if(arg1==Hf)
        {
        ui->fieldXLabel->setText("Hx");
        ui->fieldYLabel->setText("Hy");
        ui->fieldZLabel->setText("Hz");
        ui->epsilonLabel->setText("chi");
        ui->epsilon0Label->setText("mu0");
        ui->deltaEpsilonLabel->setText("Delta chi");
        }
}

void MainWindow::on_computeEnergyButton_released()
{
     ui->progressBar->setValue(0);
    landauLCForce->computeEnergy();
     ui->progressBar->setValue(90);
    scalar totalEnergy = 0.0;
    for(int ii = 0; ii < landauLCForce->energyComponents.size();++ii)
        totalEnergy+=landauLCForce->energyComponents[ii];
    QString energyString = QStringLiteral("Total energy: %1, components (phase, distortion, anchoring, E, H):  ").arg(totalEnergy);
    for(int ii = 0; ii < landauLCForce->energyComponents.size();++ii)
        energyString += QStringLiteral(" %1,  ").arg(landauLCForce->energyComponents[ii]);
    ui->testingBox->setText(energyString);
    ui->progressBar->setValue(100);
}

void MainWindow::colloidalMobilityShow()
{
    ui->colloidalMobilityBox->clear();
    ui->colloidalEvolutionWidget->show();
    vector<QString> objNames;
    for(unsigned int ii = 0; ii < Configuration->boundaries.getNumElements(); ++ii)
        ui->colloidalMobilityBox->insertItem(ii,QString::number(ii));
}

void MainWindow::colloidalTrajectoryShow()
{
    ui->colloidalTrajectoryIdxBox->clear();
    ui->colloidalTrajectoryWidget->show();
    vector<QString> objNames;
    for(unsigned int ii = 0; ii < Configuration->boundaries.getNumElements(); ++ii)
        ui->colloidalTrajectoryIdxBox->insertItem(ii,QString::number(ii));
}

void MainWindow::moveObjectShow()
{
    ui->objectIdxComboBox->clear();
    ui->moveObjectWidget->show();
    vector<QString> objNames;
    for(unsigned int ii = 0; ii < Configuration->boundaries.getNumElements(); ++ii)
        ui->objectIdxComboBox->insertItem(ii,QString::number(ii));
}
void MainWindow::on_cancelObjectFieldButton_released()
{
    ui->moveObjectWidget->hide();
}

void MainWindow::on_moveObjectButton_released()
{
    ui->moveObjectWidget->hide();
    int obj =ui->objectIdxComboBox->currentIndex();
    int dx=ui->moveXBox->text().toInt();
    int dy=ui->moveYBox->text().toInt();
    int dz=ui->moveZBox->text().toInt();
    int dxDir = (dx <0) ? 0 : 1;
    int dyDir = (dy <0) ? 2 : 3;
    int dzDir = (dz <0) ? 4 : 5;
    Configuration->displaceBoundaryObject(obj, dxDir,abs(dx));
    Configuration->displaceBoundaryObject(obj, dyDir,abs(dy));
    Configuration->displaceBoundaryObject(obj, dzDir,abs(dz));
    bool graphicalProgress = ui->visualProgressCheckBox->isChecked();
    if(graphicalProgress)
        on_drawStuffButton_released();
    QString translateString = QStringLiteral("object %1 translated by {%2 %3 %4}, components:  ").arg(obj)
                                .arg(dx).arg(dy).arg(dz);

    ui->testingBox->setText(translateString);
}

void MainWindow::on_cancelObjectFieldButton_2_released()
{
    ui->colloidalEvolutionWidget->hide();
}

void MainWindow::on_colloidalEvolutionButtom_released()
{
    iterationsPerColloidalEvolution = ui->colloidalEvolutionStepsBox->text().toInt();
    colloidalEvolutionPrefactor = ui->colloidalEvolutionPrefactorBox->text().toDouble();
    ui->colloidalEvolutionWidget->hide();
    QString mobilityString = QStringLiteral("forces on colloids set to be evaluated every %1 steps with %2").arg(iterationsPerColloidalEvolution)
                                        .arg(colloidalEvolutionPrefactor);
    ui->testingBox->setText(mobilityString);
}

void MainWindow::on_colloidalImmobilityButtom_released()
{
    int obj =ui->colloidalMobilityBox->currentIndex();
    Configuration->boundaryState[obj] = 0;
    QString mobilityString = QStringLiteral("mobility states set to {");
    for(int ii = 0; ii < Configuration->boundaryState.size();++ii)
        mobilityString += QStringLiteral(" %1,  ").arg(Configuration->boundaryState[ii]);
    mobilityString += QStringLiteral("}");
    ui->testingBox->setText(mobilityString);
}

void MainWindow::on_colloidalMobilityButtom_released()
{
    int obj =ui->colloidalMobilityBox->currentIndex();
    Configuration->boundaryState[obj] = 1;
    QString mobilityString = QStringLiteral("mobility states set to {");
    for(int ii = 0; ii < Configuration->boundaryState.size();++ii)
        mobilityString += QStringLiteral(" %1,  ").arg(Configuration->boundaryState[ii]);
    mobilityString += QStringLiteral("}");
    ui->testingBox->setText(mobilityString);
}

void MainWindow::on_cancelTrajectoryButton_released()
{
    ui->colloidalTrajectoryWidget->hide();
}

void MainWindow::on_linearTrajectoryButton_released()
{
    ui->moveObjectWidget->hide();
    int obj =ui->colloidalTrajectoryIdxBox->currentIndex();
    scalar dx=ui->xEndBox->text().toDouble();
    scalar dy=ui->yEndBox->text().toDouble();
    scalar dz=ui->zEndBox->text().toDouble();
    int dxDir = (dx <0) ? 0 : 1;
    int dyDir = (dy <0) ? 2 : 3;
    int dzDir = (dz <0) ? 4 : 5;

    int subdivisions = ui->subdivisionsBox->text().toInt();

    vector<int3> positions(subdivisions+1);
    vector<scalar> energyTrace(subdivisions+1);
    vector<scalar> forceNormTrace(subdivisions+1);
    positions[0].x = 0;positions[0].y = 0;positions[0].z = 0;
    //find nearest lattice sites
    for(int ii = 1; ii <= subdivisions; ++ii)
        {
        positions[ii].x = (int) (round(dx*(1.0*ii)/(1.0*subdivisions)) -round(dx*(1.0*(ii-1))/(1.0*subdivisions)));
        positions[ii].y = (int) (round(dy*(1.0*ii)/(1.0*subdivisions)) - round(dy*(1.0*(ii-1))/(1.0*subdivisions)));
        positions[ii].z = (int) (round(dz*(1.0*ii)/(1.0*subdivisions)) - round(dz*(1.0*(ii-1))/(1.0*subdivisions)));
        }

    bool graphicalProgress = ui->visualProgressCheckBox->isChecked();
    //minimize along the trajectory and store enrgy after each one
    for(int ii = 0; ii <=subdivisions;++ii)
        {
        Configuration->displaceBoundaryObject(obj, dxDir,abs(positions[ii].x));
        Configuration->displaceBoundaryObject(obj, dyDir,abs(positions[ii].y));
        Configuration->displaceBoundaryObject(obj, dzDir,abs(positions[ii].z));

        if(graphicalProgress)
            on_drawStuffButton_released();
        QString translateString = QStringLiteral("object %1 translated by {%2 %3 %4}, components:  ").arg(obj)
                                    .arg(positions[ii].x).arg(positions[ii].y).arg(positions[ii].z);
        ui->testingBox->setText(translateString);
        on_addIterationsButton_released();
        scalar maxForce = sim->getMaxForce();
        scalar currentEnergy = 0.0;
        landauLCForce->computeEnergy();
        for(int ii = 0; ii < landauLCForce->energyComponents.size();++ii)
            currentEnergy+=landauLCForce->energyComponents[ii];
        energyTrace[ii] = currentEnergy;
        forceNormTrace[ii] = maxForce;
        };

    //print answer to file
    QString fname = ui->fTrajNameBox->text();
    string fileName = fname.toStdString();
    ofstream myfile;
    myfile.open (fileName.c_str());
    myfile << "x \t"<<"y \t"<<"z \t";
    myfile << "    E \t\t"<<"<f> \n";
    for(int ii = 0; ii <=subdivisions;++ii)
        {
        int xx =  (int) round(dx*(1.0*ii)/(1.0*subdivisions));
        int yy =  (int) round(dy*(1.0*ii)/(1.0*subdivisions));
        int zz =  (int) round(dz*(1.0*ii)/(1.0*subdivisions));
        myfile << xx <<"\t"<<yy<<"\t"<<zz<<"\t";
        myfile << energyTrace[ii] <<"\t"<<forceNormTrace[ii]<<"\n";
        };
    myfile.close();
    QString traceString = QStringLiteral("energy trace saved to file:  ");
    traceString += fname;
    ui->testingBox->setText(traceString);

}
