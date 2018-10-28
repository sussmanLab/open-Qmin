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
    ui->label_40->hide();
    ui->latticeSkipBox->hide();
    ui->label_39->hide();
    ui->directorScaleBox->hide();
    ui->label_41->hide();ui->label_42->hide();ui->label_43->hide();ui->label_44->hide();ui->label_45->hide();
    ui->xRotSlider->hide();
    ui->yRotSlider->hide();
    ui->zRotSlider->hide();
    ui->zoomSlider->hide();
    ui->visualProgressCheckBox->hide();
    ui->defectThresholdBox->hide();
    ui->defectDrawCheckBox->hide();
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
    ui->label_39->show();
    ui->directorScaleBox->show();
    ui->label_41->show();ui->label_42->show();ui->label_43->show();ui->label_44->show();ui->label_45->show();
    ui->xRotSlider->show();
    ui->yRotSlider->show();
    ui->zRotSlider->show();
    ui->zoomSlider->show();
    ui->visualProgressCheckBox->show();
    ui->defectThresholdBox->show();
}

void MainWindow::on_initializeButton_released()
{
    ui->initializationFrame->hide();
    BoxX = ui->boxXLine->text().toInt();
    BoxY = ui->boxYLine->text().toInt();
    BoxZ = ui->boxZLine->text().toInt();
    noise.Reproducible= ui->reproducibleButton->isChecked();
    int gpu = ui->CPUORGPU->text().toInt();
    GPU = false;
    if(gpu >=0)
        GPU = chooseGPU(gpu);
    A=ui->initialPhaseA->text().toDouble();
    B=ui->initialPhaseB->text().toDouble();
    C=ui->initialPhaseC->text().toDouble();

    simulationInitialize();
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

    boundaryObject homeotropicBoundary(boundaryType::homeotropic,1.0,S0);
    boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,.582,S0);

    scalar3 left;
    left.x = 0.3*BoxX;left.y = 0.5*BoxY;left.z = 0.5*BoxZ;
    scalar3 right;
    right.x = 0.7*BoxX;right.y = 0.5*BoxY;right.z = 0.5*BoxZ;
    if(nC!= 2)
        {
            Configuration->createSimpleFlatWallZNormal(0, planarDegenerateBoundary);
            Configuration->createSimpleSpherialColloid(left,0.18*BoxX, homeotropicBoundary);
            Configuration->createSimpleSpherialColloid(right, 0.18*BoxX, homeotropicBoundary);
        };


    QString printable = QStringLiteral("N %8 Lx %1 Ly %2 Lz %3 gpu %4... A %5 B %6 C %7 ")
                        .arg(BoxX).arg(BoxY).arg(BoxZ).arg(gpu).arg(A).arg(B).arg(C).arg(Configuration->getNumberOfParticles());
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
}

void MainWindow::simulationInitialize()
{
     Configuration = make_shared<qTensorLatticeModel>(BoxX,BoxY,BoxZ);
     sim = make_shared<Simulation>();
     landauLCForce = make_shared<landauDeGennesLC>();

     sim->setConfiguration(Configuration);
     sim->setCPUOperation(!GPU);

     landauLCForce->setPhaseConstants(A,B,C);
     landauLCForce->setModel(Configuration);
     sim->addForce(landauLCForce);
     fire = make_shared<energyMinimizerFIRE>(Configuration);
     sim->addUpdater(fire,Configuration);
     on_fireParamButton_released();
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
            sim->performTimestep();
            on_drawStuffButton_released();
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
                .arg(time*fire->getCurrentIterations()).arg(fire->getCurrentIterations()).arg(maxForce);
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
    Configuration->setNematicQTensorRandomly(noise,S0);
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

    int skip = ui->latticeSkipBox->text().toInt();
    scalar scale = ui->directorScaleBox->text().toDouble();
    bool defectDraw = ui->defectDrawCheckBox->isChecked();
    scalar defectCutoff = ui->defectThresholdBox->text().toDouble();

    int N = Configuration->getNumberOfParticles();
    int n = (int)floor(N/skip);
    vector<scalar3> lineSegments;
    vector<scalar3> defects;

    vector<scalar> eVals(3);
    vector<scalar> eVec1(3);
    vector<scalar> eVec2(3);
    vector<scalar> eVec3(3);
    for (int ii = 0; ii < N; ii += skip)
    {
        if(types.data[ii]>0)
            continue;
        eigensystemOfQ(Q.data[ii],eVals,eVec1,eVec2,eVec3);
        scalar3 director;
        director.x = eVec3[0];
        director.y = eVec3[1];
        director.x = eVec3[2];
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
        for (int ii = 0; ii < N; ++ii)
        {
            if(types.data[ii]>0)
                continue;
            eigenvaluesOfQ(Q.data[ii],e1,e2,e3);

            int3 pos = Configuration->latticeIndex.inverseIndex(ii);
            scalar3 p;p.x=pos.x;p.y=pos.y;p.z=pos.z;
            if(max(max(e1,e2),e3) < defectCutoff)
                {
                defects.push_back(p);
            }
    }
    }
     QString printable1 = QStringLiteral("drawing stuff ");
     ui->testingBox->setText(printable1);
    ui->displayZone->setLines(lineSegments,Configuration->latticeIndex.sizes);
    ui->displayZone->setDefects(defects,Configuration->latticeIndex.sizes);

    ui->displayZone->update();
}

void MainWindow::on_xRotSlider_valueChanged(int value)
{
    ui->displayZone->setXRotation(value);
}

void MainWindow::on_yRotSlider_valueChanged(int value)
{
    ui->displayZone->setYRotation(value);
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
    on_drawStuffButton_released();
//    ui->displayZone->update();
}
