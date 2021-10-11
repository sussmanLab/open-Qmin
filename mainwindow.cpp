#include <QMainWindow>
#include <QGuiApplication>
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
    ui->setDistortionConstants->hide();
    ui->setDistortionConstants5->hide();
    ui->fireParametersWidget->hide();
    ui->addObjectsWidget->hide();
    ui->fileImportWidget->hide();
    ui->fileSaveWidget->hide();
    ui->fileLoadWidget->hide();
    ui->nesterovWidget->hide();
    ui->applyFieldWidget->hide();
    ui->moveObjectWidget->hide();
    ui->colloidalTrajectoryWidget->hide();

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
    QString printable = QStringLiteral("Welcome to landauDeGUI, openQmin's graphical interface!");
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
    ui->dipoleWidget->hide();

    ui->xNormalSlider->hide();
    ui->yNormalSlider->hide();
    ui->zNormalSlider->hide();
    ui->drawPlanesCheckBox->hide();
    ui->xNormalCheckBox->hide();
    ui->yNormalCheckBox->hide();
    ui->zNormalCheckBox->hide();
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
    ui->xNormalSlider->show();
    ui->yNormalSlider->show();
    ui->zNormalSlider->show();
    ui->drawPlanesCheckBox->show();
    ui->xNormalCheckBox->show();
    ui->yNormalCheckBox->show();
    ui->zNormalCheckBox->show();
}

void MainWindow::on_initializeButton_released()
{
    BoxX = ui->boxXLine->text().toInt();
    BoxY = ui->boxYLine->text().toInt();
    BoxZ = ui->boxZLine->text().toInt();
    ui->xNormalSlider->setMaximum(BoxX-1);
    ui->yNormalSlider->setMaximum(BoxY-1);
    ui->zNormalSlider->setMaximum(BoxZ-1);

    QString dScaleAns = QString::number(round(10*0.075*BoxX)*0.1);

    QString lSkipAns = QString::number(floor(1.75 +0.0625*BoxX));
    ui->directorScaleBox->setText(dScaleAns);
    ui->latticeSkipBox->setText(lSkipAns);

    noise.Reproducible= ui->reproducibleButton->isChecked();
    ui->initializationFrame->hide();
    if(noise.Reproducible)
        {
        ui->reprodicbleRNGBox->setChecked(true);
        customFile.addLine("\tnoise.Reproducible = true;");
        }
    else
        {
        ui->reprodicbleRNGBox->setChecked(false);
        customFile.addLine("\tnoise.Reproducible = false;");
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
    S0 = (-B+sqrt(B*B-24*A*C))/(6*C);
    double sampleL1=2.32;
    double constK = 9.0*S0*S0*sampleL1*0.5;
    QString oneConstantKs = QString::number(constK, 'f', 3);
    ui->K1Box1->setText(oneConstantKs);
    ui->K1K2Box1->setText(oneConstantKs);
    ui->K1K2Box2->setText(oneConstantKs);
    ui->K1K2K3Box1->setText(oneConstantKs);
    ui->K1K2K3Box2->setText(oneConstantKs);
    ui->K1K2K3Box3->setText(oneConstantKs);
    ui->K24Box->setText(oneConstantKs);


    Configuration->setNematicQTensorRandomly(noise,S0);
    sprintf(lineBit,"\tConfiguration->setNematicQTensorRandomly(noise,%f);",S0);
    customFile.addLine(lineBit);

    landauLCForce->setPhaseConstants(A,B,C);
    sprintf(lineBit,"\tlandauLCForce->setPhaseConstants(%f,%f,%f);",A,B,C);
    customFile.addLine(lineBit);
    int nC = ui->nConstantsSpinBox->value();
    QString printable;
    switch(nC)
    {
        case 1:
            ui->K1checkBox->setChecked(true);
            ui->K12checkBox->setChecked(false);
            ui->K123checkBox->setChecked(false);
            ui->K24checkBox->setChecked(true);
            break;
        case 2:
            ui->K1checkBox->setChecked(false);
            ui->K12checkBox->setChecked(true);
            ui->K123checkBox->setChecked(false);
            break;
        case 3:
            ui->K1checkBox->setChecked(false);
            ui->K12checkBox->setChecked(false);
            ui->K123checkBox->setChecked(true);
            break;
    }
    ui->setDistortionConstants->show();

    printable = QStringLiteral("N %8 Lx %1 Ly %2 Lz %3 gpu %4... A %5 B %6 C %7 ")
                        .arg(BoxX).arg(BoxY).arg(BoxZ).arg(compDevice).arg(A).arg(B).arg(C).arg(Configuration->getNumberOfParticles());
    ui->testingBox->setText(printable);
    ui->progressBar->setValue(100);
    on_drawStuffButton_released();
    ui->xNormalSlider->setValue((int) (0.5*BoxX));
    ui->yNormalSlider->setValue((int) (0.5*BoxY));
    ui->zNormalSlider->setValue((int) (0.5*BoxZ));
}

void MainWindow::simulationInitialize()
{
     Configuration = make_shared<multirankQTensorLatticeModel>(BoxX,BoxY,BoxZ,false,false,false);
     sim = make_shared<multirankSimulation>(0,1,1,1,false,false);
     landauLCForce = make_shared<landauDeGennesLC>();

     sim->setConfiguration(Configuration);
     customFile.addLine("\tsim->setConfiguration(Configuration);");

     landauLCForce->setPhaseConstants(A,B,C);
     landauLCForce->setModel(Configuration);
     sim->addForce(landauLCForce);

     sprintf(lineBit,"\tlandauLCForce->setPhaseConstants(%f,%f,%f);",A,B,C);
     customFile.addLine(lineBit);
     customFile.addLine("\tlandauLCForce->setModel(Configuration);");
     customFile.addLine("\tsim->addForce(landauLCForce);");

     on_fireParamButton_released();
     ui->reproducibleButton->setEnabled(true);
}

void MainWindow::on_phaseS0Box_textEdited(const QString &arg1)
{
    A=ui->phaseABox->text().toDouble();
    B=ui->phaseBBox->text().toDouble();
    C=ui->phaseCBox->text().toDouble();
    S0=ui->phaseS0Box->text().toDouble();
    C = (2.0-B*S0)/(3.0*S0*S0);
    QString valueAsString = QString::number(C);
    ui->phaseCBox->setText(valueAsString);
}

void MainWindow::on_phaseBBox_textEdited(const QString &arg1)
{
    A=ui->phaseABox->text().toDouble();
    B=ui->phaseBBox->text().toDouble();
    C=ui->phaseCBox->text().toDouble();
    S0=ui->phaseS0Box->text().toDouble();
    C = (2.0-B*S0)/(3.0*S0*S0);
    QString valueAsString = QString::number(C);
    ui->phaseCBox->setText(valueAsString);
}

void MainWindow::on_setPhaseConstantsButton_released()
{
    A=ui->phaseABox->text().toDouble();
    B=ui->phaseBBox->text().toDouble();
    C=ui->phaseCBox->text().toDouble();
    landauLCForce->setPhaseConstants(A,B,C);

    sprintf(lineBit,"\tlandauLCForce->setPhaseConstants(%f,%f,%f);",A,B,C);
    customFile.addLine(lineBit);

    ui->setPhaseConstants->hide();
    QString printable = QStringLiteral("Phase constant sum is %1").arg((A+B+C));
    ui->testingBox->setText(printable);
    showControls();
}

void MainWindow::on_setDistortionConstantsButton_released()
{
    ui->setDistortionConstants->hide();

    L1=0;L2=0;L3=0;L4=0;L6=0;
    double k1, k2,k3,k24,q0;
    k1=k2=k3=k24=q0=0.;
    if(ui->K1checkBox->isChecked())
        {
        k1 = ui->K1Box1->text().toDouble();
        k2 = ui->K1Box1->text().toDouble();
        k3 = ui->K1Box1->text().toDouble();
        }
    if(ui->K12checkBox->isChecked())
        {
        k1 = ui->K1K2Box1->text().toDouble();
        k2 = ui->K1K2Box2->text().toDouble();
        k3 = ui->K1K2Box1->text().toDouble();
        }
    if(ui->K123checkBox->isChecked())
        {
        k1 = ui->K1K2K3Box1->text().toDouble();
        k2 = ui->K1K2K3Box2->text().toDouble();
        k3 = ui->K1K2K3Box3->text().toDouble();
        }
    if(ui->q0checkBox->isChecked())
        q0 = ui->q0Box->text().toDouble();
    if(ui->K24checkBox->isChecked())
        k24 = ui->K24Box->text().toDouble();

    L1=2.0*(k3-k1+3.*k2)/(27.*S0*S0);

    L2=4.0*(k1-k24)/(9.*S0*S0);

    L3=4.0*(k24-k2)/(9.*S0*S0);

    L4=-8.0*q0*k2/(9.*S0*S0);

    L6=4.0*(k3-k1)/(27.*S0*S0*S0);
    printf("setting elastic constants to L1 = %.5f, L2 = %.5f, L3 = %.5f, L4 = %f, L6 = %.5f\n",L1,L2,L3,L4,L6);
    if(L2 == 0 && L3 == 0 && L4 ==0 && L6 ==0)
        {
        cout << "using a one-constant approximation" << endl;
        landauLCForce->setElasticConstants(L1,0,0,0,0);
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        landauLCForce->setModel(Configuration);
        sprintf(lineBit,"\tlandauLCForce->setElasticConstants(%f,0,0,0,0);",L1);
        customFile.addLine(lineBit);
        customFile.addLine("\tlandauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);");
        customFile.addLine("\tlandauLCForce->setModel(Configuration);");
        }
    else
        {
        cout << "using a multi-constant approximation" << endl;
        landauLCForce->setElasticConstants(L1,L2,L3,L4,L6);
        landauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);
        landauLCForce->setModel(Configuration);
        sprintf(lineBit,"\tlandauLCForce->setElasticConstants(%f,%f,%f,%f,%f);",L1,L2,L3,L4,L6);
        customFile.addLine(lineBit);
        customFile.addLine("\tlandauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);");
        customFile.addLine("\tlandauLCForce->setModel(Configuration);");
        }
    QString printable = QStringLiteral("Landau-deGennes constants used: L1 %1 L2 %2 L3 %3 L4 %4 L6 %5").arg(L1).arg(L2).arg(L3).arg(L4).arg(L6);
    ui->testingBox->setText(printable);

    showControls();
}


void MainWindow::on_setFiveConstants_released()
{
    L1=ui->fiveConstantL1Box->text().toDouble();
    L2=ui->fiveConstantL2Box->text().toDouble();
    L3=ui->fiveConstantL3Box->text().toDouble();
    L4=ui->fiveConstantL4Box->text().toDouble();
    L6=ui->fiveConstantL6Box->text().toDouble();
    ui->setDistortionConstants5->hide();
    landauLCForce->setElasticConstants(L1,L2,L3,L4,L6);
    printf("setting elastic constants to L1 = %.5f, L2 = %.5f, L3 = %.5f, L4 = %f, L6 = %.5f\n",L1,L2,L3,L4,L6);
    sprintf(lineBit,"\tlandauLCForce->setElasticConstants(%f,%f,%f,%f,%f);",L1,L2,L3,L4,L6);
    customFile.addLine(lineBit);
    if(L2==0 && L3==0 && L4==0 && L6 ==0)
        {
        cout << "using a one-constant approximation" << endl;
        landauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);
        customFile.addLine("\tlandauLCForce->setNumberOfConstants(distortionEnergyType::oneConstant);");
        }
    else
        {
        cout << "using a multi-constant approximation" << endl;
        landauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);
        customFile.addLine("\tlandauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);");
        }

    landauLCForce->setNumberOfConstants(distortionEnergyType::multiConstant);
    QString printable = QStringLiteral("five-elastic-constant approximation set: L1 %1 L2 %2 L3 %3 L4 %4 L6 %5").arg(L1).arg(L2).arg(L3).arg(L4).arg(L6);
    ui->testingBox->setText(printable);
    landauLCForce->setModel(Configuration);
    showControls();

    customFile.addLine("\tlandauLCForce->setModel(Configuration);");
}

void MainWindow::on_fireParamButton_released()
{
    sim->clearUpdaters();
    fire = make_shared<energyMinimizerFIRE>(Configuration);
    sim->addUpdater(fire,Configuration);
    sim->setCPUOperation(!GPU);

    customFile.addLine("\tsim->clearUpdaters();");
    customFile.addLine("\tfire = make_shared<energyMinimizerFIRE>(Configuration);");

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


    sprintf(lineBit,"\tfire->setCurrentIterations(0);");
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tfire->setFIREParameters(%.10f,%f,%f,%f,%f,%f,%i,%.16f,%f);",dt,alphaStart,deltaTMax,deltaTInc,deltaTDec,alphaDec,nMin,forceCutoff,alphaMin);
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tfire->setMaximumIterations(%i);",maximumIterations);
    customFile.addLine(lineBit);
    customFile.addLine("\tsim->addUpdater(fire,Configuration);");
}

void MainWindow::on_minimizeButton_released()
{
    customFile.addLine("\t{auto upd = sim->updaters[0].lock();");
    sprintf(lineBit,"\tupd->setCurrentIterations(0);");
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tupd->setMaximumIterations(%i);}",maximumIterations);
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tsim->setCPUOperation(!sim->useGPU);");
    customFile.addLine(lineBit);
    customFile.addLine("\tsim->performTimestep();");

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
        int stepsToTake = maximumIterations;
        for (int ii = 1; ii <= 10; ++ii)
        {
            upd->setMaximumIterations(upd->getCurrentIterations()+stepsToTake/10);
            sim->performTimestep();
            QString printable2 = QStringLiteral("minimizing");
            ui->testingBox->setText(printable2);
            on_drawStuffButton_released();
            ui->progressBar->setValue(10*ii);
            ui->testingBox->setText(printable2);
        };
    };
    int iterationsTaken = upd->getCurrentIterations() - initialIterations;
    ui->progressBar->setValue(50);
    auto t2 = chrono::system_clock::now();
    chrono::duration<scalar> diff = t2-t1;
    ui->progressBar->setValue(75);

    ui->progressBar->setValue(80);
    on_drawStuffButton_released();
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
    fire->setCurrentIterations(0);
    if(noise.Reproducible)
        noise.setReproducibleSeed(13377);
    bool globalAlignment = ui->globalAlignmentCheckBox->isChecked();
    Configuration->setNematicQTensorRandomly(noise,S0,globalAlignment);

    if(globalAlignment)
        {
        sprintf(lineBit,"\tConfiguration->setNematicQTensorRandomly(noise,%f,true);",S0);
        customFile.addLine(lineBit);
        }
    else
        {
        sprintf(lineBit,"\tConfiguration->setNematicQTensorRandomly(noise,%f,false);",S0);
        customFile.addLine(lineBit);
        }


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

    customFile.addLine("\t{auto upd = sim->updaters[0].lock();");
    customFile.addLine("\tupd->setCurrentIterations(0);");
    sprintf(lineBit,"\tupd->setMaximumIterations(%i);}",maximumIterations);
    customFile.addLine(lineBit);
    customFile.addLine("\tsim->performTimestep();");

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
    /*
    printf("move chain:\n");
    for(int ii = 0; ii < moveChain.size(); ++ii)
        printf("{%i,%i,%i},",moveChain[ii].x,moveChain[ii].y,moveChain[ii].z);
    printf("\nmove chain end :\n");
    */
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
    //ui->testingBox->setText(printable1);
    if(ui->drawPlanesCheckBox->isChecked())
        {
        if(ui->xNormalCheckBox->isChecked())
            {
            int xPlane = ui->xNormalSlider->sliderPosition();
            for  (int yy = 0; yy < BoxY; yy += skip)
              for (int zz = 0; zz < BoxZ; zz += skip)
                {
                int3 curIdx; curIdx.x=xPlane;curIdx.y=yy;curIdx.z=zz;
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
            }
        if(ui->yNormalCheckBox->isChecked())
            {
            int yPlane = ui->yNormalSlider->sliderPosition();
            for  (int xx = 0; xx < BoxX; xx += skip)
              for (int zz = 0; zz < BoxZ; zz += skip)
                {
                int3 curIdx; curIdx.x=xx; curIdx.y=yPlane;curIdx.z=zz;
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
            }
        if(ui->zNormalCheckBox->isChecked())
            {
            int zPlane = ui->zNormalSlider->sliderPosition();
            for  (int xx = 0; xx < BoxX; xx += skip)
              for (int yy = 0; yy < BoxY; yy += skip)
                {
                int3 curIdx; curIdx.x=xx; curIdx.y=yy; curIdx.z=zPlane;
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
            }
        }
    else
        {
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
        };
    if(defectDraw)
    {
        QString printable2 = QStringLiteral("finding defects ");
        //ui->testingBox->setText(printable2);
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
    //QString printable3 = QStringLiteral("drawing stuff ");
    //ui->testingBox->setText(printable3);
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

void MainWindow::on_xNormalSlider_valueChanged(int value)
{
    if(ui->xNormalCheckBox->isChecked())
        on_drawStuffButton_released();
}


void MainWindow::on_yNormalSlider_valueChanged(int value)
{
    if(ui->yNormalCheckBox->isChecked())
        on_drawStuffButton_released();
}

void MainWindow::on_zNormalSlider_valueChanged(int value)
{
    if(ui->zNormalCheckBox->isChecked())
        on_drawStuffButton_released();
}

void MainWindow::on_addSphereButton_released()
{
    scalar3 spherePos;
    spherePos.x = ui->xSpherePosBox->text().toDouble()*BoxX;
    spherePos.y = ui->ySpherePosBox->text().toDouble()*BoxX;
    spherePos.z = ui->zSpherePosBox->text().toDouble()*BoxX;
    scalar rad = ui->sphereRadiusBox->text().toDouble()*BoxX;

    customFile.addLine("{scalar3 spherePos;");
    sprintf(lineBit,"\tspherePos.x =%f*globalLx;",ui->xSpherePosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tspherePos.y =%f*globalLy;",ui->ySpherePosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tspherePos.z =%f*globalLz;",ui->zSpherePosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tscalar rad = %f*globalLx;",ui->sphereRadiusBox->text().toDouble());
    customFile.addLine(lineBit);


    scalar W0 = ui->boundaryEnergyBox->text().toDouble();
    scalar s0b = ui->boundaryS0Box->text().toDouble();

    QString homeotropic ="homeotropic anchoring";
    QString planarDegenerate="planar degenerate anchoring";
    if(ui->anchoringComboBox->currentText() ==homeotropic)
        {
        boundaryObject homeotropicBoundary(boundaryType::homeotropic,W0,s0b);
        sim->createSphericalColloid(spherePos,rad,homeotropicBoundary);

        sprintf(lineBit,"\tboundaryObject homeotropicBoundary(boundaryType::homeotropic,%f,%f);",W0,s0b);
        customFile.addLine(lineBit);
        customFile.addLine("sim->createSphericalColloid(spherePos,rad,homeotropicBoundary);}");
        }
    else if(ui->anchoringComboBox->currentText() ==planarDegenerate)
        {
        boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,W0,s0b);
        sim->createSphericalColloid(spherePos,rad,planarDegenerateBoundary);

        sprintf(lineBit,"\tboundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,%f,%f);",W0,s0b);
        customFile.addLine(lineBit);
        customFile.addLine("sim->createSphericalColloid(spherePos,rad,planarDegenerateBoundary);}");
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
        sim->createWall(xyz,plane,homeotropicBoundary);

        sprintf(lineBit,"\t{boundaryObject homeotropicBoundary(boundaryType::homeotropic,%f,%f);",W0,s0b);
        customFile.addLine(lineBit);
        sprintf(lineBit,"\tsim->createWall(%i,%i,homeotropicBoundary);}",xyz,plane);
        customFile.addLine(lineBit);
        }
    else if(ui->anchoringComboBox->currentText() ==planarDegenerate)
        {
        boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,W0,s0b);
        sim->createWall(xyz,plane,planarDegenerateBoundary);
        wallType = 1;

        sprintf(lineBit,"\t{boundaryObject planarDegenerateBoundary(boundaryType::degeneratePlanar,%f,%f);",W0,s0b);
        customFile.addLine(lineBit);
        sprintf(lineBit,"\tsim->createWall(%i,%i,planarDegenerateBoundary);}",xyz,plane);
        customFile.addLine(lineBit);
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
    sim->finalizeObjects();
    customFile.addLine("\tsim->finalizeObjects();");
    QString printable1 = QStringLiteral("finished adding objects...for now!");
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
        {
        noise.setReproducibleSeed(13377);
        customFile.addLine("\tnoise.Reproducible= true;");
        customFile.addLine("\tsim->setReproducible(true);");
        }
    else
        {
        customFile.addLine("\tnoise.Reproducible= false;");
        customFile.addLine("\tsim->setReproducible(false);");
        }
    sim->setReproducible(repro);
}

void MainWindow::on_builtinBoundaryVisualizationBox_released()
{
    QString printable1 = QStringLiteral("Changing style of boundary visualization");
    //ui->testingBox->setText(printable1);
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
    if(fileExists(fn))
        {
        sim->createBoundaryFromFile(fn,true);cout.flush();
        sim->finalizeObjects();
        QString printable1 = QStringLiteral("boundary imported from file");
        ui->testingBox->setText(printable1);
        ui->builtinBoundaryVisualizationBox->setChecked(false);
        sprintf(lineBit,"\tsim->createBoundaryFromFile(\"%s\",true);cout.flush();",fn.c_str());
        customFile.addLine(lineBit);
        customFile.addLine("sim->finalizeObjects();");
        on_drawStuffButton_released();
        }
    else
        {
        QString printable1 = QStringLiteral("Requested boudnary file does not exist?");
        ui->testingBox->setText(printable1);
        }

}

void MainWindow::on_saveFileNowButton_released()
{
    QString fname = ui->saveFileNameBox->text();
    string fileName = fname.toStdString();
    sim->saveState(fileName);
    sprintf(lineBit,"\tsim->saveState(\"%s\",saveStride);",fileName.c_str());
    customFile.addLine(lineBit);
    QString printable1 = QStringLiteral("File saved");
    ui->testingBox->setText(printable1);
    ui->fileSaveWidget->hide();
}

void MainWindow::on_loadFileNowButton_released()
{
    QString fname = ui->loadFileNameBox->text();
    string fileName = fname.toStdString();
    sim->loadState(fileName);
    sprintf(lineBit,"\tsim->loadState(\"%s\");",fileName.c_str());
    customFile.addLine(lineBit);
    QString printable1 = QStringLiteral("File loaded");
    ui->testingBox->setText(printable1);
    ui->fileLoadWidget->hide();
    on_drawStuffButton_released();
}

void MainWindow::on_boxLSize_textEdited(const QString &arg1)
{
    ui->boxXLine->setText(arg1);
    ui->boxYLine->setText(arg1);
    ui->boxZLine->setText(arg1);
};

void MainWindow::on_nesterovParamButton_released()
{
    ui->nesterovWidget->hide();
    sim->clearUpdaters();
    nesterov = make_shared<energyMinimizerNesterovAG>(Configuration);
    sim->addUpdater(nesterov,Configuration);
    sim->setCPUOperation(!GPU);

    customFile.addLine("\tsim->clearUpdaters();");
    customFile.addLine("\tnesterov = make_shared<energyMinimizerNesterovAG>(Configuration);");

    ui->progressBar->setValue(0);
    scalar dt = ui->nesterovDtBox->text().toDouble();
    scalar mu = ui->nesterovMomentumBox->text().toDouble();
    nesterov->scheduledMomentum = ui->scheduleMomentumCheckbox->isChecked();

    if(ui->scheduleMomentumCheckbox->isChecked())
        customFile.addLine("\tnesterov->scheduledMomentum = true;");
    else
        customFile.addLine("\tnesterov->scheduledMomentum = false;");

    scalar forceCutoff=ui->nesterovForceCutoffBox->text().toDouble();
    maximumIterations = ui->nesterovMaxIterationsBox->text().toInt();
    nesterov->setCurrentIterations(0);
    nesterov->setNesterovAGParameters(dt,mu,forceCutoff);
    nesterov->setMaximumIterations(maximumIterations);

    sprintf(lineBit,"\tnesterov->setCurrentIterations(0);");
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tnesterov->setNesterovAGParameters(%.10f,%f,%.16f);",dt,mu,forceCutoff);
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tnesterov->setMaximumIterations(%i);",maximumIterations);
    customFile.addLine(lineBit);
    customFile.addLine("\tsim->addUpdater(nesterov,Configuration);");

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

    customFile.addLine("\t{scalar3 field;");
    sprintf(lineBit,"\tfield.x = %.10f;",field.x);
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tfield.y = %.10f;",field.y);
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tfield.z = %.10f;",field.z);
    customFile.addLine(lineBit);
    QString printable;
    if(ui->fieldTypeComboBox->currentText()==Ef)
        {
        landauLCForce->setEField(field,epsilon,epsilon0,deltaEpsilon);
        printable=QStringLiteral("E field set (%1 %2 %3) %4 %5 %6").arg(field.x).arg(field.y).arg(field.z).arg(epsilon).arg(epsilon0).arg(deltaEpsilon);

        sprintf(lineBit,"\tlandauLCForce->setEField(field,%.10f,%.10f,%.10f);}",epsilon,epsilon0,deltaEpsilon);
        customFile.addLine(lineBit);
        }
    if(ui->fieldTypeComboBox->currentText()==Hf)
        {
        landauLCForce->setHField(field,epsilon,epsilon0,deltaEpsilon);
        printable=QStringLiteral("H field set (%1 %2 %3) %4 %5 %6").arg(field.x).arg(field.y).arg(field.z).arg(epsilon).arg(epsilon0).arg(deltaEpsilon);

        sprintf(lineBit,"\tlandauLCForce->setHField(field,%.10f,%.10f,%.10f);}",epsilon,epsilon0,deltaEpsilon);
        customFile.addLine(lineBit);
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
    scalar totalEnergyPer = sim->computePotentialEnergy();
    printf("%f\n",totalEnergyPer);
    ui->progressBar->setValue(90);

    int nn = sim->NActive;
    if (nn == 0)
        {
        nn = BoxX*BoxY*BoxZ;
        if(!GPU)
            totalEnergyPer /= nn;
        }
    QString energyString = QStringLiteral("Total energy per site: %1").arg(totalEnergyPer);
    if(!GPU)
        {
        energyString += QStringLiteral(", components (phase, distortion, anchoring, E, H): ");
        for(int ii = 0; ii < landauLCForce->energyComponents.size();++ii)
            energyString += QStringLiteral(" %1,  ").arg(landauLCForce->energyComponents[ii]/nn);
        }
    ui->testingBox->setText(energyString);
    ui->progressBar->setValue(100);
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

void MainWindow::on_dipoleSetFieldButton_released()
{
    scalar3 center;
    scalar3 direction; direction.x=0;direction.y=0;direction.z=1;
    center.x = BoxX*ui->dipoleXPosBox->text().toDouble();
    center.y = BoxX*ui->dipoleYPosBox->text().toDouble();
    center.z = BoxX*ui->dipoleZPosBox->text().toDouble();
    scalar radius = BoxX*ui->dipoleRadiusBox->text().toDouble();
    scalar range = BoxX*ui->dipoleRangeBox->text().toDouble();
    scalar thetaD = PI*(ui->dipoleThetaDBox->text().toDouble());
    sim->setDipolarField(center,thetaD,radius,range,S0);
    //sim->setDipolarField(center,direction,radius,range,S0);
    ui->dipoleWidget->hide();

    customFile.addLine("\t{scalar3 center,direction;");
    customFile.addLine("\tdirection.x=0;direction.y=0;direction.z=1;");
    sprintf(lineBit,"\tcenter.x = %.10f*globalLx;",ui->dipoleXPosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tcenter.y = %.10f*globalLx;",ui->dipoleYPosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tcenter.z = %.10f*globalLx;",ui->dipoleZPosBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tscalar radius = %.10f*globalLx;",ui->dipoleRadiusBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tscalar range = %.10f*globalLx;",ui->dipoleRangeBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tscalar thetaD = %.10f*PI;",ui->dipoleThetaDBox->text().toDouble());
    customFile.addLine(lineBit);
    sprintf(lineBit,"\tsim->setDipolarField(center,%f,%f,%f,%f);}",thetaD,radius,range,S0);
    customFile.addLine(lineBit);
}

void MainWindow::startCustomFile()
{
    customFile.initialize();
}
void MainWindow::saveCustomFile()
{
    customFile.save();
}

void MainWindow::on_zNormalCheckBox_released()
{
    on_drawStuffButton_released();
}

void MainWindow::on_yNormalCheckBox_released()
{
    on_drawStuffButton_released();
}

void MainWindow::on_xNormalCheckBox_released()
{
    on_drawStuffButton_released();
}

void MainWindow::on_drawPlanesCheckBox_released()
{
    on_drawStuffButton_released();
}

void MainWindow::on_K1checkBox_released()
{
    if(ui->K1checkBox->isChecked())
        {
        ui->K12checkBox->setChecked(false);
        ui->K123checkBox->setChecked(false);
        }
}
void MainWindow::on_K12checkBox_released()
{
    if(ui->K12checkBox->isChecked())
        {
        ui->K1checkBox->setChecked(false);
        ui->K123checkBox->setChecked(false);
        }
}
void MainWindow::on_K123checkBox_released()
{
    if(ui->K123checkBox->isChecked())
        {
        ui->K12checkBox->setChecked(false);
        ui->K1checkBox->setChecked(false);
        }
}


