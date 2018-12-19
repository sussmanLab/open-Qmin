#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>

#include "functions.h"
#include "gpuarray.h"
#include "multirankSimulation.h"
#include "simulation.h"
#include "qTensorLatticeModel.h"
#include "landauDeGennesLC.h"
#include "energyMinimizerFIRE.h"
#include "energyMinimizerNesterovAG.h"
#include "energyMinimizerLoLBFGS.h"
#include "energyMinimizerAdam.h"
#include "noiseSource.h"
#include "indexer.h"
#include "qTensorFunctions.h"
#include "latticeBoundaries.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void simulationInitialize();
    void hideControls();
    void showControls();

private slots:
    //parameter setting
    void on_setPhaseConstantsButton_released();
    void on_setOneConstant_released();
    void on_initializeButton_released();
    void on_setTwoConstants_released();
    void on_setThreeConstants_released();
    //simulation controls
    void on_minimizeButton_released();
    void on_resetQTensorsButton_released();
    void on_fireParamButton_released();
    void on_addIterationsButton_released();

    void on_drawStuffButton_released();

    void on_xRotSlider_valueChanged(int value);

    void on_zRotSlider_valueChanged(int value);

    void on_zoomSlider_valueChanged(int value);

    void on_addSphereButton_released();

    void on_finishedWithObjectsButton_released();

    void on_actionReset_the_system_triggered();

    void on_addWallButton_released();

    void on_reprodicbleRNGBox_stateChanged(int arg1);

    void on_builtinBoundaryVisualizationBox_released();

    void on_importFileNowButton_released();

    void on_saveFileNowButton_released();

    void on_boxLSize_textEdited(const QString &arg1);

    void on_multithreadingButton_released();

    void on_nesterovParamButton_released();

    void on_scheduleMomentumCheckbox_released();

    void on_cancelFieldButton_released();

    void on_setFieldButton_released();

    void on_fieldTypeComboBox_currentTextChanged(const QString &arg1);

    void on_computeEnergyButton_released();

    void moveObjectShow();

    void colloidalMobilityShow();

    void colloidalTrajectoryShow();

    void on_moveObjectButton_released();

    void on_cancelObjectFieldButton_released();

    void on_cancelObjectFieldButton_2_released();

    void on_colloidalEvolutionButtom_released();

    void on_colloidalImmobilityButtom_released();

    void on_colloidalMobilityButtom_released();

    void on_cancelTrajectoryButton_released();

    void on_linearTrajectoryButton_released();

    void on_lolbfgsParamButton_released();

private:
    Ui::MainWindow *ui;

public:
    bool GPU = false;
    bool reproducible = true;
    int maximumIterations=0;

    int iterationsPerColloidalEvolution = -1;
    double colloidalEvolutionPrefactor = 0.0;

    scalar BoxX = 20;
    scalar BoxY = 20;
    scalar BoxZ = 20;
    double A=0;
    double B=0;
    double C=0;
    int zoom = 1;
    vector<scalar3> spherePositions;
    vector<scalar> sphereRadii;

    noiseSource noise;

    shared_ptr<qTensorLatticeModel> Configuration;
    shared_ptr<Simulation> sim;
    shared_ptr<landauDeGennesLC> landauLCForce;
    shared_ptr<energyMinimizerFIRE> fire;
    shared_ptr<energyMinimizerNesterovAG> nesterov;
    shared_ptr<energyMinimizerLoLBFGS> lolbfgs;

    vector<QString> computationalNames;
};

#endif // MAINWINDOW_H
