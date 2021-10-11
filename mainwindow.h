#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>

#include "functions.h"
#include "multirankSimulation.h"
#include "multirankQTensorLatticeModel.h"
#include "landauDeGennesLC.h"
#include "energyMinimizerFIRE.h"
#include "energyMinimizerNesterovAG.h"
#include "energyMinimizerLoLBFGS.h"
#include "energyMinimizerAdam.h"
#include "energyMinimizerGradientDescent.h"
#include "noiseSource.h"
#include "indexer.h"
#include "qTensorFunctions.h"
#include "latticeBoundaries.h"
#include "fileGenerator.h"


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
    void on_initializeButton_released();
    void on_setPhaseConstantsButton_released();
    void on_setDistortionConstantsButton_released();
    void on_setFiveConstants_released();
    
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

    void on_loadFileNowButton_released();

    void on_boxLSize_textEdited(const QString &arg1);

    void on_nesterovParamButton_released();

    void on_scheduleMomentumCheckbox_released();

    void on_cancelFieldButton_released();

    void on_setFieldButton_released();

    void on_fieldTypeComboBox_currentTextChanged(const QString &arg1);

    void on_computeEnergyButton_released();

    void moveObjectShow();

    void colloidalTrajectoryShow();

    void on_moveObjectButton_released();

    void on_cancelObjectFieldButton_released();

    void on_cancelTrajectoryButton_released();

    void on_linearTrajectoryButton_released();

    void on_dipoleSetFieldButton_released();

    void on_phaseS0Box_textEdited(const QString &arg1);

    void on_phaseBBox_textEdited(const QString &arg1);


    void startCustomFile();
    void saveCustomFile();

    void on_xNormalSlider_valueChanged(int value);

    void on_yNormalSlider_valueChanged(int value);

    void on_zNormalSlider_valueChanged(int value);

    void on_zNormalCheckBox_released();

    void on_yNormalCheckBox_released();

    void on_xNormalCheckBox_released();

    void on_drawPlanesCheckBox_released();

    void on_K1checkBox_released();
    void on_K12checkBox_released();
    void on_K123checkBox_released();

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
    double L1=0;
    double L2=0;
    double L3=0;
    double L4=0;
    double L6=0;
    double S0=0.0;
    int zoom = 1;
    vector<scalar3> spherePositions;
    vector<scalar> sphereRadii;

    noiseSource noise;

    shared_ptr<multirankQTensorLatticeModel> Configuration;
    shared_ptr<multirankSimulation> sim;
    shared_ptr<landauDeGennesLC> landauLCForce;
    shared_ptr<energyMinimizerFIRE> fire;
    shared_ptr<energyMinimizerNesterovAG> nesterov;

    vector<QString> computationalNames;

    fileGenerator customFile;
    char lineBit[256];
};

#endif // MAINWINDOW_H
