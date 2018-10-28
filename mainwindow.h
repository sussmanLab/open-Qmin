#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "functions.h"
#include "gpuarray.h"
#include "simulation.h"
#include "qTensorLatticeModel.h"
#include "landauDeGennesLC.h"
#include "energyMinimizerFIRE.h"
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

    void on_yRotSlider_valueChanged(int value);

    void on_zRotSlider_valueChanged(int value);

    void on_zoomSlider_valueChanged(int value);

private:
    Ui::MainWindow *ui;

public:
    bool GPU = false;
    bool reproducible = true;
    int maximumIterations=0;

    scalar BoxX = 20;
    scalar BoxY = 20;
    scalar BoxZ = 20;
    double A=0;
    double B=0;
    double C=0;
    double L1=2.32;
    double L2=2.32;
    double L3=2.32;
    double q0=0.05;
    int zoom = 1;

    noiseSource noise;

    shared_ptr<qTensorLatticeModel> Configuration;
    shared_ptr<Simulation> sim;
    shared_ptr<landauDeGennesLC> landauLCForce;
    shared_ptr<energyMinimizerFIRE> fire;
};

#endif // MAINWINDOW_H
