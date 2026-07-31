#include <QApplication>
#include <QMainWindow>
#include <QSplashScreen>
#include <QScreen>
#include <QTimer>
#include <QGuiApplication>

#include <QPropertyAnimation>
#include "mainwindow.h"

int main(int argc, char*argv[])
    {
    MPI_Init(&argc, &argv);
    QApplication a(argc, argv);
    QSplashScreen *splash = new QSplashScreen;
    string dir=DIRECTORY;
    string assetName="/assets/splashWithText.jpeg";
    string splashPath = dir+assetName;
    splash->setPixmap(QPixmap(splashPath.c_str()).scaled(876,584));
    splash->show();
    MainWindow w;

    // Get the primary screen's geometry
    QScreen *screen = a.primaryScreen(); 
    QRect screenGeometry = screen->geometry(); 

    // Calculate center position
    int x = (screenGeometry.width() - w.width()) / 2;
    int y = (screenGeometry.height() - w.height()) / 2;

    // Move the window to the center
    w.move(x, y);

    QTimer::singleShot(750, splash, &QWidget::close);
    QTimer::singleShot(750, &w, &QWidget::show);
    /*QTimer::singleShot(750,splash,SLOT(close()));*/
    /*QTimer::singleShot(750,&w,SLOT(show()));*/
    return a.exec();
    MPI_Finalize();
    };
