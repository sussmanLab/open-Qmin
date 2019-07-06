#include <QApplication>
#include <QMainWindow>
#include <QSplashScreen>
#include <QDesktopWidget>
#include <QTimer>
#include <QGuiApplication>

#include <QPropertyAnimation>
#include "mainwindow.h"

int main(int argc, char*argv[])
    {
    QApplication a(argc, argv);
    QSplashScreen *splash = new QSplashScreen;
    splash->setPixmap(QPixmap("../assets/splashWithText.jpeg").scaled(876,584));
    splash->show();
    MainWindow w;
    QRect screenGeometry = QApplication::desktop()->screenGeometry();
    int x = (screenGeometry.width()-w.width())/2;
    int y = (screenGeometry.height()-w.height())/2;
    w.move(x,y);
    QTimer::singleShot(750,splash,SLOT(close()));
    QTimer::singleShot(750,&w,SLOT(show()));
    return a.exec();
    };
