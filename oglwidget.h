#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QWidget>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <GLU.h>
#include <GL.h>
#include "std_include.h"

class OGLWidget : public QOpenGLWidget
{
public:
    OGLWidget(QWidget *parent = 0);
    ~OGLWidget();

    void setLines(vector<scalar3> &lineSegments, int3 sizes);
    void setDefects(vector<scalar3> &def, int3 sizes);
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

    /*
public slots:
    // slots for xyz-rotation slider
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    */
signals:
    // signaling rotation from mouse movement
//    void xRotationChanged(int angle);
//    void yRotationChanged(int angle);
//    void zRotationChanged(int angle);

       vector<scalar3> lines;
       vector<scalar3> defects;
       int zoom=5;
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void draw();
    int iterator = 0;


    int xRot;
    int yRot;
    int zRot;



};

#endif // OGLWIDGET_H
