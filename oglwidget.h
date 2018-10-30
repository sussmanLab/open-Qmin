#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QWidget>
#include <QMainWindow>
#include <QOpenGLWidget>
#include "/home/user/CGAL/CGAL-4.9/include/CGAL/glu.h"
#include "/home/user/CGAL/CGAL-4.9/include/CGAL/gl.h"
#include "std_include.h"

class OGLWidget : public QOpenGLWidget
{
public:
    OGLWidget(QWidget *parent = 0);
    ~OGLWidget();

    void clearObjects();
    void setLines(vector<scalar3> &lineSegments, int3 sizes);
    void setDefects(vector<scalar3> &def, int3 sizes);
    void setSpheres(int3 sizes);

    void addSphere(scalar3 &pos, scalar &radii);
    void addWall(int3 planeAndNormalAndType);
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

    vector<scalar3> lines;
    vector<scalar3> defects;
    int zoom=5;
    vector<int3> walls;
    vector<scalar3> baseSpherePositions;
    vector<scalar> baseSphereRadii;

    vector<scalar3> spherePositions;
    vector<scalar> sphereRadii;
    /*
public slots:
    // slots for xyz-rotation slider
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    */
//signals:
    // signaling rotation from mouse movement
//    void xRotationChanged(int angle);
//    void yRotationChanged(int angle);
//    void zRotationChanged(int angle);


protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void draw();
    void drawSpheres();
    void drawWalls();
    int iterator = 0;


    int xRot;
    int yRot;
    int zRot;
    int3 Sizes;


};

#endif // OGLWIDGET_H
