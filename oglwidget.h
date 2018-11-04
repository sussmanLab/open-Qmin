#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QWidget>
#include <QMouseEvent>
#include <QMainWindow>
#include <QOpenGLWidget>
#include "glu.h"
#include "gl.h"
//#include "/home/user/CGAL/CGAL-4.9/include/CGAL/glu.h"
//#include "/home/user/CGAL/CGAL-4.9/include/CGAL/gl.h"
#include "std_include.h"

class OGLWidget : public QOpenGLWidget
{
    Q_OBJECT
public:
    OGLWidget(QWidget *parent = 0);
    ~OGLWidget();

    bool drawBoundaries = true;
    void setAllBoundarySites(vector<int3> &sites);
    void clearObjects();
    void setLines(vector<scalar3> &lineSegments, int3 sizes);
    void setDefects(vector<scalar3> &def, int3 sizes);
    void setSpheres(int3 sizes);

    void addSphere(scalar3 &pos, scalar &radii);
    void addWall(int3 planeAndNormalAndType);
    void setXRotation(int angle);
    void setZRotation(int angle);

    vector<scalar3> lines;
    vector<scalar3> defects;
    int zoom=4;
    vector<int3> walls;
    vector<scalar3> boundarySites;
    vector<scalar3> baseSpherePositions;
    vector<scalar> baseSphereRadii;

    vector<scalar3> spherePositions;
    vector<scalar> sphereRadii;

signals:
    void xRotationChanged(int XR);
    void zRotationChanged(int ZR);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void draw();
    void drawSpheres();
    void drawWalls();
    void drawBoundarySites();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    QPoint lastPos;
    int iterator = 0;

    int xRot;
    int zRot;
    int3 Sizes;
};

#endif // OGLWIDGET_H
