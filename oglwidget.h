#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QWidget>
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

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    int iterator = 0;
    vector<scalar3> lines;

};

#endif // OGLWIDGET_H
