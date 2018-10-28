#include "oglwidget.h"
#include <QOpenGLFunctions>


OGLWidget::OGLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;
}

OGLWidget::~OGLWidget()
{

}

void OGLWidget::initializeGL()
{
    glClearColor(0,0,0,1);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
}

void OGLWidget::setLines(vector<scalar3> &lineSegments, int3 sizes)
{
    lines =lineSegments;
    for (int ii = 0; ii < lines.size(); ++ii)
    {
        lines[ii].x = zoom*((lines[ii].x-0.5*sizes.x)/sizes.x);
        lines[ii].y = zoom*((lines[ii].y-0.5*sizes.y)/sizes.y);
        lines[ii].z = zoom*((lines[ii].z-0.5*sizes.z)/sizes.z);
    }
}

void OGLWidget::setDefects(vector<scalar3> &def, int3 sizes)
{
    defects =def;
    for (int ii = 0; ii < defects.size(); ++ii)
    {
        defects[ii].x = zoom*((defects[ii].x-0.5*sizes.x)/sizes.x);
        defects[ii].y = zoom*((defects[ii].y-0.5*sizes.y)/sizes.y);
        defects[ii].z = zoom*((defects[ii].z-0.5*sizes.z)/sizes.z);
    }
}

void OGLWidget::draw()
{
    glBegin(GL_LINES);
    for (int ii = 0; ii < lines.size(); ++ii)
    {
        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(lines[ii].x,lines[ii].y,lines[ii].z);
    }

    glEnd();
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(20.0);
    glBegin(GL_POINTS);
    for (int ii = 0; ii < defects.size(); ++ii)
    {
        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(defects[ii].x,defects[ii].y,defects[ii].z);
    }

    glEnd();
}

void OGLWidget::paintGL()
{
    glLoadIdentity();
        glTranslatef(0.0, 0.0, -10.0);
        glRotatef(xRot / 4.0, 1.0, 0.0, 0.0);
        glRotatef(yRot / 4.0, 0.0, 1.0, 0.0);
        glRotatef(zRot / 4.0, 0.0, 0.0, 1.0);
    draw();
}

void OGLWidget::resizeGL(int w, int h)
{
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, (float)w/h, 0.01, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0,0,5,0,0,0,0,1,0);
}

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360)
        angle -= 360 * 16;
}
/*
void OGLWidget::xRotationChanged(int angle)
{
}

void OGLWidget::yRotationChanged(int angle)
{
}

void OGLWidget::zRotationChanged(int angle)
{
}
*/

void OGLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        //emit xRotationChanged(angle);
        update();
    }
}

void OGLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        //emit yRotationChanged(angle);
        update();
    }
}

void OGLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        //emit zRotationChanged(angle);
        update();
    }
}
