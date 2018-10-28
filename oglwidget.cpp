#include "oglwidget.h"
#include <QOpenGLFunctions>

OGLWidget::OGLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{}

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
        lines[ii].x = 2*((lines[ii].x-0.5*sizes.x)/sizes.x);
        lines[ii].y = 2*((lines[ii].y-0.5*sizes.y)/sizes.y);
        lines[ii].z = 2*((lines[ii].z-0.5*sizes.z)/sizes.z);
    }
}

void OGLWidget::paintGL()
{

    glBegin(GL_LINES);

    for (int ii = 0; ii < lines.size(); ++ii)
    {
        glColor3f(1.0, 0.0, 0.0);
        glVertex3f(lines[ii].x,lines[ii].y,lines[ii].z);
    }

    glEnd();
    // All lines lie in the xy plane.
    /*
    GLfloat z = 0.0f;
    for(GLfloat angle = 0.0; angle <= PI; angle += PI/20.0)
      {
      // Top half of the circle
      GLfloat x = 1.0f*sin(angle);
      GLfloat y = 1.0f*cos(angle);
      GLfloat z = sin(angle);
      if(iterator %2 == 0 )
        glColor3f(1.0, 0.0, 0.0);
      else
          glColor3f(0.0, 0.0, 1.0);
      glVertex3f(x, y, z);    // First endpoint of line

      // Bottom half of the circle
      x = 1.0f*sin(angle + PI);
      y = 1.0f*cos(angle + PI);
      z = sin(angle+PI);
      if(iterator %2 == 0 )
        glColor3f(0.0, 0.0, 1.0);
      else
          glColor3f(1.0, 0.0, 0.0);
      glVertex3f(x, y, z);    // Second endpoint of line
      }

    // Done drawing points
    glEnd();
    iterator += 1;
    */
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
