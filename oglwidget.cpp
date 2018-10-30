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

void OGLWidget::clearObjects()
{
    baseSpherePositions.clear();
    baseSphereRadii.clear();
    spherePositions.clear();
    sphereRadii.clear();
    walls.clear();
}
void OGLWidget::initializeGL()
{
    glClearColor(0,0,0,1);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_LIGHT0);
    //glEnable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glDisable(GL_LIGHTING);

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
}

void OGLWidget::setLines(vector<scalar3> &lineSegments, int3 sizes)
{
    Sizes=sizes;
    lines =lineSegments;
    for (int ii = 0; ii < lines.size(); ++ii)
    {
        lines[ii].x = zoom*((lines[ii].x-0.5*sizes.x)/sizes.z);
        lines[ii].y = zoom*((lines[ii].y-0.5*sizes.y)/sizes.z);
        lines[ii].z = zoom*((lines[ii].z-0.5*sizes.z)/sizes.z);
    }
}

void OGLWidget::setDefects(vector<scalar3> &def, int3 sizes)
{
    Sizes=sizes;
    defects =def;
    for (int ii = 0; ii < defects.size(); ++ii)
    {
        defects[ii].x = zoom*((defects[ii].x-0.5*sizes.x)/sizes.z);
        defects[ii].y = zoom*((defects[ii].y-0.5*sizes.y)/sizes.z);
        defects[ii].z = zoom*((defects[ii].z-0.5*sizes.z)/sizes.z);
    }
}

void OGLWidget::setSpheres(int3 sizes)
{
    spherePositions.resize(baseSpherePositions.size());
    sphereRadii.resize(baseSpherePositions.size());

    for (int ii = 0; ii < baseSpherePositions.size(); ++ii)
    {
        spherePositions[ii].x = zoom*((baseSpherePositions[ii].x-0.5*sizes.x)/sizes.z);
        spherePositions[ii].y = zoom*((baseSpherePositions[ii].y-0.5*sizes.y)/sizes.z);
        spherePositions[ii].z = zoom*((baseSpherePositions[ii].z-0.5*sizes.z)/sizes.z);
        sphereRadii[ii] = zoom*baseSphereRadii[ii]/sizes.z;
    }
}

void OGLWidget::addSphere(scalar3 &pos,scalar &radii)
{
    baseSpherePositions.push_back(pos);
    baseSphereRadii.push_back(radii);
}

void OGLWidget::addWall(int3 planeAndNormalAndType)
{
    walls.push_back(planeAndNormalAndType);
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
    //glEnable(GL_PROGRAM_POINT_SIZE);
    //glPointSize(20.0);
    //glBegin(GL_POINTS);
    float empiricallyNiceRadius = 0.25/(pow(zoom,0.25));
    for (int ii = 0; ii < defects.size(); ++ii)
    {
        glColor3f(0.0, 0.0, 1.0);
        GLUquadric *quad;
        quad = gluNewQuadric();
        glTranslatef(defects[ii].x,defects[ii].y,defects[ii].z);
        gluSphere(quad,empiricallyNiceRadius,20,20);
        glTranslatef(-defects[ii].x,-defects[ii].y,-defects[ii].z);
        //glVertex3f(defects[ii].x,defects[ii].y,defects[ii].z);
    }

    glEnd();
}

void OGLWidget::drawSpheres()
{
    glEnable (GL_BLEND);

    glBlendFunc (GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    //zoom*((defects[ii].x-0.5*sizes.x)/sizes.x)
    for (int ii = 0; ii < spherePositions.size(); ++ii)
    {
        glColor4f(0.4,0.4,0,0.9);
        GLUquadric *quad;
        quad = gluNewQuadric();
        glTranslatef(spherePositions[ii].x,spherePositions[ii].y,spherePositions[ii].z);
        gluSphere(quad,sphereRadii[ii],100,20);
        glTranslatef(-spherePositions[ii].x,-spherePositions[ii].y,-spherePositions[ii].z);
    }
    glDisable (GL_BLEND);
}

void OGLWidget::drawWalls()
{
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    float zTop = zoom*(0.5);
    float yTop = zoom*Sizes.y/Sizes.z;
    float xTop = zoom*Sizes.x/Sizes.z;
    glBegin(GL_QUADS);
    for(int ww = 0; ww < walls.size(); ++ww)
    {
        int3 wall = walls[ww];
        int plane = wall.x;
        float xPlane = zoom*((plane-0.5*Sizes.x)/Sizes.z);
        float yPlane = zoom*((plane-0.5*Sizes.y)/Sizes.z);
        float zPlane = zoom*((plane-0.5*Sizes.z)/Sizes.z);
        int type = wall.z;
        if (type == 0)
            glColor4f(0.4, 0.4, 0.0,0.5);
        else
            glColor4f(.5,0.0,.5,0.5);
        if(wall.y==0)//x-normal
        {
            glVertex3f(xPlane,yTop,-zTop);
            glVertex3f(xPlane,yTop,zTop);
            glVertex3f(xPlane,-yTop,zTop);
            glVertex3f(xPlane,-yTop,-zTop);
        }
        if(wall.y==1)//y-normal
        {
            glVertex3f(xTop,yPlane,-zTop);
            glVertex3f(xTop,yPlane,zTop);
            glVertex3f(-yTop,yPlane,zTop);
            glVertex3f(-xTop,yPlane,-zTop);
        }
        if(wall.y==2)//z-normal
        {
            glVertex3f(xTop,-yTop,zPlane);
            glVertex3f(xTop,yTop,zPlane);
            glVertex3f(-xTop,yTop,zPlane);
            glVertex3f(-xTop,-yTop,zPlane);
        }
    }
    glEnd();
    glDisable (GL_BLEND);
}
void OGLWidget::paintGL()
{
    glLoadIdentity();
        glTranslatef(0.0, 0.0, -10.0);
        glRotatef(xRot / 1.0, 1.0, 0.0, 0.0);
        glRotatef(yRot / 1.0, 0.0, 1.0, 0.0);
        glRotatef(zRot / 1.0, 0.0, 0.0, 1.0);
    draw();
    drawSpheres();
    drawWalls();
}

void OGLWidget::resizeGL(int w, int h)
{
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45, (float)w/h, 0.01, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(5,0,5,0,0,0,0,0,1);
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
