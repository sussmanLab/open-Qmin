/*
This is a file of convenience, allowing one to specify certain boundary conditions to be used by
the openQmin.cpp file without needing to edit that cpp file directly (although of course, since this
header is simply #included, one should feel free to disregard it and directly change the cpp file
as desired). Also, since this file is included in a specific place, it assumes that certain variables
(simulation size, etc.) are already defined

Below are commented out examples outlining how to input a custom boundary file, and also how to add some
of the predefined boundaries and colloids already implemented in the openQmin package.
*/


/* 
//To load a custom boundary file which completely specifies multiple objects, do the following:

string boundaryFileToLoad ("nameOfInputFile.txt");
sim->createBoundaryFromFile(boundaryFileToLoad,true);
//Note: that "true" is a switch controlling howmuch information is printed to the screen when the
//command is run...setting it to false just disables some cout statements
*/
    
   
/*
//To use some of the built-in colloid and boundary functions, we first need to specify types of
//anchoring conditions (which are called "boundaryObjects" in the class structure):

scalar targetNematicValue = S0;
scalar anchoringStrength1 = 5.0;
scalar anchoringStrength2 = 1.0;
boundaryObject homeotropicBoundary1(boundaryType::homeotropic,anchoringStrength1,targetNematicValue);
boundaryObject homeotropicBoundary2(boundaryType::homeotropic,anchoringStrength1,targetNematicValue*0.8);
boundaryObject pdgBoundary(boundaryType::degeneratePlanar,anchoringStrength2,targetNematicValue);

//We'll see below that these anchoring conditions can be shared by multiple objects, so you only need to to
//define as many different conditions as exist in the simulation (as opposed to defining a different condition
//for every distinct object)
*/
    

/*
//Assuming the anchoring conditions above have been set, let's set four spheres in a line, with positions
//and (equal) sizes given relative to the simulation size:

scalar3 pos1, pos2, pos3, pos4;
scalar radius = boxLx*0.05;
pos1.x = 0.2*boxLx; pos1.y = 0.5*boxLy; pos1.z = 0.5*boxLz;
pos2.x = 0.4*boxLx; pos2.y = 0.5*boxLy; pos2.z = 0.5*boxLz;
pos3.x = 0.6*boxLx; pos3.y = 0.5*boxLy; pos3.z = 0.5*boxLz;
pos4.x = 0.8*boxLx; pos4.y = 0.5*boxLy; pos4.z = 0.5*boxLz;

sim->createSphericalColloid(pos1,radius,homeotropicBoundary1);
sim->createSphericalColloid(pos2,radius,homeotropicBoundary1);
sim->createSphericalColloid(pos3,radius,pdgBoundary);
sim->createSphericalColloid(pos4,radius,homeotropicBoundary2);
*/

/*
\\To create a wall whose normal is in the x, y, or z-direction, use the following:

int xyz = 0; \\0 --> wall has normal along \hat{x}, 1 --> wall has normal along \hat{y}, and 2 --> wall has normal along \hat{z}
int latticePlane = 0; \\Pick any integer in between 0 and the maximum number of lattice sites in the direction desired (e.g., (boxLx-1)), inclusive.

sim->createWall(xyz,latticePlane,homeotropicBoundary2);
*/

/*
\\cylinders and cylinders with a spherical cap require a start and end point to the cylindrical segment:
scalar3 pathStart, pathEnd;
scalar radius;
pathStart.x = 0.25*boxLx; pathStart.y=0.25*boxLy; pathStart = 0.25*boxLz;
pathEnd.x = 0.75*boxLx; pathEnd.y=0.75*boxLy; pathEnd = 0.75*boxLz;
radius = 10.0; \\specified in units of lattice spacing, rather than relative to the simulation... for variety.

\\sim->createSpherocylinder(pathStart,pathEnd,radius,homeotropicBoundary1);

bool colloidOrCapillary = true;
\\sim->createCylindricalObject(pathStart,pathEnd,radius,colloidOrCapillary,homeotropicBoundary1);
\\Note that the "createCylindricalObject" function can be used to either make a colloidal object OR form a capillary (i.e. the "object" can either be inside a cylinder specified by a start, end, and radius, or it can be all the points outside that cylinder). For making a capillary, we recommend aligning the axis with the z-axis, and making the start and end the minimum and maximum size of the simulation box in that dimension.
*/

/*
\\Finally, for... reasons, the command to create a spherical cavity requires a separate function (unlike the switch in the capillary command above):
scalar3 center;
center.x = boxLx*0.5; center.y = boxLy*0.5; center.z = boxLz*0.5;
scalar radius = floor(boxLx*0.5) - 2;
sim->createSphericalCavity(center,radius,homeotropicBoundary1);
*/
