/*
This is a file of convenience, allowing one to specify certain initial conditions to be used by openQmin.cpp without needing to edit that file directly
(although, of course, since this file is simply #included, one should feel free to disregard it and directly change the cpp file as desired
*/

/*What follows is a set up options for various common choices of initial conditions here's a guide of what the command line argument will do:

"-z 0 " -- the default behavior: for each lattice site pick a random director uniformly over the unit sphere and make a Q tensor corresponding to it (using magnitude of the order parameter S0)
"-z 1" pick a single random director and make a uniform nematic texture in that direction
"-z 2" choose a specific uniform texture to initialize (modify the function below to hard-code it)
"-z 3" define a function, f(x,y,z), that contols both the initial director and the initial s0 parameter to which to set the system to. (modify the function "localDirectors" below to whatever you want).
*/

if(initializationSwitch ==0)
    {
    if(verbose) printf("setting random configuration with S0 = %f\n",S0);
    Configuration->setNematicQTensorRandomly(noise,S0);
    };


//pick a random director and set all lattice sites to that direction (i.e., a uniform nematic texture with direction picked randomly)
if(initializationSwitch ==1)
    {
    if(verbose) printf("setting random uniform texture with S0 = %f\n",S0);
    //to pick the same random direction across all ranks, we need to have a rng with the same seed everywhere... this is a bit of a hack, and needs to be nudged to allow this to work non-reproducibly
    noiseSource multirankNoise(true);
    if(randomSeed == -1)
        multirankNoise.setReproducibleSeed(13371);
    else
        multirankNoise.setReproducibleSeed(randomSeed);

    Configuration->setRandomDirectors(multirankNoise,S0, true);
    };

//choose a specific director and S0 value, and set all Q tensors to it
if(initializationSwitch ==2)
    {
    if(verbose) printf("initializing with speicific uniform texture\n");
    scalar s0 = S0;
    scalar3 targetDirector;
    targetDirector.x = 1./2;
    targetDirector.y = 0;
    targetDirector.z = 1/2.;
    normalizeDirector(targetDirector);

    Configuration->setUniformDirectors(targetDirector,s0);
    };

/*finally: if you want to set an initial condition by specifying a Q-tensor corresponding to a director and an S0 value as a function of lattice (x,y,z) position, write your custom function in the body of the thing below (and, of course, make sure you recompile openQmin!)
Note openQmin's convention that the simulation domain starts at zero and goes up to a maximum size in each cartesian direction (as opposed to a convention where the center of the simulation domain is the origin)
*/

if(initializationSwitch == 3)
    {
    //we'll use c++11's lambda expressions to let us locally define a function that we'll pass to the multirankQtensorLatticeModel's function
    std::function<scalar4(scalar, scalar, scalar)> localDirectors=[](scalar x, scalar y, scalar z)
        {
        scalar4 result;
        scalar3 director;

        //write your function for the initial director here:
        director.x = x;
        director.y = x+y+ cos(z);
        director.z = 0;
        //write your function for the initial s0 value here:
        scalar s0 = 0.53+0.0000000001*(x+y+z);

        normalizeDirector(director);
        result.x = director.x;
        result.y = director.y;
        result.z = director.z;
        result.w = s0;
        return result;
        };
    if(verbose) printf("using custom function to define initial conditions\n");
    Configuration->setDirectorFromFunction(localDirectors);
    }

