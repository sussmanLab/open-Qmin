#include "multirankSimulation.h"
/*! \file multrankSimulationBoundaries.cpp */

void multirankSimulation::finalizeObjects()
    {
    /* this section of code now handled in the base "createBoundaryObject() function
    {
    auto Conf = mConfiguration.lock();
    ArrayHandle<int> type(Conf->returnTypes());
    for(int ii = 0; ii < Conf->getNumberOfParticles();++ii)
        {
        int3 pos = Conf->indexToPosition(ii);
        int idx = Conf->positionToIndex(pos);
        if(type.data[idx] != 0)
            continue;
        for (int xx = -1; xx <=1; ++xx)
            for (int yy = -1; yy <=1; ++yy)
                for (int zz = -1; zz <=1; ++zz)
                    {
                    int3 otherpos = pos;
                    otherpos.x += xx;
                    otherpos.y += yy;
                    otherpos.z += zz;
                    int otheridx = Conf->positionToIndex(otherpos);
                    if (type.data[otheridx] > 0)
                        {
                        type.data[idx] = -1;
                        }
                    };
        }
    }
    */
    communicateHaloSitesRoutine();

    //let updaters know number of non-object sites
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        NActive=upd->getNTotal();
        };

    cout << " objects finalized" << endl;
    }

void multirankSimulation::createMultirankBoundaryObject(vector<int3> &latticeSites, vector<dVec> &qTensors, boundaryType _type, scalar Param1, scalar Param2)
    {
    auto Conf = mConfiguration.lock();
    ArrayHandle<dVec> pos(Conf->returnPositions());
    int3 globalLatticeSize;//the maximum size of the combined simulation
    int3 latticeMin;//where this rank sites in that lattice (min)
    int3 latticeMax;//...and (max)
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    latticeMin.x = rankParity.x*Conf->latticeSites.x;
    latticeMin.y = rankParity.y*Conf->latticeSites.y;
    latticeMin.z = rankParity.z*Conf->latticeSites.z;
    latticeMax.x = (1+rankParity.x)*Conf->latticeSites.x;
    latticeMax.y = (1+rankParity.y)*Conf->latticeSites.y;
    latticeMax.z = (1+rankParity.z)*Conf->latticeSites.z;
    vector<int> latticeSitesToEmploy;
    for (int ii = 0; ii < latticeSites.size(); ++ii)
        {
        //make sure the site is within the simulation box
        int3 currentSite = wrap(latticeSites[ii],globalLatticeSize);;
        //check if it is within control of this rank
        if(currentSite >=latticeMin && currentSite < latticeMax)
            {
            int3 currentLatticePos = currentSite - latticeMin;
            int currentLatticeSite = Conf->positionToIndex(currentLatticePos);
            latticeSitesToEmploy.push_back(currentLatticeSite);
            pos.data[currentLatticeSite] = qTensors[ii];

            };
        };
    Conf->createBoundaryObject(latticeSitesToEmploy,_type,Param1,Param2);
    };

/*!
/param xyz int that is 0, 1, or 2 for wall normal x, y, and z, respectively
*/
void multirankSimulation::createWall(int xyz, int plane, boundaryObject &bObj)
    {
    auto Conf = mConfiguration.lock();
    if (xyz <0 || xyz >2)
        UNWRITTENCODE("NOT AN OPTION FOR A FLAT SIMPLE WALL");
    dVec Qtensor(0.0);
    scalar s0 = bObj.P2;
    switch(bObj.boundary)
        {
        case boundaryType::homeotropic:
            {
            if(xyz ==0)
                {Qtensor[0] = s0; Qtensor[3] = -0.5*s0;}
            else if (xyz==1)
                {Qtensor[0] = -0.5*s0; Qtensor[3] = s0;}
            else
                {Qtensor[0] = -0.5*s0; Qtensor[3] = -0.5*s0;}
            break;
            }
        case boundaryType::degeneratePlanar:
            {
            Qtensor[0]=0.0; Qtensor[1] = 0.0; Qtensor[2] = 0.0;
            Qtensor[xyz]=1.0;
            break;
            }
        default:
            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
        };

        int3 globalLatticeSize;//the maximum size of the combined simulation
        globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
        globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
        globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;

        vector<int3> boundSites;
        vector<dVec> qTensors;
        int currentSite;
        int size1,size2;
        if(xyz ==0)
            {size1=globalLatticeSize.y;size2=globalLatticeSize.z;}
        else if (xyz==1)
            {size1=globalLatticeSize.x;size2=globalLatticeSize.z;}
        else
            {size1=globalLatticeSize.x;size2=globalLatticeSize.y;}
        for (int xx = 0; xx < size1; ++xx)
            for (int yy = 0; yy < size2; ++yy)
                {
                int3 sitePos;
                if(xyz ==0)
                    {
                    sitePos.x=plane;sitePos.y=xx;sitePos.z=yy;
                    }
                else if (xyz==1)
                    {
                    sitePos.x=xx;sitePos.y=plane;sitePos.z=yy;
                    }
                else
                    {
                    sitePos.x=xx;sitePos.y=yy;sitePos.z=plane;
                    }

                boundSites.push_back(sitePos);
                qTensors.push_back(Qtensor);
                }
    printf("wall with %lu sites created\n",boundSites.size());
    createMultirankBoundaryObject(boundSites,qTensors,bObj.boundary,bObj.P1,bObj.P2);
    };

void multirankSimulation::createSphericalColloid(scalar3 center, scalar radius, boundaryObject &bObj)
    {
    dVec Qtensor(0.);
    scalar S0 = bObj.P2;
    vector<int3> boundSites;
    vector<dVec> qTensors;
    for (int xx = ceil(center.x-radius); xx < floor(center.x+radius); ++xx)
        for (int yy = ceil(center.y-radius); yy < floor(center.y+radius); ++yy)
            for (int zz = ceil(center.z-radius); zz < floor(center.z+radius); ++zz)
            {
            scalar3 disp;
            disp.x = xx - center.x;
            disp.y = yy - center.y;
            disp.z = zz - center.z;

            if((disp.x*disp.x+disp.y*disp.y+disp.z*disp.z) < radius*radius)
                {
                int3 sitePos;
                sitePos.x = xx;
                sitePos.y = yy;
                sitePos.z = zz;
                boundSites.push_back(sitePos);
                switch(bObj.boundary)
                    {
                    case boundaryType::homeotropic:
                        {
                        qTensorFromDirector(disp, S0, Qtensor);
                        break;
                        }
                    case boundaryType::degeneratePlanar:
                        {
                        scalar nrm = norm(disp);
                        disp.x /=nrm;
                        disp.y /=nrm;
                        disp.z /=nrm;
                        Qtensor[0]=disp.x; Qtensor[1] = disp.y; Qtensor[2] = disp.z;
                        break;
                        }
                    default:
                        UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
                    };
                qTensors.push_back(Qtensor);
                };
            }
    printf("sphere with %lu sites created\n",boundSites.size());
    createMultirankBoundaryObject(boundSites,qTensors,bObj.boundary,bObj.P1,bObj.P2);
    };

void multirankSimulation::createSphericalCavity(scalar3 center, scalar radius, boundaryObject &bObj)
    {
    dVec Qtensor(0.);
    scalar S0 = bObj.P2;
    int3 globalLatticeSize;//the maximum size of the combined simulation
    auto Conf = mConfiguration.lock();
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    vector<int3> boundSites;
    vector<dVec> qTensors;
    scalar radiusSquared = radius*radius;
    for (int xx = 0; xx < globalLatticeSize.x; ++xx)
        for (int yy = 0; yy < globalLatticeSize.y; ++yy)
            for (int zz = 0; zz < globalLatticeSize.z; ++zz)
            {
            scalar3 disp;
            disp.x = xx - center.x;
            disp.y = yy - center.y;
            disp.z = zz - center.z;

            if((disp.x*disp.x+disp.y*disp.y+disp.z*disp.z) > radiusSquared)
                {
                int3 sitePos;
                sitePos.x = xx;
                sitePos.y = yy;
                sitePos.z = zz;
                boundSites.push_back(sitePos);
                switch(bObj.boundary)
                    {
                    case boundaryType::homeotropic:
                        {
                        qTensorFromDirector(disp, S0, Qtensor);
                        break;
                        }
                    case boundaryType::degeneratePlanar:
                        {
                        scalar nrm = norm(disp);
                        disp.x /=nrm;
                        disp.y /=nrm;
                        disp.z /=nrm;
                        Qtensor[0]=disp.x; Qtensor[1] = disp.y; Qtensor[2] = disp.z;
                        break;
                        }
                    default:
                        UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
                    };
                qTensors.push_back(Qtensor);
                };
            }
    printf("sphercal cavity with %lu sites created\n",boundSites.size());
    createMultirankBoundaryObject(boundSites,qTensors,bObj.boundary,bObj.P1,bObj.P2);
    };

/*!
define a cylinder by it's two endpoints and it's prependicular radius. 
\param colloidOrCapillary if true, points inside the radius are the object (i.e., a colloid), else points outside are the object (i.e., a surrounding capillary)
*/
void multirankSimulation::createCylindricalObject(scalar3 cylinderStart, scalar3 cylinderEnd, scalar radius, bool colloidOrCapillary, boundaryObject &bObj)
    {
    dVec Qtensor(0.);
    scalar S0 = bObj.P2;
    int3 globalLatticeSize;//the maximum size of the combined simulation
    auto Conf = mConfiguration.lock();
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    vector<int3> boundSites;
    vector<dVec> qTensors;
    scalar radiusSquared = radius*radius;
    for (int xx = 0; xx < globalLatticeSize.x; ++xx)
        for (int yy = 0; yy < globalLatticeSize.y; ++yy)
            for (int zz = 0; zz < globalLatticeSize.z; ++zz)
                {
                //p is the lattice point, disp will get the direction from lattice point to cylinder center
                scalar3 p,disp;
                p.x=xx;p.y=yy;p.z=zz;
                scalar dist = truncatedPointSegmentDistance(p,cylinderStart,cylinderEnd,disp);
                if(dist <0) //the shortest distance is past one of the endpoints
                    continue;
                if((dist < radius && colloidOrCapillary)
                        || (dist > radius && !colloidOrCapillary))
                    {
                    int3 sitePos;
                    sitePos.x = xx;
                    sitePos.y = yy;
                    sitePos.z = zz;
                    boundSites.push_back(sitePos);
                    switch(bObj.boundary)
                        {
                        case boundaryType::homeotropic:
                            {
                            qTensorFromDirector(disp, S0, Qtensor);
                            break;
                            }
                        case boundaryType::degeneratePlanar:
                            {
                            scalar nrm = norm(disp);
                            disp.x /=nrm;
                            disp.y /=nrm;
                            disp.z /=nrm;
                            Qtensor[0]=disp.x; Qtensor[1] = disp.y; Qtensor[2] = disp.z;
                            break;
                            }
                        default:
                            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
                        };
                    qTensors.push_back(Qtensor);
                    }
                }
    if(colloidOrCapillary)
        printf("cylindrical colloid with %lu sites created\n",boundSites.size());
    else
        printf("cylindrical capillary with %lu sites created\n",boundSites.size());
    createMultirankBoundaryObject(boundSites,qTensors,bObj.boundary,bObj.P1,bObj.P2);
    };

void multirankSimulation::createSpherocylinder(scalar3 cylinderStart, scalar3 cylinderEnd, scalar radius, boundaryObject &bObj)
    {
    dVec Qtensor(0.);
    scalar S0 = bObj.P2;
    int3 globalLatticeSize;//the maximum size of the combined simulation
    auto Conf = mConfiguration.lock();
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    vector<int3> boundSites;
    vector<dVec> qTensors;
    scalar radiusSquared = radius*radius;
    for (int xx = 0; xx < globalLatticeSize.x; ++xx)
        for (int yy = 0; yy < globalLatticeSize.y; ++yy)
            for (int zz = 0; zz < globalLatticeSize.z; ++zz)
                {
                //p is the lattice point, disp will get the direction from lattice point to cylinder center
                scalar3 p,disp;
                p.x=xx;p.y=yy;p.z=zz;
                scalar dist = pointSegmentDistance(p,cylinderStart,cylinderEnd,disp);
                if(dist <0) //the shortest distance is past one of the endpoints
                    continue;
                if(dist < radius)
                    {
                    int3 sitePos;
                    sitePos.x = xx;
                    sitePos.y = yy;
                    sitePos.z = zz;
                    boundSites.push_back(sitePos);
                    switch(bObj.boundary)
                        {
                        case boundaryType::homeotropic:
                            {
                            qTensorFromDirector(disp, S0, Qtensor);
                            break;
                            }
                        case boundaryType::degeneratePlanar:
                            {
                            scalar nrm = norm(disp);
                            disp.x /=nrm;
                            disp.y /=nrm;
                            disp.z /=nrm;
                            Qtensor[0]=disp.x; Qtensor[1] = disp.y; Qtensor[2] = disp.z;
                            break;
                            }
                        default:
                            UNWRITTENCODE("non-defined boundary type is attempting to create a boundary");
                        };
                    qTensors.push_back(Qtensor);
                    }
                }
    printf("spherocylindrical colloid with %lu sites created\n",boundSites.size());
    createMultirankBoundaryObject(boundSites,qTensors,bObj.boundary,bObj.P1,bObj.P2);
    };

/*!
for lattice sites within range of center, create a z-aligned dipolar field with opening angle
ThetaD about an object of radius "radius"
Functional form from Lubensky, Pettey, Currier, and Stark. PRE 57, 610, 1998
 */ 
void multirankSimulation::setDipolarField(scalar3 center, scalar ThetaD, scalar radius, scalar range, scalar S0)
    {
    auto Conf = mConfiguration.lock();
    ArrayHandle<dVec> pos(Conf->returnPositions());
    ArrayHandle<int> types(Conf->returnTypes());
    int3 globalLatticeSize;//the maximum size of the combined simulation
    int3 latticeMin;//where this rank sites in that lattice (min)
    int3 latticeMax;//...and (max)
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    latticeMin.x = rankParity.x*Conf->latticeSites.x;
    latticeMin.y = rankParity.y*Conf->latticeSites.y;
    latticeMin.z = rankParity.z*Conf->latticeSites.z;
    latticeMax.x = (1+rankParity.x)*Conf->latticeSites.x;
    latticeMax.y = (1+rankParity.y)*Conf->latticeSites.y;
    latticeMax.z = (1+rankParity.z)*Conf->latticeSites.z;

    scalar k = 0.32;
    scalar rd = 1.22;
    scalar td = ThetaD;
    for (int ii = 0; ii < Conf->getNumberOfParticles(); ++ii)
        {
        //skip sites that are part of boundaries
        if(types.data[ii] > 0)
            continue;
        int3 site = Conf->indexToPosition(ii);
        scalar3 globalSitePosition;
        globalSitePosition.x = latticeMin.x+site.x;
        globalSitePosition.y = latticeMin.y+site.y;
        globalSitePosition.z = latticeMin.z+site.z;
        scalar3 relativePosition;
        relativePosition.x = globalSitePosition.x - center.x;
        relativePosition.y = globalSitePosition.y - center.y;
        relativePosition.z = globalSitePosition.z - center.z;
        scalar r ,t, p, Theta;
        scalar3 director;
        r= norm(relativePosition);
        if(r < range && r >radius)
            {
            t = acos(relativePosition.z/r);
            p = atan2(relativePosition.y,relativePosition.x);
            r = r/radius; //normalize properly

            scalar ri = 1./r;
            scalar ri2 = ri/r;
            scalar ri3= ri2/r;
            scalar rdi = 1./rd;
            scalar rdi2 = rdi/rd;
            scalar rdi3= rdi2/rd;
            Theta = 2*t;
            Theta -= 0.5*( atan2(r*Sin(t) - rd*Sin(td),r*Cos(t) - rd*Cos(td))
                          +atan2(r*Sin(t) + rd*Sin(td),r*Cos(t) - rd*Cos(td))
                          +atan2(r*rd*Sin(t) - Sin(td),r*rd*Cos(t) - Cos(td))
                          +atan2(r*rd*Sin(t) + Sin(td),r*rd*Cos(t) - Cos(td)) );
            Theta += exp(-k*ri3)*
                            ( (rd+rdi)*(ri-ri2)*Cos(td)*Sin(t)
                             +(rd*rd+rdi2)*(ri2-ri3)*(2.*Cos(td)*Cos(td)-1.)*Sin(t)*Cos(t)
                             +1./3.*(rd*rd*rd+rdi3)*(ri3-ri3/r)*Cos(td)*(4.*Cos(td)*Cos(td)-3.)*Sin(t)*(4.*Cos(t)*Cos(t) - 1.) );
            director.x = Sin(Theta)*Cos(p);
            director.y = Sin(Theta)*Sin(p);
            director.z = Cos(Theta);
            if(Theta == 0 && p ==0)
                {
                director.x=0;director.y=0;director.z=0;
                }
            qTensorFromDirector(director, S0, pos.data[ii]);
            }

        }
    };

/*!
for lattice sites within range of center, create a dipolar field in the direction specified
about an object of radius "radius"
 */ 
void multirankSimulation::setDipolarField(scalar3 center, scalar3 direction, scalar radius, scalar range, scalar S0)
    {
    auto Conf = mConfiguration.lock();
    ArrayHandle<dVec> pos(Conf->returnPositions());
    ArrayHandle<int> types(Conf->returnTypes());
    int3 globalLatticeSize;//the maximum size of the combined simulation
    int3 latticeMin;//where this rank sites in that lattice (min)
    int3 latticeMax;//...and (max)
    globalLatticeSize.x = rankTopology.x*Conf->latticeSites.x;
    globalLatticeSize.y = rankTopology.y*Conf->latticeSites.y;
    globalLatticeSize.z = rankTopology.z*Conf->latticeSites.z;
    latticeMin.x = rankParity.x*Conf->latticeSites.x;
    latticeMin.y = rankParity.y*Conf->latticeSites.y;
    latticeMin.z = rankParity.z*Conf->latticeSites.z;
    latticeMax.x = (1+rankParity.x)*Conf->latticeSites.x;
    latticeMax.y = (1+rankParity.y)*Conf->latticeSites.y;
    latticeMax.z = (1+rankParity.z)*Conf->latticeSites.z;

    scalar3 i = direction*(1.0/norm(direction));
    scalar P = 2.08;
    for (int ii = 0; ii < Conf->getNumberOfParticles(); ++ii)
        {
        //skip sites that are part of boundaries
        if(types.data[ii] > 0 || types.data[ii] < -1)
            continue;
        int3 site = Conf->indexToPosition(ii);
        scalar3 globalSitePosition;
        globalSitePosition.x = latticeMin.x+site.x;
        globalSitePosition.y = latticeMin.y+site.y;
        globalSitePosition.z = latticeMin.z+site.z;
        scalar3 relativePosition;
        relativePosition.x = globalSitePosition.x - center.x;
        relativePosition.y = globalSitePosition.y - center.y;
        relativePosition.z = globalSitePosition.z - center.z;
        scalar3 director;
        scalar r= norm(relativePosition);
        if(r < range && r >radius)
            {
            scalar ri3 = 1.0/(r*r*r);
            director = i + (P)*radius*radius*ri3*relativePosition;
            director = director*(1.0/norm(director));
            qTensorFromDirector(director, S0, pos.data[ii]);
            }
        }
    };

