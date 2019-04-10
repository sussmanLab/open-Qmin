#include "multirankSimulation.h"
/*! \file multrankSimulationBoundaries.cpp */

void multirankSimulation::finalizeObjects()
    {
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
                        type.data[idx] = -1;
                    };
        }
    }
    communicateHaloSitesRoutine();
    cout << " objects finalized" << endl;
    }

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
