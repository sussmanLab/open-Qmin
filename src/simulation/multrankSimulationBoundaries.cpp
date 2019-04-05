#include "multirankSimulation.h"
/*! \file multrankSimulationBoundaries.cpp */

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
