#include "multirankSimulation.h"
/*! \file multirankSimulation.cpp */

void multirankSimulation::communicateHaloSitesRoutine()
    {
    transfersUpToDate = false;
    //first, prepare the send buffers
    {
    auto Conf = mConfiguration.lock();
    if(!useGPU)
        {
        for (int ii = 0; ii < communicationDirections.size();++ii)
            {
            int directionType = communicationDirections[ii].x;
            Conf->prepareSendingBuffer(directionType);
            }
        }
    else
        Conf->prepareSendingBuffer();//a single call copies the entire buffer
    }//end buffer send prep
    {
    auto Conf = mConfiguration.lock();
    //MPI Routines
    for (int ii = 0; ii < communicationDirections.size();++ii)
        {
        int directionType = communicationDirections[ii].x;
        int2 startStop = Conf->transferStartStopIndexes[directionType];
        int receiveStart = Conf->transferStartStopIndexes[communicationDirections[ii].y].x;
        access_location::Enum dataLocation = useGPU ? access_location::device : access_location::host;
        dataLocation = access_location::host;//explicit stage through host
        int targetRank = communicationTargets[ii];
        int messageTag1 = 2*directionType;
        int messageTag2 = messageTag1+1;
        int messageSize = startStop.y-startStop.x+1;
        int dMessageSize = DIMENSION*messageSize;
        if(communicationDirectionParity[ii]) //send and receive
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive,dataLocation,access_mode::overwrite);
            ArrayHandle<scalar> dBufS(Conf->doubleTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<scalar> dBufR(Conf->doubleTransferBufferReceive,dataLocation,access_mode::overwrite);
            MPI_Isend(&iBufS.data[startStop.x],messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD,&mpiRequests[4*ii+0]);
            MPI_Irecv(&iBufR.data[receiveStart],messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiRequests[4*ii+1]);
            MPI_Isend(&dBufS.data[DIMENSION*startStop.x],dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD,&mpiRequests[4*ii+2]);
            MPI_Irecv(&dBufR.data[DIMENSION*receiveStart],dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiRequests[4*ii+3]);
            }
        else
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive,dataLocation,access_mode::overwrite);
            ArrayHandle<scalar> dBufS(Conf->doubleTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<scalar> dBufR(Conf->doubleTransferBufferReceive,dataLocation,access_mode::overwrite);
            MPI_Irecv(&iBufR.data[receiveStart],messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiRequests[4*ii+0]);
            MPI_Isend(&iBufS.data[startStop.x],messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD,&mpiRequests[4*ii+1]);
            MPI_Irecv(&dBufR.data[DIMENSION*receiveStart],dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiRequests[4*ii+2]);
            MPI_Isend(&dBufS.data[DIMENSION*startStop.x],dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD,&mpiRequests[4*ii+3]);
            }
        }
    }//end MPI routines
    }

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


void multirankSimulation::synchronizeAndTransferBuffers()
    {
    for(int ii = 0; ii < mpiRequests.size();++ii)
        MPI_Wait(&mpiRequests[ii],&mpiStatuses[ii]);
    //read readReceivingBuffer
    auto Conf = mConfiguration.lock();
    if(!useGPU)
        {
        for (int ii = 0; ii < communicationDirections.size();++ii)
            {
            int directionType = communicationDirections[ii].y;
            Conf->readReceivingBuffer(directionType);
            }
        }
    else
        Conf->readReceivingBuffer();//a single call reads and copies the entire buffer
    transfersUpToDate = false;
    }

/*!
Calls the configuration to displace the degrees of freedom, and communicates halo sites according
to the rankTopology and boolean settings
*/
void multirankSimulation::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
        {
    auto Conf = mConfiguration.lock();
    Conf->moveParticles(displacements,scale);
        }
    p1.start();
    communicateHaloSitesRoutine();
    p1.end();
    };

void multirankSimulation::setRankTopology(int x, int y, int z)
    {
    rankTopology.x=x;
    rankTopology.y=y;
    rankTopology.z=z;
    int Px, Py, Pz;
    parityTest = Index3D(rankTopology);
    rankParity = parityTest.inverseIndex(myRank);

    }

void multirankSimulation::determineCommunicationPattern( bool _edges, bool _corners)
    {
    edges = _edges;
    corners = _corners;
    bool sendReceiveParity;
    int3 nodeTarget;
    int2 sendReceive;
    int targetRank;
    //faces
    if(rankTopology.x > 1)
        {
        sendReceive.x = 0; sendReceive.y = 1;
        sendReceiveParity = (rankParity.x%2==0) ? true : false;

        nodeTarget = rankParity; nodeTarget.x -= 1;
        if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 1; sendReceive.y = 0;
        nodeTarget = rankParity; nodeTarget.x += 1;
        if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);
        }

    if(rankTopology.y > 1)
        {
        sendReceive.x = 2; sendReceive.y = 3;
        sendReceiveParity = (rankParity.y%2==0) ? true : false;
        nodeTarget = rankParity; nodeTarget.y -= 1;
        if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 3; sendReceive.y = 2;
        nodeTarget = rankParity; nodeTarget.y += 1;
        if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);
        }

    if(rankTopology.z> 1)
        {
            sendReceive.x = 4; sendReceive.y = 5;
            sendReceiveParity = (rankParity.z%2==0) ? true : false;
            nodeTarget = rankParity; nodeTarget.z -= 1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 5; sendReceive.y = 4;
            nodeTarget = rankParity; nodeTarget.z += 1;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);
        }
    if(edges)
        {
        if(rankTopology.x > 1 && rankTopology.y > 1)
            {
            sendReceive.x = 6; sendReceive.y = 11; //x-y- to x+y+
            sendReceiveParity = (rankParity.x%2==0) ? true : false;
            nodeTarget = rankParity; nodeTarget.y -= 1; nodeTarget.x -= 1;
            if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
            if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 11; sendReceive.y = 6;//x+y+ to x-y-
            nodeTarget = rankParity; nodeTarget.y += 1; nodeTarget.x += 1;
            if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
            if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 7; sendReceive.y = 10;//x-y+ to x+y-
            nodeTarget = rankParity; nodeTarget.y += 1; nodeTarget.x -= 1;
            if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
            if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 10; sendReceive.y = 7;//x+y- to x-y+
            nodeTarget = rankParity; nodeTarget.y -= 1; nodeTarget.x += 1;
            if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
            if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);
            }
        if(rankTopology.x > 1 && rankTopology.z > 1)
            {
            sendReceive.x = 8; sendReceive.y = 13; //x-z- to x+z+
            sendReceiveParity = (rankParity.x%2==0) ? true : false;
            nodeTarget = rankParity; nodeTarget.z -= 1; nodeTarget.x -= 1;
            if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 13; sendReceive.y = 8;//x+z+ to x-z-
            nodeTarget = rankParity; nodeTarget.z += 1; nodeTarget.x += 1;
            if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 9; sendReceive.y = 12;//x-z+ to x+z-
            nodeTarget = rankParity; nodeTarget.z += 1; nodeTarget.x -= 1;
            if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 12; sendReceive.y = 9;//x+z- to x-z+
            nodeTarget = rankParity; nodeTarget.z -= 1; nodeTarget.x += 1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);
            }
        if(rankTopology.y > 1 && rankTopology.z > 1)
            {
            sendReceive.x = 14; sendReceive.y = 17; //y-z- to y+z+
            sendReceiveParity = (rankParity.y%2==0) ? true : false;
            nodeTarget = rankParity; nodeTarget.z -= 1; nodeTarget.y -= 1;
            if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 17; sendReceive.y = 14;//y+z+ to y-z-
            nodeTarget = rankParity; nodeTarget.z += 1; nodeTarget.y += 1;
            if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 15; sendReceive.y = 16;//y-z+ to y+z-
            nodeTarget = rankParity; nodeTarget.z += 1; nodeTarget.y -= 1;
            if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);

            sendReceive.x = 16; sendReceive.y = 15;//y+z- to y-z+
            nodeTarget = rankParity; nodeTarget.z -= 1; nodeTarget.y += 1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
            targetRank = parityTest(nodeTarget);
            communicationDirections.push_back(sendReceive);
            communicationDirectionParity.push_back(sendReceiveParity);
            communicationTargets.push_back(targetRank);
            }
        }

    if(corners && rankTopology.x > 1 && rankTopology.y > 1 && rankTopology.z > 1)
        {
        sendReceiveParity = (rankParity.x%2==0) ? true : false;
        sendReceive.x = 18; sendReceive.y = 25; //--- to +++
        nodeTarget = rankParity; nodeTarget.x -= 1; nodeTarget.y -= 1;nodeTarget.z -= 1;
        if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
        if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
        if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 25; sendReceive.y = 18; // +++ to ---
        nodeTarget = rankParity; nodeTarget.x += 1; nodeTarget.y += 1;nodeTarget.z += 1;
        if(nodeTarget.x  == parityTest.sizes.x) nodeTarget.x = 0;
        if(nodeTarget.y  == parityTest.sizes.y) nodeTarget.y = 0;
        if(nodeTarget.z  == parityTest.sizes.z) nodeTarget.z = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 19; sendReceive.y = 24; //--+ to ++-
        nodeTarget = rankParity; nodeTarget.x -= 1; nodeTarget.y -= 1;nodeTarget.z += 1;
        if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
        if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
        if(nodeTarget.z  == parityTest.sizes.z) nodeTarget.z = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 24; sendReceive.y = 19; // ++- to --+
        nodeTarget = rankParity; nodeTarget.x += 1; nodeTarget.y += 1;nodeTarget.z -= 1;
        if(nodeTarget.x  == parityTest.sizes.x) nodeTarget.x = 0;
        if(nodeTarget.y  == parityTest.sizes.y) nodeTarget.y = 0;
        if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 20; sendReceive.y = 23; //-+- to +-+
        nodeTarget = rankParity; nodeTarget.x -= 1; nodeTarget.y += 1;nodeTarget.z -= 1;
        if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
        if(nodeTarget.y  == parityTest.sizes.y) nodeTarget.y = 0;
        if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 23; sendReceive.y = 20; // +-+ to -+-
        nodeTarget = rankParity; nodeTarget.x += 1; nodeTarget.y -= 1;nodeTarget.z += 1;
        if(nodeTarget.x  == parityTest.sizes.x) nodeTarget.x = 0;
        if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
        if(nodeTarget.z  == parityTest.sizes.z) nodeTarget.y = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 21; sendReceive.y = 22; //-++ to +--
        nodeTarget = rankParity; nodeTarget.x -= 1; nodeTarget.y += 1;nodeTarget.z += 1;
        if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
        if(nodeTarget.y  == parityTest.sizes.y) nodeTarget.y = 0;
        if(nodeTarget.z  == parityTest.sizes.z) nodeTarget.z = 0;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);

        sendReceive.x = 22; sendReceive.y = 21; //+-- to -++
        nodeTarget = rankParity; nodeTarget.x += 1; nodeTarget.y -= 1;nodeTarget.z -= 1;
        if(nodeTarget.x  == parityTest.sizes.x) nodeTarget.x = 0;
        if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
        if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
        targetRank = parityTest(nodeTarget);
        communicationDirections.push_back(sendReceive);
        communicationDirectionParity.push_back(sendReceiveParity);
        communicationTargets.push_back(targetRank);
        };
    mpiRequests.resize(4*communicationDirections.size());
    mpiStatuses.resize(4*communicationDirections.size());
    }

/*!
Add a pointer to the list of updaters, and give that updater a reference to the
model...
*/
void multirankSimulation::addUpdater(UpdaterPtr _upd, MConfigPtr _config)
    {
    _upd->setModel(_config);
    _upd->setSimulation(getPointer());
    updaters.push_back(_upd);
    };

/*!
Add a pointer to the list of force computers, and give that FC a reference to the
model...
*/
void multirankSimulation::addForce(ForcePtr _force, MConfigPtr _config)
    {
    _force->setModel(_config);
    forceComputers.push_back(_force);
    };
/*!
Set a pointer to the configuration
*/
void multirankSimulation::setConfiguration(MConfigPtr _config)
    {
    mConfiguration = _config;
    Box = _config->Box;
    communicateHaloSitesRoutine();
    };

/*!
Calls all force computers, and evaluate the self force calculation if the model demands it
*/
void multirankSimulation::computeForces()
    {
    auto Conf = mConfiguration.lock();
    if(Conf->selfForceCompute)
        Conf->computeForces(true);
    for (unsigned int f = 0; f < forceComputers.size(); ++f)
        {
        auto frc = forceComputers[f].lock();
        bool zeroForces = (f==0 && !Conf->selfForceCompute);
        //compute bulk sites... since they are in the bulk they can be done while MPI buffers are transfered for CPU-mpi jobs. GPU is not like this, though
        if(!useGPU && forceComputers.size() == 1)
            {            
            frc->computeForces(Conf->returnForces(),zeroForces,0);
            //wait for communication...
            if(!transfersUpToDate)
                synchronizeAndTransferBuffers();
            //compute boundary sites
            frc->computeForces(Conf->returnForces(),false,1);
            }
        else
            {
            frc->computeForces(Conf->returnForces(),zeroForces);
            }
        };
    if(useGPU)
        synchronizeAndTransferBuffers();

    Conf->forcesComputed = true;
    };

scalar multirankSimulation::computePotentialEnergy(bool verbose)
    {
    scalar PE = 0.0;
    for (int f = 0; f < forceComputers.size(); ++f)
        {
        auto frc = forceComputers[f].lock();
        PE += frc->computeEnergy(verbose);
        };
    return PE;
    };

scalar multirankSimulation::computeKineticEnergy(bool verbose)
    {
    auto Conf = mConfiguration.lock();
    return Conf->computeKineticEnergy(verbose);
    }

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void multirankSimulation::setIntegrationTimestep(scalar dt)
    {
    integrationTimestep = dt;
    //auto cellConf = cellConfiguration.lock();
    //cellConf->setDeltaT(dt);
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->setDeltaT(dt);
        };
    };

/*!
\post the cell configuration and e.o.m. timestep is set to the input value
*/
void multirankSimulation::setCPUOperation(bool setcpu)
    {
    auto Conf = mConfiguration.lock();
    useGPU = !setcpu;
    Conf->setGPU(useGPU);

    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->setGPU(useGPU);
        };
    for (int f = 0; f < forceComputers.size(); ++f)
        {
        auto frc = forceComputers[f].lock();
        frc->setGPU(useGPU);
        };
    };

/*!
\pre the updaters already know if the GPU will be used
\post the updaters are set to be reproducible if the boolean is true, otherwise the RNG is initialized
*/
void multirankSimulation::setReproducible(bool reproducible)
    {
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->setReproducible(reproducible);
        };
    };

void multirankSimulation::performTimestep()
    {
    integerTimestep += 1;
    Time += integrationTimestep;

    //perform any updates, one of which should probably be an EOM
    for (int u = 0; u < updaters.size(); ++u)
        {
        auto upd = updaters[u].lock();
        upd->Update(integerTimestep);
        };
    };

void multirankSimulation::saveState(string fname)
    {
    auto Conf = mConfiguration.lock();
    char fn[256];
    sprintf(fn,"%s_x%iy%iz%i.txt",fname.c_str(),rankParity.x,rankParity.y,rankParity.z);

    printf("saving state...\n");

    int xOffset = rankParity.x*Conf->latticeSites.x;
    int yOffset = rankParity.y*Conf->latticeSites.y;
    int zOffset = rankParity.z*Conf->latticeSites.z;

    Conf->getAverageEigenvalues();
    ArrayHandle<dVec> pp(Conf->returnPositions());
    ArrayHandle<int> tt(Conf->returnTypes());
    ofstream myfile;
    myfile.open(fn);
    for (int ii = 0; ii < Conf->getNumberOfParticles(); ++ii)
    //for (int ii = 0; ii < Conf->totalSites; ++ii)
        {
        int3 pos = Conf->indexToPosition(ii);
        int idx = Conf->positionToIndex(pos);
        myfile << pos.x+xOffset <<"\t"<<pos.y+yOffset<<"\t"<<pos.z+zOffset;
        for (int dd = 0; dd <DIMENSION; ++dd)
            myfile <<"\t"<<pp.data[idx][dd];
        myfile << "\t"<<tt.data[idx]<<"\n";
        }

    myfile.close();

    };
