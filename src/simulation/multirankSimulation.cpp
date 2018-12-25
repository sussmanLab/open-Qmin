#include "multirankSimulation.h"
/*! \file multirankSimulation.cpp */

void multirankSimulation::directionalHaloFaceCommunication(int direction, int directionalRankTopology)
    {
    auto Conf = mConfiguration.lock();
    access_location::Enum dataLocation = useGPU ? access_location::device : access_location::host;
    dataLocation = access_location::host;//explicit stage through host
    int dataSend, dataReceive, targetRank,messageTag1,messageTag2;
    int3 nodeTarget;
    messageTag1 = 4*direction;
    messageTag2 = messageTag1+1;
    switch(direction)
        {
        case 0:
            dataSend = 0;
            dataReceive = 1;
            nodeTarget = rankParity; nodeTarget.x -= 1;
            if(nodeTarget.x < 0) nodeTarget.x = parityTest.sizes.x-1;
            targetRank = parityTest(nodeTarget);
            break;
        case 1:
            dataSend = 1;
            dataReceive = 0;
            nodeTarget = rankParity; nodeTarget.x += 1;
            if(nodeTarget.x == parityTest.sizes.x) nodeTarget.x = 0;
            break;
        case 2:
            dataSend = 2;
            dataReceive = 3;
            nodeTarget = rankParity; nodeTarget.y -= 1;
            if(nodeTarget.y < 0) nodeTarget.y = parityTest.sizes.y-1;
            targetRank = parityTest(nodeTarget);
            break;
        case 3:
            dataSend = 3;
            dataReceive = 2;
            nodeTarget = rankParity; nodeTarget.y += 1;
            if(nodeTarget.y == parityTest.sizes.y) nodeTarget.y = 0;
            break;
        case 4:
            dataSend = 4;
            dataReceive = 5;
            nodeTarget = rankParity; nodeTarget.z -= 1;
            if(nodeTarget.z < 0) nodeTarget.z = parityTest.sizes.z-1;
            targetRank = parityTest(nodeTarget);
            break;
        case 5:
            dataSend = 5;
            dataReceive = 4;
            nodeTarget = rankParity; nodeTarget.z += 1;
            if(nodeTarget.z == parityTest.sizes.z) nodeTarget.z = 0;
            break;

        default : throw std::runtime_error("invalid directional send");
        };
    targetRank = parityTest(nodeTarget);

    p2.start();
    Conf->prepareSendData(dataSend);
    p2.end();
    int messageSize = Conf->transferElementNumber;
    int dMessageSize = DIMENSION*Conf->transferElementNumber;
    if(directionalRankTopology%2 == 0) //send and receive
        {
        ArrayHandle<int> iBufS(Conf->intTransferBufferSend,dataLocation,access_mode::read);
        ArrayHandle<int> iBufR(Conf->intTransferBufferReceive,dataLocation,access_mode::overwrite);
        ArrayHandle<scalar> dBufS(Conf->doubleTransferBufferSend,dataLocation,access_mode::read);
        ArrayHandle<scalar> dBufR(Conf->doubleTransferBufferReceive,dataLocation,access_mode::overwrite);
        MPI_Send(iBufS.data,messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD);
        MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiStatus);
        MPI_Send(dBufS.data,dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD);
        MPI_Recv(dBufR.data,dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiStatus);
        }
    else                    //receive and send
        {
        p2.start();
        ArrayHandle<int> iBufS(Conf->intTransferBufferSend,dataLocation,access_mode::read);
        ArrayHandle<int> iBufR(Conf->intTransferBufferReceive,dataLocation,access_mode::overwrite);
        ArrayHandle<scalar> dBufS(Conf->doubleTransferBufferSend,dataLocation,access_mode::read);
        ArrayHandle<scalar> dBufR(Conf->doubleTransferBufferReceive,dataLocation,access_mode::overwrite);
        p2.end();
        p4.start();
        MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiStatus);
        p4.end();
        p3.start();
        MPI_Send(iBufS.data,messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD);
        p3.end();
        p4.start();
        MPI_Recv(dBufR.data,dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiStatus);
        p4.end();
        p3.start();
        MPI_Send(dBufS.data,dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD);
        p3.end();
        }
    p2.start();
    Conf->receiveData(dataReceive);
    p2.end();
    }

void multirankSimulation::communicateHaloSiteRoutine()
    {
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
            MPI_Send(&iBufS.data[startStop.x],messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD);
            MPI_Recv(&iBufR.data[receiveStart],messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(&dBufS.data[DIMENSION*startStop.x],dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD);
            MPI_Recv(&dBufR.data[DIMENSION*receiveStart],dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiStatus);
            }
        else
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive,dataLocation,access_mode::overwrite);
            ArrayHandle<scalar> dBufS(Conf->doubleTransferBufferSend,dataLocation,access_mode::read);
            ArrayHandle<scalar> dBufR(Conf->doubleTransferBufferReceive,dataLocation,access_mode::overwrite);
            MPI_Recv(&iBufR.data[receiveStart],messageSize,MPI_INT,MPI_ANY_SOURCE,messageTag1,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(&iBufS.data[startStop.x],messageSize,MPI_INT,targetRank,messageTag1,MPI_COMM_WORLD);
            MPI_Recv(&dBufR.data[DIMENSION*receiveStart],dMessageSize,MPI_SCALAR,MPI_ANY_SOURCE,messageTag2,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(&dBufS.data[DIMENSION*startStop.x],dMessageSize,MPI_SCALAR,targetRank,messageTag2,MPI_COMM_WORLD);
            }
        }
    }//end MPI routines

    //read readReceivingBuffer
    {
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
    }//end buffer read
    }

void multirankSimulation::communicateHaloSites()
    {
    //always test for sending faces
    if(rankTopology.x > 1)
        {
        //send x faces left
        directionalHaloFaceCommunication(0,rankParity.x);
        //send x faces right
        directionalHaloFaceCommunication(1,rankParity.x);
        }
    if(rankTopology.y > 1)
        {
        directionalHaloFaceCommunication(2,rankParity.y);
        directionalHaloFaceCommunication(3,rankParity.y);
        }
    if(rankTopology.z > 1)
        {
        directionalHaloFaceCommunication(4,rankParity.z);
        directionalHaloFaceCommunication(5,rankParity.z);
        }
    if(edges)
        {
        }
    if(corners)
        {
        }
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
    //communicateHaloSites();
    communicateHaloSiteRoutine();
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
    if(edges || corners)
        UNWRITTENCODE("communication pattern for non-faces unwritten");

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
    auto Conf = mConfiguration.lock();
    if(Conf->xHalo)
        Conf->prepareSendData(0);//make sure the buffers are big enough
    if(Conf->yHalo)
        Conf->prepareSendData(2);
    if(Conf->zHalo)
        Conf->prepareSendData(4);
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
        frc->computeForces(Conf->returnForces(),zeroForces);
        };
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
    Conf->getAverageEigenvalues();
    ArrayHandle<dVec> pp(Conf->returnPositions());
    ArrayHandle<int> tt(Conf->returnTypes());
    ofstream myfile;
    myfile.open(fn);
    for (int ii = 0; ii < Conf->totalSites; ++ii)
        {
        int3 pos = Conf->expandedLatticeIndex.inverseIndex(ii);
        if(Conf->xHalo)
            pos.x-=1;
        if(Conf->yHalo)
            pos.y-=1;
        if(Conf->zHalo)
            pos.z-=1;
        int idx = Conf->indexInExpandedDataArray(pos);
        myfile << pos.x <<"\t"<<pos.y<<"\t"<<pos.z;
        for (int dd = 0; dd <DIMENSION; ++dd)
            myfile <<"\t"<<pp.data[idx][dd];
        myfile << "\t"<<tt.data[idx]<<"\n";
        }

    myfile.close();

    };
