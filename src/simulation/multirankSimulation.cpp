#include "multirankSimulation.h"
/*! \file multirankSimulation.cpp */


/*!
Calls the configuration to displace the degrees of freedom, and communicates halo sites according
to the rankTopology and boolean settings
*/
void multirankSimulation::moveParticles(GPUArray<dVec> &displacements,scalar scale)
    {
    auto Conf = mConfiguration.lock();
    Conf->moveParticles(displacements,scale);

    p1.start();
    //x first
    if(rankTopology.x > 1)
        {
        Conf->prepareSendData(0);
        int messageSize = Conf->transferElementNumber;
        int3 leftNode = rankParity; leftNode.x -=1;
        if(leftNode.x <0) leftNode.x = parityTest.sizes.x-1;
        int leftTarget = parityTest(leftNode);
        int3 rightNode = rankParity; rightNode.x +=1;
        if(rightNode.x == parityTest.sizes.x) rightNode.x = 0;
        int rightTarget = parityTest(leftNode);
        //first, send data to the left / receive onto the right
        if(rankParity.x%2 == 0) //send and receive
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive);
            ArrayHandle<dVec> dBufS(Conf->dvecTransferBufferSend);
            ArrayHandle<dVec> dBufR(Conf->dvecTransferBufferReceive);
            MPI_Send(iBufS.data,messageSize,MPI_INT,leftTarget,0,MPI_COMM_WORLD);
            MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(dBufS.data,messageSize,MPI_DOUBLE,leftTarget,1,MPI_COMM_WORLD);
            MPI_Recv(dBufR.data,messageSize,MPI_DOUBLE,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&mpiStatus);
            }
        else                    //receive and send
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive);
            ArrayHandle<dVec> dBufS(Conf->dvecTransferBufferSend);
            ArrayHandle<dVec> dBufR(Conf->dvecTransferBufferReceive);
            MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(iBufS.data,messageSize,MPI_INT,leftTarget,0,MPI_COMM_WORLD);
            MPI_Recv(dBufR.data,messageSize,MPI_DOUBLE,MPI_ANY_SOURCE,1,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(dBufS.data,messageSize,MPI_DOUBLE,leftTarget,1,MPI_COMM_WORLD);
            }
        Conf->receiveData(1);
        //...and vice versa
        Conf->prepareSendData(1);
        if(rankParity.x%2 == 0) //send and receive
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive);
            ArrayHandle<dVec> dBufS(Conf->dvecTransferBufferSend);
            ArrayHandle<dVec> dBufR(Conf->dvecTransferBufferReceive);
            MPI_Send(iBufS.data,messageSize,MPI_INT,rightTarget,2,MPI_COMM_WORLD);
            MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(dBufS.data,messageSize,MPI_DOUBLE,rightTarget,3,MPI_COMM_WORLD);
            MPI_Recv(dBufR.data,messageSize,MPI_DOUBLE,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&mpiStatus);
            }
        else                    //receive and send
            {
            ArrayHandle<int> iBufS(Conf->intTransferBufferSend);
            ArrayHandle<int> iBufR(Conf->intTransferBufferReceive);
            ArrayHandle<dVec> dBufS(Conf->dvecTransferBufferSend);
            ArrayHandle<dVec> dBufR(Conf->dvecTransferBufferReceive);
            MPI_Recv(iBufR.data,messageSize,MPI_INT,MPI_ANY_SOURCE,2,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(iBufS.data,messageSize,MPI_INT,rightTarget,2,MPI_COMM_WORLD);
            MPI_Recv(dBufR.data,messageSize,MPI_DOUBLE,MPI_ANY_SOURCE,3,MPI_COMM_WORLD,&mpiStatus);
            MPI_Send(dBufS.data,messageSize,MPI_DOUBLE,rightTarget,3,MPI_COMM_WORLD);
            }
        Conf->receiveData(0);
        }
    p1.end();
    };

void multirankSimulation::setRankTopology(int x, int y, int z, bool _edges, bool _corners)
    {
    rankTopology.x=x;
    rankTopology.y=y;
    rankTopology.z=z;
    int Px, Py, Pz;
    parityTest = Index3D(rankTopology);
    rankParity = parityTest.inverseIndex(myRank);
    edges = _edges;
    corners = _corners;
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
    Conf->prepareSendData(0);//make sure the buffers are big enough
    Conf->prepareSendData(2);
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
