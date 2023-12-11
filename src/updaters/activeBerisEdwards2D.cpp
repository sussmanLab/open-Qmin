#include "activeBerisEdwards2D.h"
#include "utilities.cuh"

void activeBerisEdwards2D::initializeFromModel()
    {
    iterations = 0;
    Ndof = activeModel->getNumberOfParticles();
    neverGPU = activeModel->neverGPU;
    if(neverGPU)
        {
        displacement.noGPU = true;
        generalizedAdvection.noGPU = true;
        velocityUpdate.noGPU=true;
        }
    displacement.resize(Ndof);
    generalizedAdvection.resize(Ndof);
    velocityUpdate.resize(Ndof);

    vector<dVec> zeroes(Ndof,make_dVec(0.0));
    fillGPUArrayWithVector(zeroes,displacement);
    fillGPUArrayWithVector(zeroes,generalizedAdvection);
    fillGPUArrayWithVector(zeroes,velocityUpdate);
    };

void activeBerisEdwards2D::integrateEOMGPU()
    {
    UNWRITTENCODE("2D active Beris-Edwards GPU branch");    
    };

void activeBerisEdwards2D::integrateEOMCPU()
    {
    calculateMolecularFieldAdvectionStressCPU();
    relaxPressureCPU();
    updateVelocityFieldCPU();
    };

void activeBerisEdwards2D::calculateMolecularFieldAdvectionStressCPU()
    {
    //the molecular field is calculated by the active model, according to whichever version of the landau theory it is implementing
    sim->computeForces();

    //loop through lattice sites, calculating the strain tensor, vorticity tensor, the generalized advection tensor, and the symmetric/antisymmetric stress tensors
    ArrayHandle<dVec> Q(activeModel->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<dVec> v(activeModel->returnVelocities(),access_location::host,access_mode::read);
    ArrayHandle<dVec> H(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<dVec> advection(generalizedAdvection);
    ArrayHandle<dVec> PiS(activeModel->symmetricStress);
    ArrayHandle<dVec> PiA(activeModel->antisymmetricStress);
    ArrayHandle<scalar> p(activeModel->pressure);
    ArrayHandle<int> nearestNeighbors(activeModel->neighboringSites,access_location::host,access_mode::read);
    dVec q,h;
    int ixd, ixu,iyd,iyu;
    scalar dxux,dxuy,dyux,omegaxy, localS;
    for (int ii = 0; ii < Ndof; ++ii)
        {
        q = Q.data[ii];
        h = H.data[ii];
        //lattice indices of four nearest neighbors
        ixd = nearestNeighbors.data[activeModel->neighborIndex(0,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(1,ii)];
        iyd = nearestNeighbors.data[activeModel->neighborIndex(2,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(3,ii)];

        //relevant strain and vorticity terms
        dxux = 0.5*(v.data[ixu].x[0] - v.data[ixd].x[0]);
        dxuy = 0.5*(v.data[ixu].x[1] - v.data[ixd].x[1]);
        dyux = 0.5*(v.data[iyu].x[0] - v.data[iyd].x[0]);
        omegaxy = 0.5*(dxuy-dyux);

        //update the generalized advection and stress terms
        localS = sqrt(q[0]*q[0]+q[1]*q[1]);
        advection.data[ii].x[0] = lambda*localS*dxux - 2.0*omegaxy*q[1];
        advection.data[ii].x[1] = lambda*localS*0.5*(dxuy+dyux) + 2.0*omegaxy*q[0];

        PiS.data[ii] = -lambda*h - zeta*q;
        PiA.data[ii].x[0] = 2.0*(q[0]*h[1] - h[0]*q[1]);
        };
    };

void activeBerisEdwards2D::relaxPressureCPU()
    {

    };
/*!
returns -(\vec{u}\cdot\nabla) f
This function could be optimized by splitting into multiple functions (so that less data needs to be accessed)
*/
dVec upwindAdvectiveDerivative(dVec &u, dVec &f, dVec &fxd, dVec &fyd, dVec &fxu, dVec &fyu, dVec &fxdd, dVec &fydd, dVec &fxuu, dVec &fyuu)
    {
    dVec ans;
    if(u[0] >0)
        {
        ans = -0.5*u[0]*(3.*f - 4.*fxd + 1.*fxdd);
        }
    else
        {
        ans = 0.5*u[0]*(3.*f - 4.*fxu + 1.*fxuu);
        }
    if(u[1] >0)
        {
        ans += -0.5*u[1]*(3.*f - 4.*fyd + 1.*fydd);
        }
    else
        {
        ans += 0.5*u[0]*(3.*f - 4.*fyu + 1.*fxuu);
        }

    return ans;
    };

void activeBerisEdwards2D::updateQFieldCPU()
    {
        {//scope for array handles
    ArrayHandle<dVec> disp(displacement);
    ArrayHandle<dVec> Q(activeModel->returnPositions(),access_location::host,access_mode::read);
    ArrayHandle<dVec> V(activeModel->returnVelocities(),access_location::host,access_mode::read);
    ArrayHandle<dVec> H(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<dVec> advection(generalizedAdvection);
    ArrayHandle<int> nearestNeighbors(activeModel->neighboringSites,access_location::host,access_mode::read);
    ArrayHandle<int> alternateNeighbors(activeModel->alternateNeighboringSites,access_location::host,access_mode::read);
        
    dVec dqdt,h,s,v;
    int ixd, ixu,iyd,iyu,ixdd,ixuu,iydd,iyuu;
    for (int ii = 0; ii < Ndof; ++ii)
        {
        h = H.data[ii];
        v = V.data[ii];
        s = advection.data[ii];
        //lattice indices of four nearest neighbors
        ixd = nearestNeighbors.data[activeModel->neighborIndex(0,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(1,ii)];
        iyd = nearestNeighbors.data[activeModel->neighborIndex(2,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(3,ii)];
        ixdd = alternateNeighbors.data[activeModel->alternateNeighborIndex(0,ii)];
        ixuu = alternateNeighbors.data[activeModel->alternateNeighborIndex(1,ii)];
        iydd = alternateNeighbors.data[activeModel->alternateNeighborIndex(2,ii)];
        iyuu = alternateNeighbors.data[activeModel->alternateNeighborIndex(3,ii)];
        
        disp.data[ii] = deltaT*((1.0/rotationalViscosity)*h+s + upwindAdvectiveDerivative(v,Q.data[ii],
                                                                Q.data[ixd],Q.data[iyd],Q.data[ixu],Q.data[iyu],
                                                                Q.data[ixdd],Q.data[iydd],Q.data[ixuu],Q.data[iyuu])
                                );
        };

        }//array handle scope end
    sim->moveParticles(displacement);
    }; 

/*!
\partial_t \vec{u} = -(\vec{u}\cdot\nabla)\vec{u} + (viscosity)*\nabla^2\vec{u} + (1/rho)*(\vec{F} - \nabla p)
*/
void activeBerisEdwards2D::updateVelocityFieldCPU()
    {
        {//scope for array handles
    ArrayHandle<dVec> V(activeModel->returnVelocities());
    ArrayHandle<dVec> disp(velocityUpdate);
    ArrayHandle<dVec> PiS(activeModel->symmetricStress);
    ArrayHandle<dVec> PiA(activeModel->antisymmetricStress);
    ArrayHandle<scalar> p(activeModel->pressure);
    ArrayHandle<int> nearestNeighbors(activeModel->neighboringSites,access_location::host,access_mode::read);
    ArrayHandle<int> alternateNeighbors(activeModel->alternateNeighboringSites,access_location::host,access_mode::read);
        
    dVec dudt,v;
    int ixd, ixu,iyd,iyu,ixdd,ixuu,iydd,iyuu,ixdyd, ixdyu, ixuyd, ixuyu;
    for (int ii = 0; ii < Ndof; ++ii)
        {
        v = V.data[ii];
        //lattice indices of four nearest neighbors
        ixd = nearestNeighbors.data[activeModel->neighborIndex(0,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(1,ii)];
        iyd = nearestNeighbors.data[activeModel->neighborIndex(2,ii)];
        ixu = nearestNeighbors.data[activeModel->neighborIndex(3,ii)];
        ixdyd =nearestNeighbors.data[activeModel->neighborIndex(4,ii)];
        ixdyu =nearestNeighbors.data[activeModel->neighborIndex(5,ii)];
        ixuyd =nearestNeighbors.data[activeModel->neighborIndex(6,ii)];
        ixuyu =nearestNeighbors.data[activeModel->neighborIndex(7,ii)];
        ixdd = alternateNeighbors.data[activeModel->alternateNeighborIndex(0,ii)];
        ixuu = alternateNeighbors.data[activeModel->alternateNeighborIndex(1,ii)];
        iydd = alternateNeighbors.data[activeModel->alternateNeighborIndex(2,ii)];
        iyuu = alternateNeighbors.data[activeModel->alternateNeighborIndex(3,ii)];

        //convective term
        dudt = upwindAdvectiveDerivative(v,v,
                                        V.data[ixd],V.data[iyd],V.data[ixu],V.data[iyu],
                                        V.data[ixdd],V.data[iydd],V.data[ixuu],V.data[iyuu]);
        //add viscous term
        dudt = dudt + laplacianStencil(viscosity,v,
                                    V.data[ixd],V.data[ixu],V.data[iyd],V.data[iyu],
                                    V.data[ixdyd],V.data[ixuyd],V.data[ixdyu],V.data[ixuyu]);

        //add pressure and active/elastic stress terms:. F_x = dx Pixx + dy Pixy,
        dudt[0] += (0.5/rho)*(-(p.data[ixu] - p.data[ixd]) 
                                // F_x = dx Pixx + dy Pixy,
                                + (PiS.data[ixu].x[0] - PiS.data[ixd].x[0] ) 
                                + ((PiS.data[iyu].x[1]+PiA.data[iyu].x[0])-(PiS.data[iyd].x[1]+PiA.data[iyd].x[0]) )
                                );
        dudt[1] += (0.5/rho)*(-(p.data[iyu] - p.data[iyd]) 
                                // F_y = dx Piyx + dy Piyy = -dy Pixx + dx Pi yx,
                                - (PiS.data[iyu].x[0] - PiS.data[iyd].x[0] ) 
                                + ((PiS.data[ixu].x[1]-PiA.data[ixu].x[0])-(PiS.data[ixd].x[1]-PiA.data[ixd].x[0]) )
                                );

        //scale by deltaT
        disp.data[ii] = deltaT*dudt;
        };
        }//array handle scope end
    
    //update all velocities
    ArrayHandle<dVec> V(activeModel->returnVelocities());
    ArrayHandle<dVec> disp(velocityUpdate);
    for (int ii = 0; ii < Ndof; ++ii)
        V.data[ii] = V.data[ii] + disp.data[ii];
        
    };
