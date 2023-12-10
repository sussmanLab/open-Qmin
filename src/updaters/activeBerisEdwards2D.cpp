#include "activeBerisEdwards2D.h"
#include "utilities.cuh"

void activeBerisEdwards2D::initializeFromModel()
    {
    /*
    iterations = 0;
    Ndof = model->getNumberOfParticles();
    displacement.resize(Ndof);
    vector<dVec> zeroes(Ndof,make_dVec(0.0));
    fillGPUArrayWithVector(zeroes,biasedMomentumEstimate);
    fillGPUArrayWithVector(zeroes,biasedMomentumSquaredEstimate);
    fillGPUArrayWithVector(zeroes,correctedMomentumEstimate);
    fillGPUArrayWithVector(zeroes,correctedMomentumSquaredEstimate);
    */
    };

void activeBerisEdwards2D::integrateEOMGPU()
    {
    UNWRITTENCODE("2D active Beris-Edwards GPU branch");    
    };

void activeBerisEdwards2D::integrateEOMCPU()
    {
    UNWRITTENCODE("sdfsdf");
    };

void activeBerisEdwards2D::calculateStrainAndVorticity()
    {
        
    };

void activeBerisEdwards2D::relaxPressure()
    {

    };


void activeBerisEdwards2D::updateQField()
    {
        
    }; 


void activeBerisEdwards2D::updateVelocityField()
    {
        
    };
