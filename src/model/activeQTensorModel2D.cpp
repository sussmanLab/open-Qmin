#include "activeQTensorModel2D.h"

activeQTensorModel2D::activeQTensorModel2D(int l, bool _useGPU, bool _neverGPU)
    : qTensorLatticeModel2D(l,_useGPU, _neverGPU)
    {
    initializeDataStructures();
    };
activeQTensorModel2D::activeQTensorModel2D(int lx, int ly, bool _useGPU, bool _neverGPU)
    : qTensorLatticeModel2D(lx,ly,_useGPU, neverGPU)
    {
    initializeDataStructures();
    };

void activeQTensorModel2D::initializeDataStructures()
    {
    if(neverGPU)
        {
        alternateNeighboringSites.noGPU = true;
        pressure.noGPU = true;
        symmetricStress.noGPU = true;
        antisymmetricStress.noGPU = true;
        }
    pressure.resize(N);
    symmetricStress.resize(N);
    antisymmetricStress.resize(N);
    vector<scalar> z(N,0.);
    fillGPUArrayWithVector(z,pressure);
    vector<dVec> zeroes(N,make_dVec(0.0));
    fillGPUArrayWithVector(zeroes, symmetricStress);
    fillGPUArrayWithVector(zeroes, antisymmetricStress);
    fillNeighborLists(2,alternateNeighboringSites,alternateNeighborIndex);
    };
