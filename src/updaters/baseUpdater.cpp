#include "baseUpdater.h"
/*! \file baseUpdater.cpp" */

void updater::performUpdate()
    {
    cout << "in the base updater... that's odd..." << endl;
    sim->computeForces();
    };

void updater::getNTotal()
    {
    if(updaterData.size()<1)
        updaterData.resize(1);

        {//scope for array handles
        nTotal = Ndof;
        ArrayHandle<int> h_t(model->returnTypes(),access_location::host,access_mode::read);
        for (int i = 0; i < Ndof; ++i)
            if(h_t.data[i] > 0)
                nTotal -= 1;

        updaterData[0] = nTotal;
        sim->sumUpdaterData(updaterData);
        nTotal = updaterData[0];
        }
    printf("nTotal set to %i\n",nTotal);
    }
