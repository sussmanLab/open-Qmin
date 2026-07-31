#ifndef velocityVerlet_H
#define velocityVerlet_H

#include "equationOfMotion.h"
/*! \file velocityVerlet.h */

class velocityVerlet : public equationOfMotion
    {
    public:
        virtual void integrateEOMCPU();
        #ifdef ENABLE_CUDA
                virtual void integrateEOMGPU();
        #endif
    };
#endif
