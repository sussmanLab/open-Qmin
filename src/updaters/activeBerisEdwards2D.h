#ifndef ACTIVEBERISEDWARDS_H
#define ACTIVEBERISEDWARDS_H

#include "equationOfMotion.h"
#include "activeQTensorModel2D.h"
/*! \file activeBerisEdwards2D.h */

class activeBerisEdwards2D : public equationOfMotion
    {
    public:
        virtual void integrateEOMGPU();
        virtual void integrateEOMCPU();

        virtual void initializeFromModel();

        //! A pointer to the active model that contains the extra data structures needed 
        shared_ptr<activeQTensorModel2D> activeModel;
        //! virtual function to allow the model to be a derived class
        virtual void setModel(shared_ptr<simpleModel> _model)
            {
            model=_model;
            activeModel = dynamic_pointer_cast<activeQTensorModel2D>(model);
            initializeFromModel();
            };
    protected:
        void relaxPressure();
        void calculateStrainAndVorticity();
        void updateQField();
        void updateVelocityField();
    };
#endif
