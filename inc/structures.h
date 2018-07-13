#ifndef structures_H
#define structures_H

/*! \file structures.h
defines simpleBond class
defines simpleAngle class
defines simpleDihedral class
*/

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif
//!simpleBond carries two integers and two scalars
class simpleBond
    {
    public:
        HOSTDEVICE simpleBond(int _i, int _j, scalar _r0,scalar _k)
            {
            setBondIndices(_i,_j);
            setStiffness(_k);
            setRestLength(_r0);
            };
        int i,j;
        scalar k,r0;
        HOSTDEVICE void setBondIndices(int _i, int _j){i=_i;j=_j;};
        HOSTDEVICE void setStiffness(scalar _k){k=_k;};
        HOSTDEVICE void setRestLength(scalar _r0){r0=_r0;};
    };
//!simpleAngle carries three integers and two scalars
class simpleAngle
    {
    public:
        HOSTDEVICE simpleAngle(int _i, int _j, int _k, scalar _t0,scalar _kt)
            {
            setAngleIndices(_i,_j,_k);
            setStiffness(_kt);
            setRestLength(_t0);
            };
        int i,j,k;
        scalar kt,t0;
        HOSTDEVICE void setAngleIndices(int _i, int _j,int _k){i=_i;j=_j;k=_k;};
        HOSTDEVICE void setStiffness(scalar _kt){kt=_kt;};
        HOSTDEVICE void setRestLength(scalar _t0){t0=_t0;};
    };
//!simpleDihedral carries four integers and two scalars
class simpleDihedral
    {
    public:
        HOSTDEVICE simpleDihedral(int _i, int _j, int _k, int _l,scalar _d0,scalar _kd)
            {
            setDihedralIndices(_i,_j,_k,_l);
            setStiffness(_kd);
            setRestLength(_d0);
            };
        int i,j,k,l;
        scalar kd,d0;
        HOSTDEVICE void setDihedralIndices(int _i, int _j,int _k,int _l){i=_i;j=_j;k=_k;l=_l;};
        HOSTDEVICE void setStiffness(scalar _kd){kd=_kd;};
        HOSTDEVICE void setRestLength(scalar _d0){d0=_d0;};
    };

#undef HOSTDEVICE
#endif
