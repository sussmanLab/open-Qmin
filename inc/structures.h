#ifndef structures
#define structures

/*! \file structures.h 
defines dVec class
defines simpleBond class
defines simpleAngle class
*/


//!dVec is an array whose length matches the dimension of the system
class dVec
    {
    public:
        HOSTDEVICE dVec(){};
        HOSTDEVICE dVec(const scalar value)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = value;
            };
        HOSTDEVICE dVec(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                x[dd] = other.x[dd];
            };

        scalar x[DIMENSION];

        //mutating operators
        dVec& operator=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] = other.x[dd];
            return *this;
            }
        dVec& operator-=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] -= other.x[dd];
            return *this;
            }
        dVec& operator+=(const dVec &other)
            {
            for (int dd = 0; dd < DIMENSION; ++dd)
                this->x[dd] += other.x[dd];
            return *this;
            }
    };

//!Less than operator for dVecs just sorts by the x-coordinate
HOSTDEVICE bool operator<(const dVec &a, const dVec &b)
    {
    return a.x[0]<b.x[0];
    }

//!Equality operator tests for.... equality of all elements
HOSTDEVICE bool operator==(const dVec &a, const dVec &b)
    {
    for (int dd = 0; dd <DIMENSION; ++dd)
        if(a.x[dd]!= b.x[dd]) return false;
    return true;
    }

//!return a dVec with all elements equal to one number
HOSTDEVICE dVec make_dVec(scalar value)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = value;
    return ans;
    }

//!component-wise addition of two dVecs
HOSTDEVICE dVec operator+(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]+b.x[dd];
    return ans;
    }

//!component-wise subtraction of two dVecs
HOSTDEVICE dVec operator-(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]-b.x[dd];
    return ans;
    }

//!component-wise multiplication of two dVecs
HOSTDEVICE dVec operator*(const dVec &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a.x[dd]*b.x[dd];
    return ans;
    }

//!multiplication of dVec by scalar
HOSTDEVICE dVec operator*(const scalar &a, const dVec &b)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a*b.x[dd];
    return ans;
    }

//!multiplication of dVec by scalar
HOSTDEVICE dVec operator*(const dVec &b, const scalar &a)
    {
    dVec ans;
    for (int dd = 0; dd < DIMENSION; ++dd)
        ans.x[dd] = a*b.x[dd];
    return ans;
    }

//!print a dVec to screen
inline __attribute__((always_inline)) void printdVec(dVec a)
    {
    cout <<"{";
    for (int dd = 0; dd < DIMENSION; ++dd)
        if(dd != DIMENSION-1)
            cout << a.x[dd] <<", ";
        else
            cout << a.x[dd];

    cout << "}" << endl;
    };

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

#endif
