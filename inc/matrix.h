#ifndef Matrix_H
#define Matrix_H

#include "std_include.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

/*! \file Matrix.h */
//!contains a {{x11,x12},{x21,x22}} set, and matrix manipulations, and a (d x d) structure
/*!
Matrix2x2 provides a simple interface for operations using 2x2 matrices. In particular, it implement
matrix-maxtrix multiplication, and has specialized matrix-vector and vector-matrix multiplication in
which scalar2 variables take the place of vectors. A dyadic product is implemented which takes two
scalar2s and returns a Matrix2x2
*/
struct Matrix2x2
    {
    public:
        //!The entries of the matrix
        scalar x11, x12, x21, x22;
        //!Default constructor is the identity matrix
        HOSTDEVICE Matrix2x2() : x11(1.0), x12(0.0), x21(0.0),x22(1.0) {};
        //!Generic constructor is whatever you wnat it to be
        HOSTDEVICE Matrix2x2(scalar y11, scalar y12, scalar y21,scalar y22) : x11(y11), x12(y12), x21(y21),x22(y22) {};

        //!Set the values to some desired set
        HOSTDEVICE void set(scalar y11, scalar y12, scalar y21, scalar y22)
                            {
                            x11=y11; x12=y12;x21=y21;x22=y22;
                            };

        //!Transpose
        HOSTDEVICE void transpose()
                            {
                            scalar y21,y12;
                            y21=x12;
                            y12=x21;
                            x12=y12;
                            x21=y21;
                            };

        //!assignment operator
        HOSTDEVICE void operator=(const Matrix2x2 &m2)
                            {
                            set(m2.x11,m2.x12,m2.x21,m2.x22);
                            };
        //!matrix multiplication operator
        HOSTDEVICE void operator*=(const Matrix2x2 &m2)
                            {
                            set(x11*m2.x11 + x12*m2.x21,
                                x11*m2.x12 + x12*m2.x22,
                                x21*m2.x11 + x22*m2.x21,
                                x21*m2.x12 + x22*m2.x22
                                );
                            };

        //!matrix multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp*=m2;
                            return temp;
                            };

        //!scalar multiplication operator
        HOSTDEVICE void operator*=(scalar a)
                            {
                            set(a*x11,a*x12,a*x21,a*x22);
                            };

        //!scalar right multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const Matrix2x2 &m,const scalar a)
                            {
                            Matrix2x2 temp(m);
                            temp*=a;
                            return temp;
                            };

        //!scalar left multiplication operator
        HOSTDEVICE friend Matrix2x2 operator*(const scalar a, const Matrix2x2 &m)
                            {
                            Matrix2x2 temp(m);
                            temp*=a;
                            return temp;
                            };

        //!Matrix addition operator
        HOSTDEVICE void operator+=(const Matrix2x2 &m2)
                            {
                            set(x11+m2.x11,
                                x12+m2.x12,
                                x21+m2.x21,
                                x22+m2.x22
                               );
                            };

        //!Matrix addition operator
        HOSTDEVICE friend Matrix2x2 operator+(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp+=m2;
                            return temp;
                            };

        //!Matrix subtraction operator
        HOSTDEVICE void operator-=(const Matrix2x2 &m2)
                            {
                            set(x11-m2.x11,
                                x12-m2.x12,
                                x21-m2.x21,
                                x22-m2.x22
                               );
                            };

        //!matrix subtraction operator
        HOSTDEVICE friend Matrix2x2 operator-(const Matrix2x2 &m1,const Matrix2x2 &m2)
                            {
                            Matrix2x2 temp(m1);
                            temp-=m2;
                            return temp;
                            };

        //!matrix-vector multiplication operator
        HOSTDEVICE friend scalar2 operator*(const scalar2 &v, const Matrix2x2 &m)
                            {
                            scalar2 temp;
                            temp.x = v.x*m.x11 + v.y*m.x21;
                            temp.y = v.x*m.x12 + v.y*m.x22;
                            return temp;
                            };

        //!matrix-vector multiplication operator
        HOSTDEVICE friend scalar2 operator*(const Matrix2x2 &m, const scalar2 &v)
                            {
                            scalar2 temp;
                            temp.x = m.x11*v.x+m.x12*v.y;
                            temp.y = m.x21*v.x+m.x22*v.y;
                            return temp;
                            };

    };

//! Form a matrix by the dyadic product of two vectors
HOSTDEVICE Matrix2x2 dyad(const scalar2 &v1, const scalar2 &v2)
    {
    return Matrix2x2(v1.x*v2.x,v1.x*v2.y,v1.y*v2.x,v1.y*v2.y);
    };

inline void printMatrix(Matrix2x2 &m)
    {
    cout << endl << m.x11 << "    " << m.x12 << endl <<m.x21<<"    "<<m.x22<<endl<<endl;
    };


/*!
MatrixDxD provides a simple interface for operations using DxD matrices. In particular, it implement
matrix-maxtrix multiplication, and has specialized matrix-vector and vector-matrix multiplication in
which dVecs variables take the place of vectors. A dyadic product is implemented which takes two
dVecs and returns a MatrixDxD
*/
struct MatrixDxD
    {
    public:
        //!The entries of the matrix
        vector<dVec> mat;
        //!Default constructor is the identity matrix
        HOSTDEVICE MatrixDxD(bool makeIdentity = true)
            {
            mat.resize(DIMENSION);
            for (int dd = 0; dd <DIMENSION; ++dd)
                {
                mat[dd] = make_dVec(0.);
                if(makeIdentity)
                    mat[dd].x[dd] = 1.0;
                };
            };
        //!Generic constructor is whatever you want it to be
        HOSTDEVICE MatrixDxD(vector<dVec> &_mat){mat = _mat;};

        //!Set the values to some desired set
        HOSTDEVICE void set(const vector<dVec> &_mat){mat = _mat;};

        //!Transpose
        /*
        HOSTDEVICE void transpose()
                            {
                            scalar y21,y12;
                            y21=x12;
                            y12=x21;
                            x12=y12;
                            x21=y21;
                            };
        */

        //!assignment operator
        HOSTDEVICE void operator=(const MatrixDxD &m2)
                            {
                            set(m2.mat);
                            };
        //!matrix multiplication operator
        HOSTDEVICE void operator*=(const MatrixDxD &m2)
                            {
                            MatrixDxD temp;
                            for(int i = 0; i < DIMENSION; ++i)
                                for(int j = 0; j < DIMENSION; ++j)
                                    {
                                    temp.mat[i].x[j] = 0.0;
                                    for(int k  = 0; k < DIMENSION; ++k)
                                        temp.mat[i].x[j] += mat[i].x[k]*m2.mat[k].x[j];
                                    }
                            set(temp.mat);
                            };

        //!matrix multiplication operator
        HOSTDEVICE friend MatrixDxD operator*(const MatrixDxD &m1,const MatrixDxD &m2)
                            {
                            MatrixDxD temp(m1);
                            temp*=m2;
                            return temp;
                            };

        //!scalar multiplication operator
        HOSTDEVICE void operator*=(scalar a)
                            {
                            for(int i = 0; i < DIMENSION; ++i)
                                for(int j = 0; j < DIMENSION; ++j)
                                    mat[i].x[j] *= a;
                            };

        //!scalar right multiplication operator
        HOSTDEVICE friend MatrixDxD operator*(const MatrixDxD &m,const scalar a)
                            {
                            MatrixDxD temp(m);
                            temp*=a;
                            return temp;
                            };

        //!scalar left multiplication operator
        HOSTDEVICE friend MatrixDxD operator*(const scalar a, const MatrixDxD &m)
                            {
                            MatrixDxD temp(m);
                            temp*=a;
                            return temp;
                            };

        //!Matrix addition operator
        HOSTDEVICE void operator+=(const MatrixDxD &m2)
                            {
                            for(int i = 0; i < DIMENSION; ++i)
                                for(int j = 0; j < DIMENSION; ++j)
                                    mat[i].x[j] += m2.mat[i].x[j];
                            };

        //!Matrix addition operator
        HOSTDEVICE friend MatrixDxD operator+(const MatrixDxD &m1,const MatrixDxD &m2)
                            {
                            MatrixDxD temp(m1);
                            temp+=m2;
                            return temp;
                            };

        //!Matrix subtraction operator
        HOSTDEVICE void operator-=(const MatrixDxD &m2)
                            {
                            for(int i = 0; i < DIMENSION; ++i)
                                for(int j = 0; j < DIMENSION; ++j)
                                    mat[i].x[j] -= m2.mat[i].x[j];
                            };

        //!matrix subtraction operator
        HOSTDEVICE friend MatrixDxD operator-(const MatrixDxD &m1,const MatrixDxD &m2)
                            {
                            MatrixDxD temp(m1);
                            temp-=m2;
                            return temp;
                            };

        //!matrix-vector multiplication operator
        HOSTDEVICE friend dVec operator*(const MatrixDxD &m, const dVec &v)
                            {
                            dVec temp;
                            for (int i =0; i < DIMENSION; ++i)
                                {
                                temp.x[i] = 0;
                                for (int j =0; j < DIMENSION; ++j)
                                    temp.x[i] += m.mat[i].x[j]*v.x[j];
                                };
                            return temp;
                            };

    };

//! Form a matrix by the dyadic product of two vectors
HOSTDEVICE MatrixDxD dyad(const dVec &v1, const dVec &v2)
    {
    MatrixDxD temp;
    for(int i = 0; i < DIMENSION; ++i)
        for(int j = 0; j < DIMENSION; ++j)
        temp.mat[i].x[j] = v1.x[i]*v2.x[j];
    return temp;
    };

//! Form a matrix by the dyadic product of two vectors
HOSTDEVICE scalar  trace(const MatrixDxD &m)
    {
    scalar ans = 0.0;
    for(int i = 0; i < DIMENSION; ++i)
        ans += m.mat[i].x[i];
    return ans;
    };
#undef HOSTDEVICE
#endif
