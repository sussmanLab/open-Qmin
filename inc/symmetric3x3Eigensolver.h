#ifndef SymmetricEigensolver3x3_H
#define SymmetricEigensolver3x3_H

#ifdef __NVCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

#include <vector>
#include <array>
#include <cmath>
#include <limits>

//this class throws a lot of __host__ __device__ warnings.... suppress them
#pragma hd_warning_disable
#pragma diag_suppress 2739

// David Eberly, Geometric Tools, Redmond WA 98052
// Copyright (c) 1998-2018
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
// File Version: 3.0.5 (2018/10/09)

// The document
// http://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
// describes algorithms for solving the eigensystem associated with a 3x3
// symmetric real-valued matrix.  The iterative algorithm is implemented
// by class SymmmetricEigensolver3x3.  The noniterative algorithm is
// implemented by class NISymmetricEigensolver3x3.  The code does not use
// GTEngine objects.

class NISymmetricEigensolver3x3
{
public:
    // The input matrix must be symmetric, so only the unique elements must
    // be specified: a00, a01, a02, a11, a12, and a22.  The eigenvalues are
    // sorted in ascending order: eval0 <= eval1 <= eval2.

    HOSTDEVICE void operator()(scalar a00, scalar a01, scalar a02, scalar a11, scalar a12, scalar a22,
        std::array<scalar, 3>& eval, std::array<std::array<scalar, 3>, 3>& evec) const;

private:
    HOSTDEVICE static std::array<scalar, 3> Multiply(scalar s, std::array<scalar, 3> const& U);
    HOSTDEVICE static std::array<scalar, 3> Subtract(std::array<scalar, 3> const& U, std::array<scalar, 3> const& V);
    HOSTDEVICE static std::array<scalar, 3> Divide(std::array<scalar, 3> const& U, scalar s);
    HOSTDEVICE static scalar Dot(std::array<scalar, 3> const& U, std::array<scalar, 3> const& V);
    HOSTDEVICE static std::array<scalar, 3> Cross(std::array<scalar, 3> const& U, std::array<scalar, 3> const& V);

    HOSTDEVICE void ComputeOrthogonalComplement(std::array<scalar, 3> const& W,
        std::array<scalar, 3>& U, std::array<scalar, 3>& V) const;

    HOSTDEVICE void ComputeEigenvector0(scalar a00, scalar a01, scalar a02, scalar a11, scalar a12, scalar a22,
        scalar eval0, std::array<scalar, 3>& evec0) const;

    HOSTDEVICE void ComputeEigenvector1(scalar a00, scalar a01, scalar a02, scalar a11, scalar a12, scalar a22,
        std::array<scalar, 3> const& evec0, scalar eval1, std::array<scalar, 3>& evec1) const;
};

void NISymmetricEigensolver3x3::operator() (scalar a00, scalar a01, scalar a02,
    scalar a11, scalar a12, scalar a22, std::array<scalar, 3>& eval,
    std::array<std::array<scalar, 3>, 3>& evec) const
{
    // Precondition the matrix by factoring out the maximum absolute value
    // of the components.  This guards against floating-point overflow when
    // computing the eigenvalues.
    scalar max0 = (std::abs(a00)> std::abs(a01)) ? std::abs(a00) : std::abs(a01);
    scalar max1 = (std::abs(a02)> std::abs(a11)) ? std::abs(a02) : std::abs(a11);
    scalar max2 = (std::abs(a12)> std::abs(a22)) ? std::abs(a12) : std::abs(a22);
    scalar maxAbsElement = (max0>max1) ? max0 : max1;
    maxAbsElement = (max2>maxAbsElement) ? max2 : maxAbsElement;

    if (maxAbsElement == (scalar)0)
    {
        // A is the zero matrix.
        eval[0] = (scalar)0;
        eval[1] = (scalar)0;
        eval[2] = (scalar)0;
        evec[0] = { (scalar)1, (scalar)0, (scalar)0 };
        evec[1] = { (scalar)0, (scalar)1, (scalar)0 };
        evec[2] = { (scalar)0, (scalar)0, (scalar)1 };
        return;
    }

    scalar invMaxAbsElement = (scalar)1 / maxAbsElement;
    a00 *= invMaxAbsElement;
    a01 *= invMaxAbsElement;
    a02 *= invMaxAbsElement;
    a11 *= invMaxAbsElement;
    a12 *= invMaxAbsElement;
    a22 *= invMaxAbsElement;

    scalar norm = a01 * a01 + a02 * a02 + a12 * a12;
    if (norm > (scalar)0)
    {
        // Compute the eigenvalues of A.

        // In the PDF mentioned previously, B = (A - q*I)/p, where q = tr(A)/3
        // with tr(A) the trace of A (sum of the diagonal entries of A) and where
        // p = sqrt(tr((A - q*I)^2)/6).
        scalar q = (a00 + a11 + a22) / (scalar)3;

        // The matrix A - q*I is represented by the following, where b00, b11 and
        // b22 are computed after these comments,
        //   +-           -+
        //   | b00 a01 a02 |
        //   | a01 b11 a12 |
        //   | a02 a12 b22 |
        //   +-           -+
        scalar b00 = a00 - q;
        scalar b11 = a11 - q;
        scalar b22 = a22 - q;

        // The is the variable p mentioned in the PDF.
        scalar p = std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * (scalar)2) / (scalar)6);

        // We need det(B) = det((A - q*I)/p) = det(A - q*I)/p^3.  The value
        // det(A - q*I) is computed using a cofactor expansion by the first
        // row of A - q*I.  The cofactors are c00, c01 and c02 and the
        // determinant is b00*c00 - a01*c01 + a02*c02.  The det(B) is then
        // computed finally by the division with p^3.
        scalar c00 = b11 * b22 - a12 * a12;
        scalar c01 = a01 * b22 - a12 * a02;
        scalar c02 = a01 * a12 - b11 * a02;
        scalar det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);

        // The halfDet value is cos(3*theta) mentioned in the PDF. The acos(z)
        // function requires |z| <= 1, but will fail silently and return NaN
        // if the input is larger than 1 in magnitude.  To avoid this problem
        // due to rounding errors, the halfDet/ value is clamped to [-1,1].
        scalar halfDet = det * (scalar)0.5;
        if(halfDet <-1.)
            halfDet = -1.;
        if(halfDet > 1.)
            halfDet = 1.;

        // The eigenvalues of B are ordered as beta0 <= beta1 <= beta2.  The
        // number of digits in twoThirdsPi is chosen so that, whether float or
        // double, the floating-point number is the closest to theoretical 2*pi/3.
        scalar angle = std::acos(halfDet) / (scalar)3;
        scalar const twoThirdsPi = (scalar)2.09439510239319549;
        scalar beta2 = std::cos(angle) * (scalar)2;
        scalar beta0 = std::cos(angle + twoThirdsPi) * (scalar)2;
        scalar beta1 = -(beta0 + beta2);

        // The eigenvalues of A are ordered as alpha0 <= alpha1 <= alpha2.
        eval[0] = q + p * beta0;
        eval[1] = q + p * beta1;
        eval[2] = q + p * beta2;
        vector<scalar> eVals(3);eVals[0] = eval[0];eVals[1]=eval[1];eVals[2]=eval[2];
        eVals[0] = (eval[0] < eval[1]) ? (eval[0]< eval[2] ? eval[0] : eval[2])  : ((eval[1]< eval[2]) ? eval[1] : eval[2]);
        eVals[1] = (eval[0] < eval[1]) ? (eval[1]< eval[2] ? eval[1] : eval[2])  : ((eval[0]< eval[2]) ? eval[0] : eval[2]);
        eVals[2] = (eval[0] > eval[1]) ? (eval[0] > eval[2] ? eval[0] : eval[2])  : ((eval[1] > eval[2]) ? eval[1] : eval[2]);
        //sort(eVals.begin(),eVals.end());
        eval[0]=eVals[0];eval[1]=eVals[1];eval[2]=eVals[2];
        ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eVals[2], evec[2]);
        ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eVals[1], evec[1]);
        ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eVals[0], evec[0]);
        //ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[0], eVals[1], evec[1]);
        //evec[2] = Cross(evec[0], evec[1]);
/*
        // Compute the eigenvectors so that the set {evec[0], evec[1], evec[2]}
        // is right handed and orthonormal.
        if (halfDet >= (scalar)0)
        {
            ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[2], evec[2]);
            ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[2], eval[1], evec[1]);
            evec[0] = Cross(evec[1], evec[2]);
        }
        else
        {
            ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[0], evec[0]);
            ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[0], eval[1], evec[1]);
            evec[2] = Cross(evec[0], evec[1]);
        }
*/
    }
    else
    {
        // The matrix is diagonal.
        eval[0] = a00;
        eval[1] = a11;
        eval[2] = a22;
        //evec[0] = { (scalar)1, (scalar)0, (scalar)0 };
        //evec[1] = { (scalar)0, (scalar)1, (scalar)0 };
        //evec[2] = { (scalar)0, (scalar)0, (scalar)1 };
        vector<scalar> eVals(3);eVals[0] = eval[0];eVals[1]=eval[1];eVals[2]=eval[2];
        eVals[0] = (eval[0] < eval[1]) ? (eval[0]< eval[2] ? eval[0] : eval[2])  : ((eval[1]< eval[2]) ? eval[1] : eval[2]);
        eVals[1] = (eval[0] < eval[1]) ? (eval[1]< eval[2] ? eval[1] : eval[2])  : ((eval[0]< eval[2]) ? eval[0] : eval[2]);
        eVals[2] = (eval[0] > eval[1]) ? (eval[0] > eval[2] ? eval[0] : eval[2])  : ((eval[1] > eval[2]) ? eval[1] : eval[2]);
        //sort(eVals.begin(),eVals.end());
        if(eVals[0] == a00)
            evec[0] = { (scalar)1, (scalar)0, (scalar)0 };
        else if (eVals[0] == a11)
            evec[0] = { (scalar)0, (scalar)1, (scalar)0 };
        else if (eVals[0] == a22)
            evec[0] = { (scalar)0, (scalar)0, (scalar)1 };

        if(eVals[1] == a00)
            evec[1] = { (scalar)1, (scalar)0, (scalar)0 };
        else if (eVals[1] == a11)
            evec[1] = { (scalar)0, (scalar)1, (scalar)0 };
        else if (eVals[1] == a22)
            evec[1] = { (scalar)0, (scalar)0, (scalar)1 };

        if(eVals[2] == a00)
            evec[2] = { (scalar)1, (scalar)0, (scalar)0 };
        else if (eVals[2] == a11)
            evec[2] = { (scalar)0, (scalar)1, (scalar)0 };
        else if (eVals[2] == a22)
            evec[2] = { (scalar)0, (scalar)0, (scalar)1 };
        eval[0]=eVals[0];eval[1]=eVals[1];eval[2]=eVals[2];
    }

    // The preconditioning scaled the matrix A, which scales the eigenvalues.
    // Revert the scaling.
    eval[0] *= maxAbsElement;
    eval[1] *= maxAbsElement;
    eval[2] *= maxAbsElement;
    /*
    printf("\n\n %f %f %f\n %f %f %f \n %f %f %f \n\n\n",evec[0][0],evec[0][1],evec[0][2],
                                                         evec[1][0],evec[1][1],evec[1][2],
                                                         evec[2][0],evec[2][1],evec[2][2]);
    */
}

std::array<scalar, 3> NISymmetricEigensolver3x3::Multiply(
    scalar s, std::array<scalar, 3> const& U)
{
    std::array<scalar, 3> product = { s * U[0], s * U[1], s * U[2] };
    return product;
}

std::array<scalar, 3> NISymmetricEigensolver3x3::Subtract(
    std::array<scalar, 3> const& U, std::array<scalar, 3> const& V)
{
    std::array<scalar, 3> difference = { U[0] - V[0], U[1] - V[1], U[2] - V[2] };
    return difference;
}

std::array<scalar, 3> NISymmetricEigensolver3x3::Divide(
    std::array<scalar, 3> const& U, scalar s)
{
    scalar invS = (scalar)1 / s;
    std::array<scalar, 3> division = { U[0] * invS, U[1] * invS, U[2] * invS };
    return division;
}

scalar NISymmetricEigensolver3x3::Dot(std::array<scalar, 3> const& U,
    std::array<scalar, 3> const& V)
{
    scalar dot = U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
    return dot;
}

std::array<scalar, 3> NISymmetricEigensolver3x3::Cross(std::array<scalar, 3> const& U,
    std::array<scalar, 3> const& V)
{
    std::array<scalar, 3> cross =
    {
        U[1] * V[2] - U[2] * V[1],
        U[2] * V[0] - U[0] * V[2],
        U[0] * V[1] - U[1] * V[0]
    };
    return cross;
}

void NISymmetricEigensolver3x3::ComputeOrthogonalComplement(
    std::array<scalar, 3> const& W, std::array<scalar, 3>& U, std::array<scalar, 3>& V) const
{
    // Robustly compute a right-handed orthonormal set { U, V, W }.  The
    // vector W is guaranteed to be unit-length, in which case there is no
    // need to worry about a division by zero when computing invLength.
    scalar invLength;
    if (std::abs(W[0]) > std::abs(W[1]))
    {
        // The component of maximum absolute value is either W[0] or W[2].
        invLength = (scalar)1 / std::sqrt(W[0] * W[0] + W[2] * W[2]);
        U = { -W[2] * invLength, (scalar)0, +W[0] * invLength };
    }
    else
    {
        // The component of maximum absolute value is either W[1] or W[2].
        invLength = (scalar)1 / std::sqrt(W[1] * W[1] + W[2] * W[2]);
        U = { (scalar)0, +W[2] * invLength, -W[1] * invLength };
    }
    V = Cross(W, U);
}

void NISymmetricEigensolver3x3::ComputeEigenvector0(scalar a00, scalar a01,
    scalar a02, scalar a11, scalar a12, scalar a22, scalar eval0, std::array<scalar, 3>& evec0) const
{
    // Compute a unit-length eigenvector for eigenvalue[i0].  The matrix is
    // rank 2, so two of the rows are linearly independent.  For a robust
    // computation of the eigenvector, select the two rows whose cross product
    // has largest length of all pairs of rows.
    std::array<scalar, 3> row0 = { a00 - eval0, a01, a02 };
    std::array<scalar, 3> row1 = { a01, a11 - eval0, a12 };
    std::array<scalar, 3> row2 = { a02, a12, a22 - eval0 };
    std::array<scalar, 3>  r0xr1 = Cross(row0, row1);
    std::array<scalar, 3>  r0xr2 = Cross(row0, row2);
    std::array<scalar, 3>  r1xr2 = Cross(row1, row2);
    scalar d0 = Dot(r0xr1, r0xr1);
    scalar d1 = Dot(r0xr2, r0xr2);
    scalar d2 = Dot(r1xr2, r1xr2);

    scalar dmax = d0;
    int imax = 0;
    if (d1 > dmax)
    {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax)
    {
        imax = 2;
    }

    if (imax == 0)
    {
        evec0 = Divide(r0xr1, std::sqrt(d0));
    }
    else if (imax == 1)
    {
        evec0 = Divide(r0xr2, std::sqrt(d1));
    }
    else
    {
        evec0 = Divide(r1xr2, std::sqrt(d2));
    }
}

void NISymmetricEigensolver3x3::ComputeEigenvector1(scalar a00, scalar a01,
    scalar a02, scalar a11, scalar a12, scalar a22, std::array<scalar, 3> const& evec0,
    scalar eval1, std::array<scalar, 3>& evec1) const
{
    // Robustly compute a right-handed orthonormal set { U, V, evec0 }.
    std::array<scalar, 3> U, V;
    ComputeOrthogonalComplement(evec0, U, V);

    // Let e be eval1 and let E be a corresponding eigenvector which is a
    // solution to the linear system (A - e*I)*E = 0.  The matrix (A - e*I)
    // is 3x3, not invertible (so infinitely many solutions), and has rank 2
    // when eval1 and eval are different.  It has rank 1 when eval1 and eval2
    // are equal.  Numerically, it is difficult to compute robustly the rank
    // of a matrix.  Instead, the 3x3 linear system is reduced to a 2x2 system
    // as follows.  Define the 3x2 matrix J = [U V] whose columns are the U
    // and V computed previously.  Define the 2x1 vector X = J*E.  The 2x2
    // system is 0 = M * X = (J^T * (A - e*I) * J) * X where J^T is the
    // transpose of J and M = J^T * (A - e*I) * J is a 2x2 matrix.  The system
    // may be written as
    //     +-                        -++-  -+       +-  -+
    //     | U^T*A*U - e  U^T*A*V     || x0 | = e * | x0 |
    //     | V^T*A*U      V^T*A*V - e || x1 |       | x1 |
    //     +-                        -++   -+       +-  -+
    // where X has row entries x0 and x1.

    std::array<scalar, 3> AU =
    {
        a00 * U[0] + a01 * U[1] + a02 * U[2],
        a01 * U[0] + a11 * U[1] + a12 * U[2],
        a02 * U[0] + a12 * U[1] + a22 * U[2]
    };

    std::array<scalar, 3> AV =
    {
        a00 * V[0] + a01 * V[1] + a02 * V[2],
        a01 * V[0] + a11 * V[1] + a12 * V[2],
        a02 * V[0] + a12 * V[1] + a22 * V[2]
    };

    scalar m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
    scalar m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
    scalar m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;

    // For robustness, choose the largest-length row of M to compute the
    // eigenvector.  The 2-tuple of coefficients of U and V in the
    // assignments to eigenvector[1] lies on a circle, and U and V are
    // unit length and perpendicular, so eigenvector[1] is unit length
    // (within numerical tolerance).
    scalar absM00 = std::abs(m00);
    scalar absM01 = std::abs(m01);
    scalar absM11 = std::abs(m11);
    scalar maxAbsComp;
    if (absM00 >= absM11)
    {
        maxAbsComp = (absM00 > absM01) ? absM00 : absM01;
        if (maxAbsComp > (scalar)0)
        {
            if (absM00 >= absM01)
            {
                m01 /= m00;
                m00 = (scalar)1 / std::sqrt((scalar)1 + m01 * m01);
                m01 *= m00;
            }
            else
            {
                m00 /= m01;
                m01 = (scalar)1 / std::sqrt((scalar)1 + m00 * m00);
                m00 *= m01;
            }
            evec1 = Subtract(Multiply(m01, U), Multiply(m00, V));
        }
        else
        {
            evec1 = U;
        }
    }
    else
    {
        maxAbsComp = (absM11 > absM01) ? absM11 : absM01;
        if (maxAbsComp > (scalar)0)
        {
            if (absM11 >= absM01)
            {
                m01 /= m11;
                m11 = (scalar)1 / std::sqrt((scalar)1 + m01 * m01);
                m01 *= m11;
            }
            else
            {
                m11 /= m01;
                m01 = (scalar)1 / std::sqrt((scalar)1 + m11 * m11);
                m11 *= m01;
            }
            evec1 = Subtract(Multiply(m11, U), Multiply(m01, V));
        }
        else
        {
            evec1 = U;
        }
    }
}

#endif
