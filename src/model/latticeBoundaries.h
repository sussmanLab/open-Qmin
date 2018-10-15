#ifndef latticeBoundaries_H
#define latticeBoundaries_H

/*! \file latticeBoundaries.h */

//! Distinguish the potential types of non-liquid-crystalline lattice sites
enum class boundaryType {homeotropic,degeneratePlanar};

//! "boundary objects" have a boundary type and few parameters that will be used in force/energy calculations
/*!
For homeotropic anchoring (Rapini-Papoular) with
F_{anchoring} = W_0  tr((Q-Q^B)^2)
P1 is W_0.

For degenerate planar anchoring (Fournier and Galatola) with
F_{anchoring} = W_1 tr((\tilde{Q} - \tilde{Q}^\perp)^2) + W_1 (tr(\tilde{Q}^2) - (3 s_0/2)^2)^2
P1 is W_1 and P2 is S_0 (at the moment we use the W_2 = W_1 simplification)
*/
class boundaryObject
    {
    public:
        boundaryObject(boundaryType _boundary, scalar _p1, scalar _p2)
            {
            boundary= _boundary;
            P1 = _p1;
            P2 = _p2;
            }

        //mutating operators
        boundaryObject& operator=(const boundaryObject &other)
                {
                this->boundary = other.boundary;
                this->P1 = other.P1;
                this->P2 = other.P2;
                return *this;
                }
        boundaryType boundary;
        scalar P1, P2 ;
    };

#endif
