#ifndef UTILITIES_H
#define UTILITIES_H

#include "std_include.h"

//!Take two vectors of dVecs and compute the sum of the dot products between them on the host
scalar host_dVec_dot_products(dVec *input1,dVec *input2,int N);

//! vec1 += a*vec2... on the host!
void host_dVec_plusEqual_dVec(dVec *d_vec1,dVec *d_vec2,scalar factor,int N);

//! (dVec) ans = input * factor... on the host
void host_dVec_times_scalar(dVec *d_vec1,
                              scalar factor,
                              dVec *d_ans,
                              int N);
#endif
