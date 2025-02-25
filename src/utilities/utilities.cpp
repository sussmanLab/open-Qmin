#include "utilities.h"
#include "std_include.h"

scalar host_dVec_dot_products(dVec *input1,dVec *input2,int N)
    {
    scalar ans = 0.0;
    for (int ii = 0; ii < N; ++ii)
        for (int dd = 0; dd < DIMENSION; ++dd)
            ans +=input1[ii][dd]*input2[ii][dd];
    return ans;
    }

void host_dVec_plusEqual_dVec(dVec *d_vec1,dVec *d_vec2,scalar factor,int N)
    {
    for (int ii = 0; ii < N; ++ii)
        d_vec1[ii] = d_vec1[ii] + factor*d_vec2[ii];
    }

void host_dVec_times_scalar(dVec *d_vec1, scalar factor, dVec *d_ans, int N)
    {
    for(int ii = 0; ii < N; ++ii)
        d_ans[ii] = factor*d_vec1[ii];
    }
