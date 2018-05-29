#ifndef cubicLattice_H
#define cubicLattice_H

#include "simpleModel.h"
#include "indexer.h"
/*! \file cubicLattic.h
\brief puts degrees of freedom on a cubic lattice... probably for spin-like models
*/

//!define a type of simple model which places all degrees of freedom (which are still d-dimensional) on a cubic lattice with nearest neighbor interactions
class cubicLattice : public simpleModel
    {
    public:
        //!The base constructor takes the number of lattice sites along the cubic edge
        cubicLattice(int l, bool _slice = false,bool _useGPU = false);

        //!move the degrees of freedom
        virtual void moveParticles(GPUArray<dVec> &displacements,scalar scale = 1.);

        //!initialize each d.o.f. to be a unit spin on the sphere
        void setSpinsRandomly(noiseSource &noise);

        //! return the integer corresponding to the given site, along with the indices of the six nearest neighbors
        int getNeighbors(int target, vector<int> &neighbors, int &neighs);
        //!decide to slice sites
        void sliceIndices(bool _s=true){sliceSites = _s;};
        //!given a triple, determine what 
        int latticeSiteToLinearIndex(const int3 &target);
        //!indexer for lattice sites
        Index3D latticeIndex;

        //!return the mean spin
        virtual dVec averagePosition()
            {
            dVec ans(0.0);
            ArrayHandle<dVec> spins(positions);
            for(int i = 0; i < N; ++i)
                ans += spins.data[i];
            ans = (1.0/N)*ans;
            return ans;
            }
    protected:
        //! should we use a memory-efficient slicing scheme?
        bool sliceSites;

        //!lattice sites per edge
        int L;

        //!normalize vector length when moving spins?
        bool normalizeSpins;
    };
#endif
