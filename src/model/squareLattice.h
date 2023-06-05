#ifndef cubicLattice_H
#define cubicLattice_H

#include "simpleModel.h"
#include "indexer.h"
#include "latticeBoundaries.h"
#include "kernelTuner.h"

/*! \file cubicLattice.h
\brief puts degrees of freedom on a cubic lattice... probably for spin-like models
*/

//!define a type of simple model which places all degrees of freedom (which are still d-dimensional) on a cubic lattice with nearest neighbor interactions
class cubicLattice : public simpleModel
    {
    public:
        //!The base constructor takes the number of lattice sites along the cubic edge
        cubicLattice(int l, bool _slice = false,bool _useGPU = false, bool _neverGPU = false);

        //!A rectilinear set of lattice sits
        cubicLattice(int lx, int ly, int lz, bool _slice = false,bool _useGPU = false, bool _neverGPU = false);

        //!move the degrees of freedom
        virtual void moveParticles(GPUArray<dVec> &displacements,scalar scale = 1.);
        //!move a different GPU array according to the same rules
        virtual void moveParticles(GPUArray<dVec> &dofs,GPUArray<dVec> &displacements,scalar scale = 1.);

        //!initialize each d.o.f. to be a unit spin on the sphere
        void setSpinsRandomly(noiseSource &noise);

        //! return the integer corresponding to the given site, along with the indices of the six nearest neighbors
        virtual int getNeighbors(int target, vector<int> &neighbors, int &neighs, int stencilType = 0);
        //!decide to slice sites
        void sliceIndices(bool _s=true){sliceSites = _s;};
        //!given a triple, determine what
        int latticeSiteToLinearIndex(const int3 &target);
        //!indexer for lattice sites
        Index3D latticeIndex;
        int3 latticeSites;

        //!indexer for neighbors
        Index2D neighborIndex;
        //!List of neighboring lattice sites
        GPUArray<int> neighboringSites;
        //!store the neighbors of each lattice site. The i'th neighbor of site j is given by neighboringSites[neighborIndex(i,j)]
        virtual void fillNeighborLists(int stencilType = 0);

        //!return the mean spin
        virtual dVec averagePosition()
            {
            dVec ans(0.0);
            ArrayHandle<dVec> spins(positions);
            ArrayHandle<int> t(types);
            int nSites=0;
            for(int i = 0; i < N; ++i)
                if(t.data[i] <= 0)
                    {
                    nSites+=1;
                    ans += spins.data[i];
                    };
            ans = (1.0/nSites)*ans;
            return ans;
        };
        virtual int positionToIndex(int px, int py, int pz)
            {
            int3 temp; temp.x = px; temp.y = py; temp.z=pz;
            return positionToIndex(temp);
            };
        virtual int positionToIndex(int3 &pos){UNWRITTENCODE("position to index in cubicLattice... currently this function exists only for multirankQTensor models");};

        //!Displace a boundary object (and surface sites) by one of the six primitive cubic lattice directions
        virtual void displaceBoundaryObject(int objectIndex, int motionDirection, int magnitude);

        //!assign a collection of lattice sites to a new boundaryObject
        void createBoundaryObject(vector<int> &latticeSites, boundaryType _type, scalar Param1, scalar Param2);

        //!list of the non-bulk objects in the simulations
        GPUArray<boundaryObject> boundaries;
        //!A vector that keeps track of the sites associated with each boundary object
        vector<GPUArray<int> > boundarySites;
        //!A vector that keeps track of the surface sites associated with each boundary object
        vector<GPUArray<int> > surfaceSites;
        //!A vector of flags that specifies the state of each boundary object...most schemes will be 0 = fixed boundary, 1 = movable boudnary
        vector<int> boundaryState;
        //!The force (from integrating the stress tensor) on each object
        vector<scalar3> boundaryForce;
        //!An assist vector that can keep track of changes to boundary sites during a move. First element is the index a Qtensor (second ) will move to
        GPUArray<pair<int,dVec> > boundaryMoveAssist1;
        //!An assist vector that can keep track of changes to surface sites during a move. First element is the index a Qtensor (second ) will move to
        GPUArray<pair<int,dVec> > boundaryMoveAssist2;

        virtual scalar getClassSize()
            {
            int bsSites = 0;
            for (int bb = 0; bb < boundarySites.size(); ++bb)
                bsSites += boundarySites[bb].getNumElements() + surfaceSites[bb].getNumElements();
            return 0.000000001*(2*sizeof(bool) +
            (1+neighboringSites.getNumElements()+bsSites+boundaryState.size()+boundaryMoveAssist1.getNumElements()+boundaryMoveAssist2.getNumElements())*sizeof(int) +
            sizeof(kernelTuner) + sizeof(boundaryObject) +
            2*sizeof(scalar)*DIMENSION*(boundaryMoveAssist2.getNumElements()+boundaryMoveAssist1.getNumElements()) +
            sizeof(Index2D) + sizeof(Index3D)) +
            simpleModel::getClassSize();
            }
    protected:
        //!a utility function for initialization
        void initializeNSites();
        //! should we use a memory-efficient slicing scheme?
        bool sliceSites;

        //!lattice sites per edge
        int L;

        //!normalize vector length when moving spins?
        bool normalizeSpins;

        //!performance for the moveParticles kernel
        shared_ptr<kernelTuner> moveParticlesTuner;
    };
#endif
