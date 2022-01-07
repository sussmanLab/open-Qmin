import numpy as np

class Scene():
    """ Class to contain a collection of BoundaryObjects and write the boundaryFile """
    def __init__(self, Lx, Ly, Lz):
        self.dims = (Lx, Ly, Lz)
        self.boundary_objects = []
                
    def to_file(self, filename):
        """ write the boundaryFile """
        with open(filename, 'w') as f:
            f.write(str(len(self.boundary_objects)) + '\n')
            for bdy_obj in self.boundary_objects:
                f.write(bdy_obj.for_file(self.dims))
                
class AnchoringCondition():
    def __init__(self, b_type=0, strength=1., S0=0.53):
        self.b_type = b_type # 0 for oriented, 1 for degenerate planar 
        self.strength = strength
        self.S0 = S0
        
class OrientedAnchoringCondition(AnchoringCondition):
    def __init__(self, strength=1., S0=0.53):
        super().__init__(b_type=0, strength=strength, S0=S0)

class DegeneratePlanarAnchoringCondition(AnchoringCondition):
    def __init__(self, strength=1., S0=0.53):
        super().__init__(b_type=1, strength=strength, S0=S0)
        
class BoundaryObject():
    """ Any set of sites sharing the same AnchoringCondition can be a BoundaryObject """
    def __init__(
        self, anch_cond,
        member_func = (lambda X, Y, Z: False), # True iff site is in (or on) boundary
        normal_func = (lambda X, Y, Z: positions) # Surface normal pointing toward nematic
    ):
        self.anch_cond = anch_cond
        self.b_type = anch_cond.b_type
        self.strength = anch_cond.strength
        self.S0 = anch_cond.S0        
        self.positions = np.empty((0,3))
        self.Q = np.empty((0,5))
        self.member_func = member_func
        self.normal_func = normal_func
        
    def for_file(self, dims):
        """ the Scene will call this and provide the dimensions 'dims' """
        self.get_bdy_data(dims)
        n_entries = len(self.positions)           
        bdy_obj_string = f'{int(self.b_type)} {self.strength} {self.S0} {n_entries}' + '\n'
        bdy_obj_string += self.create_file_string()
        return bdy_obj_string 
    
    def create_file_string(self):
        """ generate string for this object's contribution to the boundaryFile """
        ret = ''
        Q5 = self.Q.reshape(-1,9)[:,[0,1,2,4,5]]
        for (pt, qline) in zip(self.positions, Q5):
            ret += ' '.join([str(int(item)) for item in pt])
            ret += ' '
            ret += ' '.join([str(item) for item in qline])
            ret += '\n'
        return ret 
        
    def get_bdy_data(self, dims):
        """ calculate values for boundaryFile from AnchoringCondition, member_func, normal_func, and Scene's dims """
        Lx, Ly, Lz = dims
        # grid of coordinates
        Z, Y, X = np.meshgrid(np.arange(Lz), np.arange(Ly), np.arange(Lx))
        # flattened list of xyz coordinates
        all_positions = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        # indices of boundary sites in list of coordinates
        bdy_idxs = np.argwhere(
            self.member_func(X,Y,Z).flatten()                
        ).flatten()
        # coordinates of boundary sites
        self.positions = np.asarray(all_positions[bdy_idxs], dtype=float)
        
        # outward normals at boundary sites
        self.normals = np.array(self.normal_func(*self.v3_to_XYZ(self.positions))).T
        normals_norm = np.linalg.norm(self.normals, axis=-1) # calculate norm 
        normals_norm[normals_norm==0] = 1 # avoid dividing by zero
        self.normals /= np.stack((normals_norm,)*3, axis=-1) # normalize 
        
        # convert to information about preferred Q-tensor
        if self.b_type == 0:
            # for oriented anchoring, the Q-tensor is the target Q-tensor
            self.Q = self.calculate_uniaxial_Q()
        else:
            # for degenerate planar anchoring, the "Q_tensor" variable takes a simpler but more specialized form
            self.Q = np.zeros((len(self.positions),3,3))
            self.Q[:,0,:] = self.normals
        self.Q[np.abs(self.Q) < 1e-10] = 0 # mimic Mathematica's Chop        

    def v3_to_XYZ(self, v3):
        """ convert Nx3 array to 3 1D arrays of length N """
        return (v3[:,0], v3[:,1], v3[:,2])

    def calculate_uniaxial_Q(self):
        """ Q-tensor from director and degree of order """
        n = self.normals
        S = self.S0
        return 3*S/2 * (np.einsum("...i,...j->...ij", n, n) - np.eye(3)/3)

    def object_Q_value(boundaryType, S0, direction):
        """ Convert boundaryObject attributes to information about preferred Q-tensor """
        vec = np.asarray(direction, dtype=float)
        vec_norm = np.linalg.norm(vec, axis=-1) # calculate norm 
        vec_norm[vec_norm==0] = 1 # avoid dividing by zero
        vec /= np.stack((vec_norm,)*3, axis=-1) # normalize 
        if boundaryType == 0:
            # for oriented anchoring, the Q-tensor is the target Q-tensor
            q = Q(vec, S0)
        else:
            # for degenerate planar anchoring, the "Q-tensor" variable takes a simpler but more specialized form
            q = np.zeros((3,3))
            q[0,:] = vec
        q[np.abs(q) < 1e-10] = 0 # mimic Mathematica's Chop
        return q

class SphericalColloid(BoundaryObject):
    """ Spherical inclusion within nematic """
    def __init__(self, anch_cond, center, radius):
        self.center = center
        self.radius = radius
        super().__init__(
            anch_cond,
            member_func = self.sphere_member_func,
            normal_func = self.sphere_normal_func
        )
        
    def sphere_member_func(self, X, Y, Z):
        Cx, Cy, Cz = self.center
        return ((X-Cx)**2 + (Y-Cy)**2 + (Z-Cz)**2 <= self.radius**2)
    
    def sphere_normal_func(self, X, Y, Z):
        Cx, Cy, Cz = self.center        
        return (X-Cx, Y-Cy, Z-Cz)
    
class SphericalDroplet(BoundaryObject):
    """ Spherical droplet containing nematic; has radius R and is centered at (x,y,z)=(R+1,R+1,R+1) """
    def __init__(self, anch_cond, radius):        
        self.radius = radius        
        self.center =  np.array((1+self.radius,)*3)
        super().__init__(
            anch_cond,
            member_func = self.droplet_member_func,
            normal_func = self.droplet_normal_func
        )
        
    def droplet_member_func(self, X, Y, Z):
        Cx, Cy, Cz = self.center
        return ((X-Cx)**2 + (Y-Cy)**2 + (Z-Cz)**2 >= self.radius**2)
    
    def droplet_normal_func(self, X, Y, Z):
        Cx, Cy, Cz = self.center        
        return (Cx-X, Cy-Y, Cz-Z)
    
class Wall(BoundaryObject):
    """ Planar boundary with nematic on *both* sides. """
    def __init__(self, anch_cond, normal, height):
        if normal == "x":
            normal = [1,0,0]
        elif normal == "y":
            normal = [0,1,0]
        elif normal == "z":
            normal = [0,0,1]
        normal = np.array(normal)
        self.normal = normal / np.linalg.norm(normal)
        self.height = height # plane contains point (self.height * self.normal) relative to [0,0,0]
        super().__init__(
            anch_cond,
            member_func = self.wall_member_func,
            normal_func = self.wall_normal_func
        )        
        
    def wall_member_func(self, X, Y, Z):
        Nx, Ny, Nz = self.normal
        position_dot_normal = X*Nx + Y*Ny + Z*Nz 
        return (position_dot_normal < self.height+1) * (position_dot_normal > self.height-1)
    
    def wall_normal_func(self, X, Y, Z):
        Nx, Ny, Nz = self.normal
        ones = np.ones_like(X)
        return (Nx*ones, Ny*ones, Nz*ones)
    
class CylindricalCapillary(BoundaryObject):
    def __init__(self, anch_cond, radius):
        self.radius = radius
        self.base_center = np.array([1+self.radius, 1+self.radius, 0])
        super().__init__(
            anch_cond,
            member_func = self.capillary_member_func,
            normal_func = self.capillary_normal_func
        )
        
    def capillary_member_func(self, X, Y, Z):
        Cx, Cy, _ = self.base_center
        return ((X-Cx)**2 + (Y-Cy)**2 >= self.radius**2)
    
    def capillary_normal_func(self, X, Y, Z):
        Cx, Cy, _ = self.base_center        
        return (Cx-X, Cy-Y, 0*Z)
        
