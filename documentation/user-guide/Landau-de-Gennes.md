# Landau-de Gennes Theory

## The Q-tensor

Nematic liquid crystals are fluids with a certain kind of anisotropy, that is, properties that vary depending on direction. Nearby particles are oriented preferentially along a common direction, $\hat n$, called the nematic director. Mathematically, a director is like a unit vector but with an added "head-tail" symmetry by which $- \hat n$ represents the same physical state as $\hat n $. Because the director may vary from place to place in the liquid crystal, we speak of a director *field* $\hat n(\mathbf{r})$. 

Partly to account for this $\hat n = - \hat n $ symmetry, the nematic liquid crystal's orientational state is better described by a symmetric, second-rank tensor (which in $d$ dimensions is represented by a symmetric $d\times d$ matrix). This is the nematic orientation tensor, or Q-tensor, ${\bf Q}$. In openQmin, the state of the nematic liquid crystal is represented by a value of the Q-tensor at each lattice site. 
 

Because we are only interested in *anisotropic* properties of the liquid crystal, we subtract out the isotropic properties by making the Q-tensor traceless, $Q_{xx} + Q_{yy} + Q_{zz} = 0$. The simplest relation between $\bf Q$ and $\hat n$ is given by 

$$ Q_{\alpha \beta} = \tfrac{3}{2} S \left(n_\alpha n_\beta - \tfrac{1}{3} \delta_{\alpha \beta}\right)  \quad \text{(uniaxial limit)}  $$

where $\delta$ is the Kronecker delta and the scalar prefactor $S$ is the *nematic degree of order*. 

The above relation is only true in the uniaxial limit, in which the particles of the nematic have no orientational preference among the directions in the plane perpendicular to $\hat n$. However, the more general expression for $\bf Q$ allows for *biaxial order* $S_B$ which favors one direction, $\hat m$, in the plane perpendicular to $\hat n $: 

$$ Q_{\alpha \beta} = \tfrac{3}{2} S \left( n_\alpha n_\beta - \tfrac{1}{3} \delta_{\alpha \beta} \right) + \tfrac{1}{2} S_B \left( m_\alpha m_\beta - l_\alpha l_\beta \right) $$

Here $\hat l \equiv n \times \hat m$. 


## Phenomenological free energy density

### Bulk free energy 

### Distortion free energy

#### Relation to Frank elastic constants


### External fields free energy 

### Boundary free energy

#### Homeotropic (and other oriented) anchoring 

#### Degenerate planar anchoring

