# Landau-de Gennes Theory

Here we briefly overview the Landau-de Gennes theoretical framework for nematic liquid crystals implemented in open-Qmin. For a fuller explanation with references, please see Section 2 of our [article in *Frontiers in Physics*](https://www.frontiersin.org/articles/10.3389/fphy.2019.00204/full#h3).

## The Q-tensor

Nematic liquid crystals are fluids with anisotropy, that is, properties that vary depending on direction. Nearby particles are oriented preferentially along a common direction, $\hat n$, called the nematic director. Mathematically, a director is like a unit vector but with an added "head-tail" symmetry by which $- \hat n$ represents the same physical state as $\hat n $. Because the director may vary from place to place in the liquid crystal, we speak of a director *field* $\hat n(\mathbf{r})$. 

Partly to account for this $\hat n = - \hat n $ symmetry, the nematic liquid crystal's orientational state is better described by a symmetric, second-rank tensor (which in $d$ dimensions is represented by a symmetric $d\times d$ matrix). This is the nematic orientation tensor, or Q-tensor, ${\bf Q}$. In open-Qmin, the state of the nematic liquid crystal is represented by a value of the Q-tensor at each lattice site. 
 
### From director to Q-tensor 

Because we are only interested in *anisotropic* properties of the liquid crystal, we subtract out the isotropic properties by making the Q-tensor traceless, $Q_{xx} + Q_{yy} + Q_{zz} = 0$. 

#### Uniaxial nematics 

The simplest relation between $\bf Q$ and $\hat n$ is given by 

$$ Q_{\alpha \beta} = \tfrac{3}{2} S \left(n_\alpha n_\beta - \tfrac{1}{3} \delta_{\alpha \beta}\right)  \quad \text{(uniaxial limit)}  $$

where $\delta$ is the Kronecker delta and the scalar prefactor $S$ is the *nematic degree of order*. 

The above relation is only true in the uniaxial limit, in which the particles of the nematic have no orientational preference among the directions in the plane perpendicular to $\hat n$. 

#### Biaxial nematics 

A more general expression for $\bf Q$ allows for *biaxial order* $S_B$ which favors one direction, $\hat m$, in the plane perpendicular to $\hat n $: 

$$ Q_{\alpha \beta} = \tfrac{3}{2} S \left( n_\alpha n_\beta - \tfrac{1}{3} \delta_{\alpha \beta} \right) + \tfrac{1}{2} S_B \left( m_\alpha m_\beta - l_\alpha l_\beta \right) .$$

Here $\hat l \equiv n \times \hat m$. 

### From Q-tensor to director 

For a given $Q_{\alpha \beta}$, the degree of nematic order $S$ is recovered as the greatest eigenvalue, and the director $\hat n $ is the corresponding eigenvector: 

$$ Q_{\alpha \beta} n_\beta = S n_\alpha $$ 

Because ${\bf Q}$ is traceless, the remaining two eigenvalues must sum to $-S$. The difference between these two eigenvalues becomes the degree of biaxial order $S_B$. ${\bf Q}$ is diagonal in the basis $\{ \hat n, \hat m, \hat l \}$, with diagonal elements 

$$ {\bf Q}^{\text{(diag)}}  = \mathrm{diag}\left(S, \tfrac{1}{2} (-S + S_B), \tfrac{1}{2} (-S-S_B) \right) $$ 

## Landau-de Gennes free energy

open-Qmin conducts numerical minimization of the Landau-de Gennes free energy, which is a functional of the Q-tensor. Schematically:

$$ {\mathcal F}[{\bf Q}] = \int_V \left( f_{\text{bulk}} + f_{\text{distortion}} + f_{\text{external}}\right) dv + \sum_{\alpha} \int_{S_\alpha} \left( f^\alpha_{\text{boundary}}\right)  ds. $$

The first integral is over the volume of the nematic; the second term is a sum of surface integrals over each boundary surface $S_\alpha$. The energy density terms are described below.

### Bulk free energy 

The bulk free energy density describes the isotropic-nematic phase transition in the Landau paradigm, using rotational invariants of ${\bf Q}$: 

$$ f_{\text{bulk}} = \tfrac{1}{2} A {\, \rm tr} ({\bf Q}^2) + \tfrac{1}{3} B {\,\rm tr} ({\bf Q}^3) + \tfrac{1}{4} C \left( {\rm tr}({\bf Q}^2) \right)^2 .$$ 

Here, $A$, $B$, and $C$ are material constants with $A$ depending on temperature $T$ as $A\propto (T-T^*_{NI})$, with $T^*_{NI}$ the temperature below which the isotropic phase is unstable. 

In the uniaxial limit, $f_{\text{bulk}}$ becomes a polynomial in the nematic degree of order, 

$$ f_{\text{bulk}} = \tfrac{3}{4} A S^2 + \tfrac{1}{4} B S^3 + \tfrac{9}{16} C S^4 , $$

which is minimized either by $S=0$ (isotropic phase) or by 

$$ S = S_0 \equiv \frac{ - B + \sqrt{ B^2 - 24 A C}}{6 C} \qquad \text{(preferred degree of order)} . $$ 

### Distortion free energy

The distortion (or "elastic") free energy density penalizes spatial gradients of ${\bf Q}$ as follows:

\begin{align*}
f_{\text{distortion}} &= \tfrac{1}{2} L_1 \frac{\partial Q_{ij}}{\partial x_k} \frac{\partial Q_{ij}}{\partial x_k}  + \tfrac{1}{2} L_2 \frac{\partial Q_{ij}}{\partial x_j} \frac{\partial Q_{ik}}{\partial x_k} + \tfrac{1}{2} L_3 \frac{\partial Q_{ik}}{\partial x_j} \frac{\partial Q_{ij}}{\partial x_k}  \\
& \quad + \tfrac{1}{2} L_4 \epsilon_{lik} Q_{lj} \frac{\partial Q_{ij}}{\partial x_k} + \tfrac{1}{2} L_6 Q_{lk} \frac{\partial Q_{ij}}{\partial x_l} \frac{\partial Q_{ij}}{\partial x_k} .
\end{align*}

Here, Einstein summation over repeated indices is implied, and $\epsilon$ is the Levi-Civita tensor. ***Please pay close attention to how we define the $L_i$ coefficients here, since other definitions are also commonly used in the literature.***



#### Relation to Frank elastic constants

In the uniaxial limit, $f_{\text{distortion}}$ maps onto the Frank-Oseen elastic free energy density,

\begin{align*} 
f_{\text{FO}} &= \tfrac{1}{2} \left\{ K_1 (\nabla \cdot \hat n )^2 + K_2 \left( \hat n \cdot (\nabla \times \hat n ) + q_0 \right)^2 \right. \\
& \qquad\;  \left. + K_3 \left| (\hat n \cdot \nabla) \hat n \right|^2 + K_{24} \nabla \cdot \left[ (\hat n \cdot \nabla ) \hat n - \hat n (\nabla \cdot \hat n ) \right] \right\},
\end{align*}

where $K_1$, $K_2$, $K_3$, and $K_{24}$ are respectively the splay, twist, bend, and saddle-splay elastic constants, and $q_0$ is the spontaneous chiral wavenumber in chiral nematics. The relations between these constants and the Landau-de Gennes distortion free energy density parameters in the uniaxial limit are:

\begin{align*}
L_1 &= \frac{2}{27 S^2} (K_3 - K_1 + 3 K_2) \\
L_2 &= \frac{4}{9 S^2} (K_1 - K_{24}) \\
L_3 &= \frac{4}{9 S^2} (K_{24} - K_2)  \\
L_4 &= -\frac{8}{9S^2} q_0 K_2 \\
L_6 &= \frac{4}{27 S^3} (K_3 - K_1) 
\end{align*}

Inverting these relations gives the following forms for the Frank-Oseen constants: 

\begin{align*}
K_1 &= \tfrac{9}{4} S^2 (2 L_1 + L_2 + L_3 - L_6 S) \\
K_2 &= \tfrac{9}{4} S^2 (2 L_1 - S L_6)  \\
K_3 &= \tfrac{9}{4} S^2 (2 L_1 + L_2 + L_3 + 2 S L_6)  \\ 
K_{24} &= -\tfrac{9}{4} S^2 (-2 L_1 - L_3 + S L_6)  \\
q_0 &= -\frac{L_4}{2(2L_1 - S L_6)}  
\end{align*}

#### One-constant approximation 

A common simplifying assumption for the Frank-Oseen free energy density is the "one-constant approximation", sometimes called "isotropic elasticity": 

$$ K_1 = K_2 = K_3 = K_{24} \equiv K$$ 

along with $q_0=0$. This assumption is a reasonable approach for many common molecular liquid crystals, where the elastic constants often have, at least, the same order of magnitude. Along with dramatically simplifying analytical approaches, the one-constant approximation also permits much faster computation with the corresponding choice of $L_i$ coefficients: 

$$ L_2 = L_3 = L_4 = L_6 = 0, $$ 
$$ L_1 = \frac{2}{9 S^2} K. $$ 

This leaves only one term in the distortion free energy density: 

$$ f_{\text{distortion}} ^{(1)} = \tfrac{1}{2} L_1 \frac{\partial Q_{ij}}{\partial x_k} \frac{\partial Q_{ij}}{\partial x_k} . $$

### External fields free energy 

External electric fields $\mathbf{E}$ and magnetic fields $\mathbf{H}$ contribute the following free energy densities, respectively, calculated at each site in the nematic:

$$ f_E = -\tfrac{1}{2} \varepsilon_0 E_i \varepsilon_{ij} E_j, \qquad \varepsilon_{ij} = \varepsilon \delta_{ij} + \Delta \varepsilon Q_{ij} $$

$$ f_H = - \tfrac{1}{2} \mu_0 H_i \chi_{ij} H_j, \qquad \chi_{ij} = \chi \delta_{ij} + \Delta \chi Q_{ij} $$ 

where

* $\varepsilon_0$ is the vacuum permittivity
* $\varepsilon_{ij}$ is the material's dielectric tensor
* $\varepsilon$ is the material's dielectric constant (relative permittivity)
* $\Delta \varepsilon$ is the material's dielectric anisotropy 
* $\mu_0$ is the magnetic permeability of free space
* $\chi_{ij}$ is the material's magnetic susceptibility tensor
* $\chi$ is the isotropic part of the material's magnetic susceptibility 
* $\Delta \chi$ is the anisotropic part of the material's magnetic susceptibility

and $\delta_{ij}$ is the Kronecker delta.

### Boundary free energy

At boundary surfaces, liquid crystals typically experience an anisotropic surface tension, called "anchoring", that depends on the relative orientations of the director $\hat n $ and a certain special direction $\hat \nu^\alpha$ picked out by a given point on the surface. (The superscript $\alpha$ indexes the different boundary surfaces.) 

open-Qmin has two categories of anchoring conditions: oriented and degenerate planar. 

#### Oriented (including homeotropic) anchoring

Oriented anchoring conditions penalize the director's deviations from a unique direction $\hat\nu^\alpha$ preferred by the surface. One common example is "homeotropic" anchoring, in which $\hat \nu^\alpha$ is the surface normal. Another is "oriented planar" ancohring, in which a particular direction in the surface's tangent plane serves as $\hat \nu^\alpha$. 

For oriented anchoring, the surface free energy density at boundary surface $\alpha$ takes the Nobili-Durand form: 

$$ f_{\text{boundary}}^\alpha  = W^\alpha_{\text{ND}}  (Q_{ij} - Q_{ij}^\alpha) (Q_{ij} - Q_{ij}^\alpha),  $$

where $W^\alpha_{\text{ND}} > 0$ is the anchoring strength of surface $\alpha$ and the surface-preferred Q-tensor is 

$$ Q^\alpha_{ij} = \tfrac{3}{2} S_0 (\nu^\alpha_i \nu^\alpha_j - \tfrac{1}{3} \delta_{ij} ) .$$ 


#### Degenerate planar anchoring

Surfaces with degenerate planar anchoring disfavor director orientation along the surface normal $\hat \nu^\alpha$, but exhibit no preference (i.e. a degeneracy) among the directions in the surface's  tangent plane. When modeling 3D nematics, it is *not* sufficient to use the above Nobili-Durand form with a negative anchoring strength. Instead, we use the Fournier-Galatola form: 

$$ f_{\text{boundary}}^\alpha = W^\alpha_{\text{FG}} ( \tilde Q_{ij} - \tilde Q_{ij}^\perp ) ( \tilde Q_{ij} - \tilde Q_{ij}^\perp ) , $$ 

where $ \tilde Q_{ij} = Q_{ij} + \tfrac{1}{2} S_0 \delta_{ij}$, and $\tilde Q_{ij}^\perp$ is the projection of $\tilde Q_{ij}$ onto the plane perpendicular to $\hat \nu^\alpha$,

$$ \tilde Q^\perp_{ij} = P_{ik} \tilde Q_{kl}P_{lj}, $$

using the projection operator $P_{ij} = \delta_{ij} - \nu^\alpha _i \nu^\alpha_j $ .

#### Anchoring strength relation to Rapini-Papoular form

In terms of the director, both oriented and degenerate planar anchoring are often modeled with the Rapini-Papoular free energy density,

$$ - \tfrac{1}{2} W^\alpha_{\text{RP}} (\hat \nu^\alpha \cdot \hat n )^2, $$

with $W_{\text{RP}}>0$ for oriented anchoring, and $<0$ for degenerate planar anchoring. 

Comparing to the Nobili-Durand and Fournier-Galatola forms, the anchoring strengths are related in both cases by 

$$ |W_{\text{RP}}^\alpha|  = 9 S_0^2 W^\alpha_{\text{ND,FG}}, $$ 

assuming ${\bf Q}$ is uniaxial with leading eigenvalue $S=S_0$. 