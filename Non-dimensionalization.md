# Non-dimensionalization 

As with any computer simulation of a physical system, dimensionful quantities are represented numerically by non-dimensionalized counterparts, so it's important to rescale quantities consistently. 

Within open-Qmin, all energy densities are scaled in units of the magnitude of the first Landau-de Gennes bulk free energy coefficient:

$$ \tilde f = \frac{f}{|A|}. $$

This applies to the LdG bulk free energy coefficients themselves: 

$$ \tilde A = -1, \quad \tilde B = B/|A|, \quad \tilde C = C/|A|. $$ 

You may choose to provide a value of $\tilde A$ besides $-1$ using the `--phaseConstantA` or `-a` flag; indeed the default value is $-0.172$. However, before beginning computation, open-Qmin rescales the entered $B$ and $C$ values by $|A|$, so `-a <A_value> -b <B_value> -c <C_value>` is equivalent to `-a -1 -b <B_value/|A_value|> -c <C_value/|A_value|>`. No such automatic rescaling is performed for other values such as the elastic constants. 

The elastic constants, which have dimensions of $ (\text{energy density}) \times (\text{length})^2 $, are non-dimensionalized using $A$ and the lattice spacing $\Delta x$:

$$ \tilde L_i = \frac{L_i}{|A| \Delta x^2} . $$

Thus, in the one-elastic-constant approximation, for a nematic material with a given $L_1$ and $|A|$, the chosen value for $\tilde L_1$ sets the dimensionful length that corresponds to the lattice spacing:

$$\Delta x = \sqrt{\frac{L_1}{\tilde L_1 |A|}} . $$ 

Generally, the dimensionful $\Delta x$ should be slightly smaller than the material's defect core size; larger values make it challenging to resolve defect cores in a finite difference approach. 

For external fields, the dimensionful products $\mu_0 |\bf H|^2 $ and $\varepsilon_0 |\bf E|^2$ have units of energy density, so the non-dimensionalized versions of these products must be given in units of $|A|$. One option is to set $\mu_0=1$ and thus define your non-dimensionalized magnetic field  as $\tilde {\bf H} = {\bf H} \sqrt{\mu_0 / |A|} $, (and likewise for $\tilde {\bf E} = {\bf E}\sqrt{\varepsilon_0/|A|}$).

