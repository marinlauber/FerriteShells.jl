
### Discretization

Substituting the discrteized variation of the covariant basis vector
$\delta\mathbf{a}_\alpha=\frac{\partial\delta\mathbf{x}}{\partial\xi^\alpha} = \frac{\partial\delta(\mathbf{X}+\mathbf{u})}{\partial\xi^\alpha} = \frac{\partial \delta\mathbf{u}}{\partial\xi^\alpha}\simeq\sum_I\delta\mathbf{u}_I\frac{\partial N_I}{\partial\xi^\alpha}
$ into the variation of the internal energy, we get
```math
\delta\mathcal{W}_\text{int} = \int_\omega \mathbb{N}^{\alpha\beta}\left(\sum_I\delta\mathbf{u}_I\frac{\partial N_I}{\partial\xi^\alpha}\cdot\mathbf{a}_\beta\right) \, \sqrt{a}\,\mathrm{d}y = \sum_I\delta\mathbf{u}_I\cdot\int_\omega\frac{\partial N_I}{\partial\xi^\alpha}\mathbb{N}^{\alpha\beta}\mathbf{a}_\beta \, \sqrt{a}\,\mathrm{d}y
```
where the last part is the residual contribution of the membrane term.

From the residual equation, we can derive the consistent tangent by taking the directional derivative of the residual with respect to the displacement increment, which gives us the following expression for the tangent stiffness matrix
```math
\begin{split}
\mathbb{K}_{IJ} &= \frac{\partial}{\partial\mathbf{u}_J}\int_\omega\frac{\partial N_I}{\partial\xi^\alpha}\mathbb{N}^{\alpha\beta}\mathbf{a}_\beta \, \sqrt{a}\,\mathrm{d}y \\
&= \int_\omega\frac{\partial N_I}{\partial\xi^\alpha}\left(\frac{\partial\mathbb{N}^{\alpha\beta}}{\partial\mathbf{u}_J}\mathbf{a}_\beta + \mathbb{N}^{\alpha\beta}\frac{\partial\mathbf{a}_\beta}{\partial\mathbf{u}_J}\right) \, \sqrt{a}\,\mathrm{d}y
\end{split}
```

---
## KL Bending Residual: Step-by-Step Derivation

Setup

The bending energy is:
```math
W_\text{bend} = \frac{1}{2} \int \kappa_{\alpha\beta} D^{\alpha\beta\gamma\delta} \kappa_{\gamma\delta} d\Omega
```

with $\kappa_{\alpha\beta} = b_{\alpha\beta} - B_{\alpha\beta}$ and the second fundamental form $b_{\alpha\beta} =  a_{\alpha\beta} \cdot n$, where $a_{\alpha\beta} = \partial^2\varphi / \partial\xi^\alpha \partial\xi^\beta$ and $n = (a_1 \times a_2)/|a_1 \times a_2|$.

---
Step 1 — Virtual work

Taking the Gâteaux derivative:
```math
\delta W = M^{\alpha\beta} , \delta\kappa_{\alpha\beta} = M^{\alpha\beta} , \delta b_{\alpha\beta}
```
where $M^{\alpha\beta} = D^{\alpha\beta\gamma\delta} \kappa_{\gamma\delta}$ is the bending moment resultant. The reference curvature $B_{\alpha\beta}$ is fixed, so $\delta\kappa_{\alpha\beta} = \delta b_{\alpha\beta}$.

---
Step 2 — Variation of $b_{\alpha\beta}$

Applying the product rule to $b_{\alpha\beta} = a_{\alpha\beta} \cdot n$:
```math
\delta b_{\alpha\beta} = \underbrace{\delta a_{\alpha\beta} \cdot n}_{\text{Term 1}} + \underbrace{a{\alpha\beta} \cdot  \delta n}_{\text{Term 2}}
```
Term 1 is straightforward. For virtual displacement $\delta u_I$ at node $I$:
```math
\delta a_{\alpha\beta} = \frac{\partial^2 N_I}{\partial\xi^\alpha \partial\xi^\beta} \delta u_I \implies \delta a_{\alpha\beta} \cdot n = \frac{\partial^2 N_I}{\partial\xi^\alpha \partial\xi^\beta} (n \cdot \delta u_I)
```

---
Step 3 — Variation of $n$ (Term 2)

The unit normal satisfies two constraints: $n \cdot n = 1$ and $n \cdot a_\alpha = 0$. Varying both:
$$n \cdot \delta n = 0 \implies \delta n \perp n \implies \delta n = \lambda^\mu a_\mu \quad \text{(lies in tangent plane)}$$
```math
\delta n \cdot a_\alpha = -n \cdot \delta a_\alpha = -\frac{\partial N_I}{\partial\xi^\alpha}(n \cdot \delta u_I)
```

Substituting $\delta n = \lambda^\mu a_\mu$ into the second constraint:
$$\lambda^\mu a_{\mu\alpha} = -\frac{\partial N_I}{\partial\xi^\alpha}(n \cdot \delta u_I)$$
This is a $2\times2$ linear system for $\lambda^\mu$. Multiplying by the contravariant metric $a^{\alpha\gamma}$ and summing over $\alpha$:
```math\lambda^\gamma = -(n \cdot \delta u_I) , a^{\alpha\gamma} \frac{\partial N_I}{\partial\xi^\alpha}
```
So:
```math
\delta n = -(n \cdot \delta u_I) , a^{\alpha\gamma} \frac{\partial N_I}{\partial\xi^\alpha} , a_\gamma
```
---
Step 4 — Evaluate Term 2
```math
a_{\alpha\beta} \cdot \delta n = -(n \cdot \delta u_I) \sum_{\mu,\gamma} a^{\mu\gamma} \frac{\partial N_I}{\partial\xi^\mu}
(a_{\alpha\beta} \cdot a_\gamma)
```
Recognising the Christoffel symbols of the current metric:
```math
\Gamma^\mu_{\alpha\beta} = \sum_\gamma a^{\mu\gamma} (a_{\alpha\beta} \cdot a_\gamma)
```

Term 2 becomes:
```math
a_{\alpha\beta} \cdot \delta n = -(n \cdot \delta u_I) \sum_\mu \Gamma^\mu_{\alpha\beta} \frac{\partial N_I}{\partial\xi^\mu}
```
Step 5 — Combine
```math
\delta b_{\alpha\beta} = \left(\frac{\partial^2 N_I}{\partial\xi^\alpha\partial\xi^\beta} - \Gamma^\mu_{\alpha\beta}
\frac{\partial N_I}{\partial\xi^\mu}\right)(n \cdot \delta u_I)
```
This is a scalar times $(n \cdot \delta u_I)$: bending only does work through the normal component of the displacement, which is the hallmark of the Kirchhoff constraint. Define the covariant curvature shape function:
```math
h_I^{\alpha\beta} = \frac{\partial^2 N_I}{\partial\xi^\alpha\partial\xi^\beta} - \Gamma^\mu_{\alpha\beta} \frac{\partial
N_I}{\partial\xi^\mu}
```
---
Step 6 — Residual
```math
\delta W = M^{\alpha\beta} , h_I^{\alpha\beta} , (n \cdot \delta u_I) = \left[M^{\alpha\beta} , h_I^{\alpha\beta}\right] n
\cdot \delta u_I = r_I \cdot \delta u_I
```
```math
\boxed{r_I = \left[M^{11} h_I^{11} + 2 M^{12} h_I^{12} + M^{22} h_I^{22}\right] n}
```
(factor 2 from $M$ and $h$ both symmetric). The three scalar components are:
```math
h_I^{11} = \partial^2_{11} N_I - \Gamma^1_{11} \partial_1 N_I - \Gamma^2_{11} \partial_2 N_I
```
```math
h_I^{12} = \partial^2_{12} N_I - \Gamma^1_{12} \partial_1 N_I - \Gamma^2_{12} \partial_2 N_I
```
```math
h_I^{22} = \partial^2_{22} N_I - \Gamma^1_{22} \partial_1 N_I - \Gamma^2_{22} \partial_2 N_I
```

with Christoffel symbols $\Gamma^\gamma_{\alpha\beta} = a^{\gamma\mu}(a_{\alpha\beta} \cdot a_\mu)$ (all quantities from the current configuration).
---
### Simplification: flat reference, small rotations

For a flat reference ($B_{\alpha\beta}=0$, orthonormal $\xi$-parametrisation) and moderate deformations, $a_{\alpha\beta} = \Sigma_I \partial^2_{\alpha\beta} N_I , u_I$ is first-order in $u$, so $\Gamma^\gamma_{\alpha\beta} \sim O(u)$ and the term $\Gamma^\gamma_{\alpha\beta} \partial_\gamma N_I \sim O(u)$. Dropping this gives the linearised approximation $h_I^{\alpha\beta} \approx \partial^2_{\alpha\beta} N_I$, which is what you'd use for a linear plate. For a geometrically nonlinear flat shell the Christoffel terms must be retained.

---
This is what the ForwardDiff.gradient on bending_energy_KL computes exactly. The explicit residual adds the Christoffel
correction that isn't visible in the energy-based form but is essential for large-rotation accuracy.


---

### 2.3 Reissner-Mindlin / Naghdi shell
The Reissner-Mindlin kinematic assumption
```math
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \mathbf{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)
```
