# Kirchhoff-Love / Koiter shell

## 1. Kinematics

![Shell kinematics](images/shell_kinematic.png)

The Kirchhoff-Love kinematic assumption prevents transverse shear strain by constraining the cross-section to remain normal to the shell's midsurface during deformation.
```math
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\hat{\mathbf{a}}_3(\xi^1,\xi^2)
```
where ``\hat{\mathbf{a}}_3`` is the unit normal to the midsurface. The surface basis vector are given by
```math
\begin{split}
\mathbf{g}_\alpha &= \frac{\partial \Phi(\xi^1,\xi^2)}{\partial \xi^\alpha} = \frac{\partial}{\partial\xi^\alpha}\left[\phi(\xi^1,\xi^2) + \xi^3\hat{\mathbf{a}}_3(\xi^1,\xi^2)\right]\\
&= \mathbf{a}_\alpha + \xi^3 \hat{\mathbf{a}}_{3,\alpha} \\
\mathbf{g}_3 &= \frac{\partial}{\partial\xi^3}\left[\phi(\xi^1,\xi^2) + \xi^3\hat{\mathbf{a}}_3(\xi^1,\xi^2)\right] =\hat{\mathbf{a}}_3
\end{split}
```
using the definition of ``\mathbf{a}_\alpha``. From this, we can get the components of the metric tensor
```math
\begin{split}
g_{\alpha\beta} &= \mathbf{g}_\alpha \cdot \mathbf{g}_\beta = (\mathbf{a}_\alpha + \xi^3 \hat{\mathbf{a}}_{3,\alpha}) \cdot (\mathbf{a}_\beta + \xi^3 \hat{\mathbf{a}}_{3,\beta})\\
 &= a_{\alpha\beta} + \xi^3 (\mathbf{a}_\alpha \cdot \hat{\mathbf{a}}_{3,\beta} + \hat{\mathbf{a}}_{3,\alpha} \cdot \mathbf{a}_\beta) + (\xi^3)^2 \hat{\mathbf{a}}_{3,\alpha} \cdot \hat{\mathbf{a}}_{3,\beta}\\
 &= a_{\alpha\beta} - 2\xi^3 b_{\alpha\beta} + (\xi^3)^2 c_{\alpha\beta}\\
g_{\alpha 3} &= g_{3\alpha} = \hat{\mathbf{a}}_{3}\cdot(\mathbf{a}_\alpha + \xi^3 \hat{\mathbf{a}}_{3,\alpha}) = 0\\
g_{33} &= \hat{\mathbf{a}}_{3} \cdot \hat{\mathbf{a}}_{3}= 1.
\end{split}
```
since ``\hat{\mathbf{a}}_3 \cdot \mathbf{a}_\alpha = 0`` and ``\hat{\mathbf{a}}_{3}\cdot\hat{\mathbf{a}}_{3,\alpha} = \frac{1}{2} (\hat{\mathbf{a}}_{3}\cdot\hat{\mathbf{a}}_{3})_{,\alpha} = 0``.

A common assumption made in shells is to omit the ``(\xi^3)^2`` term in ``g_{\alpha\beta}``, this assumption is called the Love--Kirchhoff strain assumption and requires the smallest radius of curvature of the shell ``R_\text{min}>t/2`` where ``t`` is the shell's thickness, see [ciarlet2005](@citet).

!!! info
    The Love Kirchhoff **strain** assumption is not to be confused the the Kirchhoff-Love **kinematics** assumption.

Using the components of the metric tensor, we can compute the Green-Lagrange strain tensor
```math
\begin{split}
e_{\alpha\beta} &= \frac{1}{2} (g_{\alpha\beta} - G_{\alpha\beta}) = \frac{1}{2} (a_{\alpha\beta} - 2\xi^3 b_{\alpha\beta} - A_{\alpha\beta} + 2\xi^3 B_{\alpha\beta})\\
& = \frac{1}{2} (a_{\alpha\beta} - A_{\alpha\beta}) - \xi^3 (b_{\alpha\beta} - B_{\alpha\beta}) \\
& = \gamma_{\alpha\beta} + \xi^3 \kappa_{\alpha\beta} \\
e_{\alpha 3} &= e_{3\alpha} = 0 \\
e_{33} &= 0
\end{split}
```
where we can clearly identify the ``\gamma_{\alpha\beta}`` and ``\kappa_{\alpha\beta}`` as the membrane and bending strain components, respectively.

!!! warning
    Something interesting happened, we specialized 3D continuum strains onto the curvilinear coordinate of the shell, the Kirchhoff-Love kinematic and the plane stress assumption result in surface strains only since only ``e_{\alpha\beta}`` are non-zero.

## 2. Internal energy

The internal energy of the shell is found by substitution of the Green-Lagrange strain tensor into the expression for the internal elastic energy
```math
\mathcal{W}_\text{int} =\frac{1}{2}\int_{-t/2}^{t/2}\!\!\int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}e_{\gamma\delta} e_{\alpha\beta} \, \sqrt{A}\,\mathrm{d}y\,\mathrm{d}\xi^3 = \frac{1}{2}\int_\omega t\,\mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \kappa_{\alpha\beta} \, \sqrt{A}\,\mathrm{d}y
```
where ``\mathbb{C}^{\alpha\beta\gamma\delta}`` is the contravariant elasticity tensor. For an isotropic elastic material, it takes the following form
```math
\mathbb{C}^{\alpha\beta\gamma\delta} = \frac{4\lambda\mu}{\lambda + 2\mu}A^{\alpha\beta}A^{\gamma\delta} + 2\mu\left( A^{\alpha\gamma}A^{\beta\delta} + A^{\alpha\delta}A^{\beta\gamma} \right),
```
where ``\lambda,\mu`` are the Lam\'e parameters and ``A^{\alpha\beta} = (A_{\alpha\beta})^{-1}`` is the **reference** contravariant metric (evaluated once on the reference surface, as in the code).

### 2.1 Residual and first variation

To obtain the residual equation, we apply the principal of stationary action in the internal energy of the system
```math
\delta\mathcal{W}_\text{int} = \int_\omega t\,\mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \delta\gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \delta\kappa_{\alpha\beta} \, \sqrt{A}\,\mathrm{d}y
```

!!! info
    ``\mathcal{W}(\gamma+\epsilon\delta\gamma) = \int_\omega\lim_{\epsilon\to0}\frac{d}{d\epsilon}\left(t\,\mathbb{C}^{\alpha\beta\gamma\delta}(\gamma_{\gamma\delta}+\epsilon\delta\gamma_{\gamma\delta})\gamma_{\alpha\beta}\right)\sqrt{A}\text{ d}y``

The variation of the membrane term is given by
```math
\delta\gamma_{\alpha\beta} = \frac{1}{2}\delta\left(\mathbf{a}_\alpha\cdot\mathbf{a}_\beta - \mathbf{A}_\alpha\cdot\mathbf{A}_\beta\right) = \frac{1}{2}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta + \mathbf{a}_\alpha\cdot\delta\mathbf{a}_\beta \right)
```
where we have used the fact that the reference configuration is fixed, so its variation is zero. For the bending part, we get a similar expression
```math
\delta\kappa_{\alpha\beta} = \delta\left(B_{\alpha\beta} - b_{\alpha\beta}\right) = -\delta b_{\alpha\beta} = -\delta\hat{\mathbf{a}}_3\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}
```
where we have used the fact that the reference curvature is fixed, so its variation is zero. The second term is relatively easy to evaluate since it only depends on the variation of the surface basis vector, but the first term is more tricky as it depends on the variation of the normal vector, which is a function of the surface basis vector. We can use the fact that the normal vector is unitary to get
```math
\hat{\mathbf{a}}_3 \cdot \hat{\mathbf{a}}_3 = 1 \implies \delta\hat{\mathbf{a}}_3 \cdot \hat{\mathbf{a}}_3 = 0
```
This can be used to transform the variation of the normal vector into a variation of the surface basis vector, which is easier to evaluate. Substituting these variations back into the first variation of the internal energy, we get
```math
\delta\kappa_{\alpha\beta} = \left(\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\gamma}\right)\mathbf{a}^\gamma\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}.
```
Where the contravariant basis can be obtained as ``\mathbf{a}^\gamma = a^{\gamma\delta}\mathbf{a}_\delta=[a_{\gamma\delta}]^{-1}\mathbf{a}_\delta``. Combining these term together, we arrive at the variational problem for the Kirchhoff-Love shell
```math
\begin{split}
\delta\mathcal{W}_\text{int} =& \int_\omega N^{\alpha\beta} \frac{1}{2}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta + \mathbf{a}_\alpha\cdot\delta\mathbf{a}_\beta \right) + \\
& M^{\alpha\beta}\left[\left(\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\gamma}\right)\mathbf{a}^\gamma\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right] \, \sqrt{A}\,\mathrm{d}y
\end{split}
```
where we have substituted ``N^{\alpha\beta} = t\,\mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta}`` and ``M^{\alpha\beta}=\frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta}``, the membrane force and bending moment resultants, respectively. Since ``N^{\alpha\beta}=N^{\beta\alpha}`` (by the symmetry of ``\mathbb{C}``), this simplifies to
```math
\delta\mathcal{W}_\text{int} = \int_\omega N^{\alpha\beta}(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta) + M^{\alpha\beta}\left[\left(\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\gamma}\right)\mathbf{a}^\gamma\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right] \sqrt{A}\,\mathrm{d}y.
```

### 2.2 Consistent tangent and second variation

The consistent tangent is obtained by taking the second variation of the internal energy, which gives us
```math
\delta\delta\mathcal{W}_\text{int} = \int_\omega \delta\left[N^{\alpha\beta}(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta)\right] + \delta\left[M^{\alpha\beta}\left(\delta\hat{\mathbf{a}}_3\cdot\mathbf{a}_{\alpha,\beta}+ \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right)\right] \sqrt{A}\,\mathrm{d}y
```
The second variation of the membrane term decomposes as
```math
\delta\left[N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right)\right] = \delta N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right) + N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\delta\mathbf{a}_\beta \right)
```
where ``\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta`` is first-order in ``\delta\mathbf{u}``, so ``\delta\delta\mathbf{a}_\alpha = 0`` at fixed ``\mathbf{a}_\alpha`` (the linearisation point). Substituting ``N^{\alpha\beta}=t\,\mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta}`` and the expression for ``\delta\gamma_{\gamma\delta}`` from Section 2.1, and using the minor symmetry ``\mathbb{C}^{\alpha\beta\gamma\delta}=\mathbb{C}^{\alpha\beta\delta\gamma}``, the membrane tangent becomes
```math
\delta\left[N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right)\right] = \delta\mathbf{a}_\alpha\left(t\,\mathbb{C}^{\alpha\beta\gamma\delta}(\delta\mathbf{a}_\gamma\cdot\mathbf{a}_\delta)\mathbf{a}_\beta + N^{\alpha\beta}\delta\mathbf{a}_\beta \right).
```
The first inner term is the **material stiffness** (depends on ``\mathbb{C}``); the second is the **geometric stiffness** (depends on the current stress resultant ``N^{\alpha\beta}``).

#### Bending contribution

The bending energy density is ``\tfrac{1}{2}\kappa_{\alpha\beta}D^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta}`` with ``D^{\alpha\beta\gamma\delta}=\frac{t^3}{12}\mathbb{C}^{\alpha\beta\gamma\delta}`` and curvature change ``\kappa_{\alpha\beta}=b_{\alpha\beta}-B_{\alpha\beta}``, ``b_{\alpha\beta}=\mathbf{a}_{,\alpha\beta}\cdot\hat{\mathbf{a}}_3`` (the overall sign of ``\kappa`` is immaterial in the quadratic energy, so we use the ``b-B`` convention of the implementation). Only ``b_{\alpha\beta}`` varies. The mid-surface enters both through ``\mathbf{a}_{,\alpha\beta}`` (linear in ``\mathbf{u}``) and through the unit normal ``\hat{\mathbf{a}}_3`` (nonlinear in ``\mathbf{u}``), so it is cleanest to differentiate the energy directly.

**First variation.** With ``\mathbf{m}=\mathbf{a}_1\times\mathbf{a}_2``, ``\hat{\mathbf{a}}_3=\mathbf{m}/\Vert\mathbf{m}\Vert`` and the tangential projector ``\mathbf{P}=\mathbf{I}-\hat{\mathbf{a}}_3\otimes\hat{\mathbf{a}}_3``,
```math
\delta\hat{\mathbf{a}}_3 = \frac{1}{\Vert\mathbf{m}\Vert}\mathbf{P}\,\delta\mathbf{m},\qquad \delta\mathbf{m}=\delta\mathbf{a}_1\times\mathbf{a}_2 + \mathbf{a}_1\times\delta\mathbf{a}_2,
```
so that
```math
\delta\kappa_{\alpha\beta}=\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{,\alpha\beta} + \frac{1}{\Vert\mathbf{m}\Vert}\big(\mathbf{a}_{,\alpha\beta}-b_{\alpha\beta}\hat{\mathbf{a}}_3\big)\cdot\delta\mathbf{m}.
```
The first term is the bending produced by the transverse curvature of the displacement field; the second is the rotation of the normal (only the tangential part ``\mathbf{P}\mathbf{a}_{,\alpha\beta}=\mathbf{a}_{,\alpha\beta}-b_{\alpha\beta}\hat{\mathbf{a}}_3`` survives). Inserting the nodal interpolation ``\delta\mathbf{a}_\gamma=\sum_I \partial_\gamma N_I\,\delta\mathbf{u}_I`` and ``\delta\mathbf{a}_{,\alpha\beta}=\sum_I \partial_{\alpha\beta}N_I\,\delta\mathbf{u}_I``, the residual is
```math
\mathbf{r}_I = \int_\omega M^{\alpha\beta}\,\frac{\partial\kappa_{\alpha\beta}}{\partial\mathbf{u}_I}\,\sqrt{A}\,\mathrm{d}y,\qquad M^{\alpha\beta}=D^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta},
```
where, writing ``[\mathbf{v}]_\times`` for the skew tensor with ``[\mathbf{v}]_\times\mathbf{w}=\mathbf{v}\times\mathbf{w}``,
```math
\frac{\partial\hat{\mathbf{a}}_3}{\partial\mathbf{u}_I}=\frac{1}{\Vert\mathbf{m}\Vert}\mathbf{P}\big(\partial_2 N_I\,[\mathbf{a}_1]_\times-\partial_1 N_I\,[\mathbf{a}_2]_\times\big),\qquad \frac{\partial\kappa_{\alpha\beta}}{\partial\mathbf{u}_I}=\partial_{\alpha\beta}N_I\,\hat{\mathbf{a}}_3 + \Big(\frac{\partial\hat{\mathbf{a}}_3}{\partial\mathbf{u}_I}\Big)^{\!\top}\mathbf{a}_{,\alpha\beta}.
```

**Second variation (consistent tangent).** Differentiating once more and using that ``\mathbf{a}_{,\alpha\beta}`` and ``\mathbf{a}_\gamma`` are linear in ``\mathbf{u}`` (all curvature nonlinearity is carried by ``\hat{\mathbf{a}}_3``),
```math
\mathbf{K}_{IJ}=\int_\omega \underbrace{\frac{\partial\kappa_{\alpha\beta}}{\partial\mathbf{u}_I}\otimes D^{\alpha\beta\gamma\delta}\frac{\partial\kappa_{\gamma\delta}}{\partial\mathbf{u}_J}}_{\text{material}} + \underbrace{M^{\alpha\beta}\,\frac{\partial^2\kappa_{\alpha\beta}}{\partial\mathbf{u}_I\partial\mathbf{u}_J}}_{\text{geometric}}\;\sqrt{A}\,\mathrm{d}y,
```
with the geometric curvature Hessian
```math
\frac{\partial^2\kappa_{\alpha\beta}}{\partial\mathbf{u}_I\partial\mathbf{u}_J}=\partial_{\alpha\beta}N_I\,\frac{\partial\hat{\mathbf{a}}_3}{\partial\mathbf{u}_J}+\partial_{\alpha\beta}N_J\,\frac{\partial\hat{\mathbf{a}}_3}{\partial\mathbf{u}_I}+\mathbf{a}_{,\alpha\beta}\cdot\frac{\partial^2\hat{\mathbf{a}}_3}{\partial\mathbf{u}_I\partial\mathbf{u}_J},
```
and the normal Hessian obtained by differentiating ``\delta\hat{\mathbf{a}}_3`` once more (write ``\rho=\Vert\mathbf{m}\Vert``, ``\rho_{,s}=\hat{\mathbf{a}}_3\cdot\mathbf{m}_{,s}``, with ``s,t`` any two displacement components):
```math
\hat{\mathbf{a}}_{3,st}=\frac{1}{\rho}\big(\mathbf{m}_{,st}-\hat{\mathbf{a}}_{3,t}\,\rho_{,s}-\hat{\mathbf{a}}_3\,\rho_{,st}\big)-\frac{\rho_{,t}}{\rho^2}\big(\mathbf{m}_{,s}-\hat{\mathbf{a}}_3\,\rho_{,s}\big),\qquad \rho_{,st}=\hat{\mathbf{a}}_{3,t}\cdot\mathbf{m}_{,s}+\hat{\mathbf{a}}_3\cdot\mathbf{m}_{,st},
```
where the only non-zero second derivative of ``\mathbf{m}`` is the node-coupling bilinear term ``\mathbf{m}_{,st}=\partial^2\mathbf{m}/\partial u_{Ic}\partial u_{Jd}=\partial_1 N_I\partial_2 N_J\,(\mathbf{e}_c\times\mathbf{e}_d)+\partial_1 N_J\partial_2 N_I\,(\mathbf{e}_d\times\mathbf{e}_c)``.

The material term is a rank-one outer product of curvature gradients (positive semi-definite); the geometric term carries the current moment ``M^{\alpha\beta}`` and is what renders the tangent indefinite under compressive bending. In the code these two contributions are assembled together by taking `ForwardDiff.hessian` of [`FerriteShells.bending_energy_KL`](@ref); the closed forms above are exactly what that differentiation evaluates, and the FD tangent tests (`kl_fd_tangent`) verify the agreement.

!!! note
    Because the Kirchhoff-Love formulation requires C¹ continuity between elements (the bending energy depends on second derivatives of the displacement field), standard C⁰ Lagrange elements are not strictly conforming. In FerriteShells the `_KL` functions use C⁰ quadratic elements (Q9), which are conforming for membrane but only approximately so for bending. In practice this works well for flat or mildly curved shells, but KL on strongly curved geometries requires C¹ or subdivision elements.