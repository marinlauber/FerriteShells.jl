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
\mathcal{W}_\text{int} =\int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}e_{\gamma\delta} e_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \kappa_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y
```
where ``\mathbb{C}^{\alpha\beta\gamma\delta}`` is the contravariant elasticity tensor. For an isotropic elastic material, it takes the following form
```math
\mathbb{C}^{\alpha\beta\gamma\delta} = \frac{4\lambda\mu}{\lambda + 2\mu}a^{\alpha\beta}a^{\gamma\delta} + 2\mu\left( a^{\alpha\gamma}a^{\beta\delta} + a^{\alpha\delta}a^{\beta\gamma} \right),
```
where ``\lambda,\mu`` are Lam\'e parameters.

### 2.1 Residual and first variation

To obtain the residual equation, we apply the principal of stationary action in the internal energy of the system
```math
\delta\mathcal{W}_\text{int} = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \delta\gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \delta\kappa_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y
```

!!! info
    ``\mathcal{W}(\gamma+\epsilon\delta\gamma) = \int_\omega\lim_{\epsilon\to0}\frac{d}{d\epsilon}\left(\mathbb{C}^{\alpha\beta\gamma\delta}(\gamma_{\gamma\delta}+\epsilon\delta\gamma_{\gamma\delta})\gamma_{\alpha\beta}\right)\sqrt{a}\text{ d}y``

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
& M^{\alpha\beta}\left[\left(\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\gamma}\right)\mathbf{a}^\gamma\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right] \, \sqrt{a}\,\mathrm{d}y
\end{split}
```
where we have substituted ``N^{\alpha\beta} = \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta}`` and ``M^{\alpha\beta}=\frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta}``, the membrane force and bending moment resultants, respectively. Since ``N^{\alpha\beta}=N^{\beta\alpha}`` (by the symmetry of ``\mathbb{C}``), this simplifies to
```math
\delta\mathcal{W}_\text{int} = \int_\omega N^{\alpha\beta}(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta) + M^{\alpha\beta}\left[\left(\hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\gamma}\right)\mathbf{a}^\gamma\cdot\mathbf{a}_{\alpha,\beta} - \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right] \sqrt{a}\,\mathrm{d}y.
```

### 2.2 Consistent tangent and second variation

The consistent tangent is obtained by taking the second variation of the internal energy, which gives us
```math
\delta\delta\mathcal{W}_\text{int} = \int_\omega \delta\left[N^{\alpha\beta}(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta)\right] + \delta\left[M^{\alpha\beta}\left(\delta\hat{\mathbf{a}}_3\cdot\mathbf{a}_{\alpha,\beta}+ \hat{\mathbf{a}}_3\cdot\delta\mathbf{a}_{\alpha,\beta}\right)\right] \sqrt{a}\,\mathrm{d}y
```
The second variation of the membrane term decomposes as
```math
\delta\left[N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right)\right] = \delta N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right) + N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\delta\mathbf{a}_\beta \right)
```
where ``\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta`` is first-order in ``\delta\mathbf{u}``, so ``\delta\delta\mathbf{a}_\alpha = 0`` at fixed ``\mathbf{a}_\alpha`` (the linearisation point). Substituting ``N^{\alpha\beta}=\mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta}`` and the expression for ``\delta\gamma_{\gamma\delta}`` from Section 2.1, and using the minor symmetry ``\mathbb{C}^{\alpha\beta\gamma\delta}=\mathbb{C}^{\alpha\beta\delta\gamma}``, the membrane tangent becomes
```math
\delta\left[N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right)\right] = \delta\mathbf{a}_\alpha\left(\mathbb{C}^{\alpha\beta\gamma\delta}(\delta\mathbf{a}_\gamma\cdot\mathbf{a}_\delta)\mathbf{a}_\beta + N^{\alpha\beta}\delta\mathbf{a}_\beta \right).
```
The first inner term is the **material stiffness** (depends on ``\mathbb{C}``); the second is the **geometric stiffness** (depends on the current stress resultant ``N^{\alpha\beta}``).

The bending contribution follows analogously but is considerably more involved because the second variation of ``\hat{\mathbf{a}}_3`` introduces third-order terms in ``\mathbf{a}_\alpha``. In practice the bending residual and tangent are computed via automatic differentiation of [`FerriteShells.bending_energy_KL`](@ref) using ForwardDiff.jl, which avoids the need for the explicit second variation.

!!! note
    Because the Kirchhoff-Love formulation requires C¹ continuity between elements (the bending energy depends on second derivatives of the displacement field), standard C⁰ Lagrange elements are not strictly conforming. In FerriteShells the `_KL` functions use C⁰ quadratic elements (Q9), which are conforming for membrane but only approximately so for bending. In practice this works well for flat or mildly curved shells, but KL on strongly curved geometries requires C¹ or subdivision elements.