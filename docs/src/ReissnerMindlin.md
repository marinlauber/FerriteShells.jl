# Reissner-Mindlin / Naghdi shell

## 1. Kinematics

![Shell kinematics](images/shell_kinematic.png)

The Reissner-Mindlin kinematic relaxes the kirchhoff-Love zero shear strain assumption through orthogonality of material lines. The shear strain measures the rotation of these material lines around the normal vector of the shell's midsurface ``\hat{\mathbf{a}}_3``
```math
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \mathbf{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)
```
where ``\mathbf{d}(\xi^1,\xi^2)`` is the director at a point ``(\xi^1,\xi^2)`` on the midsurface and ``\gamma_\alpha=\mathbf{A}_\alpha\cdot\mathbf{d}`` is the transverse shear strain (evaluated against the reference basis ``\mathbf{A}_\alpha``).

The surface basis vector are given by
```math
\begin{split}
\mathbf{g}_\alpha &= \frac{\partial \Phi(\xi^1,\xi^2)}{\partial \xi^\alpha} = \frac{\partial}{\partial\xi^\alpha}\left[\phi(\xi^1,\xi^2) + \xi^3\mathbf{d}_3(\xi^1,\xi^2)\right]\\
&= \mathbf{a}_\alpha + \xi^3 \mathbf{d}_{,\alpha} \\
\mathbf{g}_3 &= \frac{\partial}{\partial\xi^3}\left[\phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)\right] = \mathbf{d}
\end{split}
```
using the definition of ``\mathbf{a}_\alpha``. From this, we can get the components of the metric tensor
```math
\begin{split}
g_{\alpha\beta} &= \mathbf{g}_\alpha \cdot \mathbf{g}_\beta = (\mathbf{a}_\alpha+\xi^3\mathbf{d}_{,\alpha})\cdot(\mathbf{a}_\beta+\xi^3\mathbf{d}_{,\beta})\\
 &= a_{\alpha\beta} + \xi^3(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) + (\xi^3)^2\mathbf{d}_{,\alpha}\cdot\mathbf{d}_{,\beta}\\
 &= a_{\alpha\beta} + \xi^3(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) + O(t^2) \\
g_{\alpha 3} &= g_{3\alpha} = (\mathbf{a}_\alpha+\xi^3\mathbf{d}_{,\alpha})\cdot\mathbf{d} = \mathbf{a}_\alpha\cdot\mathbf{d} + \xi^3\mathbf{d}_{,\alpha}\cdot\mathbf{d}\\
g_{33} &= 1.
\end{split}
```
where the plane stress assumption gives ``g_{33}=\mathbf{d}\cdot\mathbf{d}=1`` (unit director), and the shear components are non-zero. Two simplifications are applied:

**In-plane metric** ``g_{\alpha\beta}``: the ``(\xi^3)^2`` term is dropped (Love-Kirchhoff strain assumption, valid for ``R_\text{min}>t/2``).

**Shear metric** ``g_{\alpha3}``: the full expression is ``\mathbf{a}_\alpha\cdot\mathbf{d} + \xi^3\mathbf{d}_{,\alpha}\cdot\mathbf{d}``. The second term vanishes for two consistent reasons. First, if ``\Vert\mathbf{d}\Vert=1`` everywhere then differentiating ``\mathbf{d}\cdot\mathbf{d}=1`` gives ``\mathbf{d}_{,\alpha}\cdot\mathbf{d}=0`` exactly (the Rodrigues director satisfies this at nodes; the interpolated director deviates by ``O(h^2)``). Second, even without a unit director, ``\xi^3\in[-t/2,t/2]`` and ``\mathbf{d}_{,\alpha}=O(1/R)``, so the term is ``O(t/R)`` — the same order as the already-dropped ``(\xi^3)^2`` correction, so consistency requires dropping it too.

The simplified metric is therefore
```math
\begin{split}
g_{\alpha\beta} &\approx a_{\alpha\beta} + \xi^3(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha})\\
g_{\alpha 3} &= g_{3\alpha} \approx \mathbf{a}_\alpha\cdot\mathbf{d}\\
g_{33} &= 1.
\end{split}
```

The Green-Lagrange strain components follow from ``e_{ij}=\tfrac{1}{2}(g_{ij}-G_{ij})``:
```math
\begin{split}
e_{\alpha\beta} &= \underbrace{\frac{1}{2}(a_{\alpha\beta} - A_{\alpha\beta})}_{\gamma_{\alpha\beta}} + \xi^3\underbrace{\frac{1}{2}(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) - B_{\alpha\beta}}_{\kappa_{\alpha\beta}}\\
e_{\alpha3} &= e_{3\alpha} = \frac{1}{2}(\mathbf{a}_\alpha\cdot\mathbf{d} - \mathbf{A}_\alpha\cdot\mathbf{G}_3) = \frac{1}{2}\gamma_\alpha\\
e_{33} &= 0.
\end{split}
```

where ``\gamma_{\alpha\beta}`` is the membrane (in-plane) strain, ``\kappa_{\alpha\beta}`` the bending curvature change, and ``\gamma_\alpha = \mathbf{a}_\alpha\cdot\mathbf{d} - \mathbf{A}_\alpha\cdot\mathbf{G}_3`` the transverse shear strain. In the reference configuration (``\mathbf{d}=\mathbf{G}_3``, ``\mathbf{a}_\alpha=\mathbf{A}_\alpha``), all strains vanish identically.

!!! note
    Because ``g_{\alpha3}`` is independent of ``\xi^3``, the shear strain ``e_{3\alpha}`` is **constant through the thickness**. This is the Reissner-Mindlin assumption: the director rotates rigidly, so shear does not vary. In 3D elasticity the shear stress is parabolic; the shear correction factor ``\kappa_s=5/6`` compensates by matching the constant-strain energy to the parabolic-distribution energy of a rectangular cross-section. The Kirchhoff constraint ``e_{3\alpha}=0`` is recovered in the limit ``\mathbf{d}\to\hat{\mathbf{a}}_3``.

### 1.1 Director parametrization

There are a few ways to parametrize the the director vector, and the different choice lead to different discretization. One way is to discretize each of its components, leading to an additional 3 degrees of freedom per node. This is the simplest way, but requires enforcing ``\Vert\mathbf{d}\Vert=1`` through a Lagrange multiplier approach and static condensation, which results in an overall complex implementation.

Another way is to use additive vector rotations starting from the midsurface normal
```math
\mathbf{d} = \hat{\mathbf{a}}_3 + \theta_1\mathbf{T}_1 + \theta_2\mathbf{T}_2
```
which removes one unknown since we only require ``\theta_1,\theta_2`` to fully describe ``\mathbf{d}``. One issue with this formulation is that the unitary of the director is not enforced ``\Vert\mathbf{d}\Vert\neq1``. This limits the formulation to small rotations ``\Vert\mathbf{\theta}\Vert\ll1`` as large ``\Vert\mathcal{d}\Vert`` would lead to large shear strains (``\gamma_\alpha=\mathbf{a}_\alpha\cdot\mathbf{d}``) resulting in shear locking as all the internal energy is taken by shear.

For finite rotation nonlinear shell, we would like to parametrize ``\mathbf{d}`` in a way that naturally enforces the ``\Vert\mathbf{d}\Vert=1`` constraint. One way to do this is through Rodrigue's parametrization
```math
\mathbf{d} = \cos{\Vert\mathbf{\theta}\Vert}\cdot\hat{\mathbf{a}}_3 + \text{sinc}{\Vert\theta\Vert}\cdot(\theta_1\cdot\mathbf{T}_1 + \theta_2\cdot\mathbf{T}_2)
```
which guarantees ``\Vert\mathbf{d}\Vert=1`` for any ``\theta_1,\theta_2`` (since ``\cos^2\Vert\theta\Vert + \Vert\theta\Vert^2\mathrm{sinc}^2\Vert\theta\Vert = 1``). The parametrization has a coordinate singularity at ``\Vert\boldsymbol{\theta}\Vert=\pi``. In practice this is avoided by keeping each load increment small enough that ``\Vert\boldsymbol{\theta}\Vert<\pi`` within a step, or by using a total-Lagrangian update that resets the reference configuration periodically.
In the following, we will keep the director variation terms general since explicit variation of the director is messy, especially here since we use a Rodrigue's parametrization.

!!! info
    In practice, we use ``\theta^2`` in the trigonometric functions to enforce directly the constraint on the rotations, but this means that for small rotations, we could take the square-root of a very small number, which could lead to overflow. To avoid this, we use a Taylor-series expansion to evaluate the trigonometric functions for ``\mathbf{\theta}^2<10^{-6}``, and the normal expression otherwise.

## 2. Internal energy

The internal energy splits into membrane, bending, and transverse shear contributions
```math
\mathcal{W}_\text{int} = \int_\omega \frac{t}{2} N^{\alpha\beta} \gamma_{\alpha\beta} + \frac{t^3}{24} M^{\alpha\beta} \kappa_{\alpha\beta} + \frac{5t}{12} Q^\alpha \gamma_\alpha \, \sqrt{A}\,\mathrm{d}y
```
where the stress resultants are
```math
N^{\alpha\beta} = \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta}, \quad
M^{\alpha\beta} = \frac{t^2}{12}\mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta}, \quad
Q^\alpha = \frac{E}{2(1+\nu)} a^{\alpha\beta}\gamma_\beta,
```
and the strain measures are (using the Naghdi form with current base vectors ``\mathbf{a}_\alpha``)
```math
\gamma_{\alpha\beta} = \tfrac{1}{2}(a_{\alpha\beta} - A_{\alpha\beta}), \quad
\kappa_{\alpha\beta} = \tfrac{1}{2}(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha}) - B_{\alpha\beta}, \quad
\gamma_\alpha = \mathbf{A}_\alpha\cdot\mathbf{d} - \delta_{\alpha 3}.
```

### 2.1 Residual and first variation

The first variation of the internal energy gives the residual vector. Denoting the virtual displacement as ``\delta\mathbf{u}`` and virtual rotation as ``\delta\boldsymbol{\varphi}``, the three contributions are

**Membrane** (displacement DOFs):
```math
\delta\mathcal{W}_\text{mem} = \int_\omega N^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{a}_\beta\right)\sqrt{A}\,\mathrm{d}y
```

**Bending** (displacement and rotation DOFs):
```math
\delta\mathcal{W}_\text{bend} = \int_\omega M^{\alpha\beta}\left(\delta\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta} + \mathbf{a}_\alpha\cdot\delta\mathbf{d}_{,\beta}\right)\sqrt{A}\,\mathrm{d}y
```

**Shear** (displacement and rotation DOFs):
```math
\delta\mathcal{W}_\text{shear} = \int_\omega Q^\alpha\left(\delta\mathbf{a}_\alpha\cdot\mathbf{d} + \mathbf{A}_\alpha\cdot\delta\mathbf{d}\right)\sqrt{A}\,\mathrm{d}y
```

The variation of the director ``\delta\mathbf{d}`` follows from the Rodrigues parametrization. Denoting the Rodrigues Jacobian ``\mathrm{dd}_{Il} = \partial\mathbf{d}_I/\partial\varphi_{I,l}``, the rotation DOF residual at node ``I`` is
```math
r_{I,k}^\varphi = \int_\omega F_I \cdot \mathrm{dd}_{Ik}\,\sqrt{A}\,\mathrm{d}y, \quad F_I = \partial_1 N_I S^1 + \partial_2 N_I S^2 + N_I(Q_1\mathbf{a}_1+Q_2\mathbf{a}_2)
```
where ``S^\alpha = M^{\alpha\beta}\mathbf{d}_{,\beta}``.

In FerriteShells the explicit forms are implemented in [`membrane_residuals_RM!`](@ref) and [`bending_residuals_RM!`](@ref). ForwardDiff-based residuals are also available as [`residuals_RM_FD!`](@ref) for both membrane, bending and shear contributions.

### 2.2 Consistent tangent and second variation

The consistent tangent is the second variation of ``\mathcal{W}_\text{int}``. It has four blocks per node pair ``(I,J)``:

- **uu** (3×3): material part ``\partial_\alpha N_I \partial_\gamma N_J \mathbb{C}^{\alpha\beta\gamma\delta}\mathbf{a}_\beta\otimes\mathbf{a}_\delta`` + geometric part ``(\partial_\alpha N_I N^{\alpha\beta}\partial_\beta N_J)\mathbf{I}_3``.
- **uφ** (3×2): couples displacement gradient of node ``I`` with director variation at node ``J``.
- **φu** (2×3): transpose of the uφ block at ``(J,I)``, by symmetry of ``\mathcal{W}``.
- **φφ** (2×2): material part from ``\delta F_I\cdot\mathrm{dd}_{Jk}`` plus a geometric part from the second Rodrigues derivative ``\partial^2\mathbf{d}_I/\partial\varphi_k\partial\varphi_l`` (non-zero only for the diagonal ``J=I`` block).

The explicit implementation is in [`membrane_tangent_RM!`](@ref) and [`bending_tangent_RM!`](@ref). ForwardDiff-based variants are available as [`tangent_RM_FD!`](@ref).