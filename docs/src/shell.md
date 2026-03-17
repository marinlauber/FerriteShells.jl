## 1. Shell formulations

### 1.1 Shell theory comparison

| | Linear KL | Koiter | Reissner–Mindlin | Naghdi |
|---|---|---|---|---|
| **DOFs/node** | 3 (``u_1,u_2,u_3``) | 3 (``u_1,u_2,u_3``) | 5 (``u_1,u_2,u_3,\varphi_1,\varphi_2``) | 5 (``u_1,u_2,u_3,\varphi_1,\varphi_2``) |
| **Director** | implicit: ``\mathbf{n} = \mathbf{a}_1\times\mathbf{a}_2/\|\cdot\|`` | implicit: ``\mathbf{n} = \mathbf{a}_1\times\mathbf{a}_2/\|\cdot\|`` | additive: ``\mathbf{d} = \mathbf{G}_3+\varphi_1\mathbf{T}_1+\varphi_2\mathbf{T}_2``, ``\|\mathbf{d}\|\neq 1`` | Rodrigues: ``\mathbf{d} = \cos\|\varphi\|\,\mathbf{G}_3+\mathrm{sinc}\|\varphi\|(\varphi_1\mathbf{T}_1+\varphi_2\mathbf{T}_2)``, ``\|\mathbf{d}\|=1`` |
| **Membrane strain** | linear: ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{u}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{u}_{,\alpha})`` | Green–Lagrange: ``\tfrac{1}{2}(a_{\alpha\beta}-A_{\alpha\beta})`` | linear: ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{u}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{u}_{,\alpha})`` | Green–Lagrange: ``\tfrac{1}{2}(a_{\alpha\beta}-A_{\alpha\beta})`` |
| **Bending strain** | ``\kappa_{\alpha\beta} = -u_{3,\alpha\beta}`` (flat ref.) | ``\kappa_{\alpha\beta} = b_{\alpha\beta}-B_{\alpha\beta}`` | ``\tfrac{1}{2}(\mathbf{A}_\alpha\cdot\mathbf{d}_{,\beta}+\mathbf{A}_\beta\cdot\mathbf{d}_{,\alpha})-B_{\alpha\beta}`` | ``\tfrac{1}{2}(\mathbf{a}_\alpha\cdot\mathbf{d}_{,\beta}+\mathbf{a}_\beta\cdot\mathbf{d}_{,\alpha})-B_{\alpha\beta}`` |
| **Transverse shear** | ``\gamma=0`` (Kirchhoff) | ``\gamma=0`` (Kirchhoff) | ``\gamma_\alpha = \mathbf{A}_\alpha\cdot\mathbf{d}`` | ``\gamma_\alpha = \mathbf{a}_\alpha\cdot\mathbf{d}`` |
| **Finite rotations** | no | yes | small only (``\|\varphi\|\ll 1``) | yes |
| **C¹ for bending** | yes | yes | no | no |
| **In FerriteShells** | — | ✓ `_KL` functions | — | ✓ `_RM` functions |

The key distinction between RM and Naghdi is which base vectors appear in the strain measures: RM uses the **reference** base vectors ``A_\alpha`` (linearised around the reference configuration), while Naghdi replaces them with the **current** ``a_\alpha`` everywhere, giving fully nonlinear strains. The director parametrisation (non-unit additive vs unit Rodrigues) is a separate but related choice — in practice the two always appear together.

Koiter has no director DOFs; the normal is always implicit from the surface geometry, so the Kirchhoff constraint (zero shear) is built in and C¹ continuity is required for bending.

Classical RM (additive director, ``\|\mathbf{d}\|\neq 1``) is not implemented; the `_RM` functions go directly to the geometrically exact Naghdi form via Rodrigues parametrisation.

## 2. Shells kinematics

Curvilinear covariant basis vector (3D)
```math
\mathbf{g}_i=\frac{\partial \Phi(\xi^1,\xi^2,\xi^3)}{\partial \xi^i}, \quad i\in 1,2,3
```
where the latin indices run from 1 to 3. The **surface** covariant basis vector
```math
\mathbf{a}_\alpha=\frac{\partial \phi(\xi^1,\xi^2)}{\partial \xi^\alpha}, \quad \alpha\in 1,2
```
with greek indices running from 1 to 2. The first fundamental form or **metric tensor** of the surface and the curvilinear coordinate is then given
```math
a_{\alpha\beta} = \mathbf{a}_\alpha \cdot \mathbf{a}_\beta, \quad g_{\alpha\beta} = \mathbf{g}_\alpha \cdot \mathbf{g}_\beta
```
inverse of the **surface** first fundamental form
```math
a^{\alpha\beta} = [a_{\alpha\beta}]^{-1}
```
which allows to transform covariant quantities into their contravariant form
```math
\mathbf{a}^\alpha = a^{\alpha\beta}\mathbf{a}_\beta
```

The (unit) surface normal is then
```math
\hat{\mathbf{a}}_3 = \frac{\mathbf{a}_1 \times \mathbf{a}_2}{\|\mathbf{a}_1 \times \mathbf{a}_2\|}
```
which satisfies ``\mathbf{a}_3\cdot\mathbf{a}_\alpha=0``, and thus the second fundamental form of the surface is found
```math
b_{\alpha\beta} = \hat{\mathbf{a}}_3\cdot \mathbf{a}_{\alpha,\beta} = -\mathbf{a}_\alpha\cdot \hat{\mathbf{a}}_{3,\beta}
```

#### 2.1 Green-Lagrange strain tensor

The Green-Lagrange strain tensor is given by half the increment in the metric tensor [chapelle2011](@cite)
```math
e_{ij} = \frac{1}{2}\left(g_{ij} - G_{ji}\right).
```
Subsituting the definition of ``g_{ij}``, we get
```math
e_{ij} = \frac{1}{2}\left(\mathbf{g}_i\cdot\mathbf{g}_j - \mathbf{G}_j\cdot\mathbf{G}_i\right)
```

### 2.2 Kirchhoff-Love / Koiter shell

The Kirchhoff-Love kinematic assumption prevents transverse shear strain by constraining the cross-section to remain normal the the shell's midsurface during defomation.
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

A common assumtion made in shells is to ommit the ``(\xi^3)^2`` term in ``g_{\alpha\beta}``, this assumption is called the Love--Kirchhoff strain assumption and requires the smallest radius of curvature of the shell ``R_\text{min}>t/2`` where ``t`` is the shell's thickness, see [ciarlet2005](@cite).

> [!WARNING]
> The Love Kirchhoff **strain** assumption is not to be confused the the Kirchhoff-Love **kinematics** assumption.

Using the components of the metric tensor, we can compute the Green-Lagrange strain tensor
```math
\begin{split}
e_{\alpha\beta} &= \frac{1}{2} (g_{\alpha\beta} - G_{\alpha\beta}) = \frac{1}{2} (a_{\alpha\beta} - 2\xi^3 b_{\alpha\beta} - A_{\alpha\beta} + 2\xi^3 B_{\alpha\beta})\\
& = \frac{1}{2} (a_{\alpha\beta} - A_{\alpha\beta}) - \xi^3 (b_{\alpha\beta} - B_{\alpha\beta}) \\
& = \gamma_{\alpha\beta} - \xi^3 \kappa_{\alpha\beta} \\
e_{\alpha 3} &= e_{3\alpha} = 0 \\
e_{33} &= 0
\end{split}
```
where we can clearly identify the ``\gamma_{\alpha\beta}`` and ``\kappa_{\alpha\beta}`` as the membrane and bending strain components, respectively.

> [!NOTE]
> Something interesting happened, we specialized 3D continuum strains onto the curvilinear coordinate of the shell, the Kirchhoff-Love kinematic and the plane stress assumption result in surface strains only since only ``e_{\alpha\beta}`` are non-zero.

#### 2.2.1 Weak form, residual and consistent tangent

The internal energy of the shell is given by
```math
\mathcal{W}_\text{int} =\int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}e_{\gamma\delta} e_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \kappa_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y = \int_\omega p^\alpha\eta_\alpha \, \sqrt{a}\,\mathrm{d}y
```
where ``\mathbb{C}^{\alpha\beta\gamma\delta}`` is the contravariant elasticity tensor
```math
\mathbb{C}^{\alpha\beta\gamma\delta} = \frac{4\lambda\mu}{\lambda + 2\mu}a^{\alpha\beta}a^{\gamma\delta} + 2\mu\left( a^{\alpha\gamma}a^{\beta\delta} + a^{\alpha\delta}a^{\beta\gamma} \right)
```

To obtain the residual equation, we apply the principal of stationnary action in the internal energy of the system
```math
\delta\mathcal{W}_\text{int} = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \delta\gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \delta\kappa_{\alpha\beta} \, \sqrt{a}\,\mathrm{d}y = \int_\omega \delta p^\alpha\eta_\alpha \, \sqrt{a}\,\mathrm{d}y
```

### 2.3 Reissner-Mindlin / Naghdi shell
The Reissner-Mindlin kinematic assumption
```math
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \mathbf{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\mathbf{d}(\xi^1,\xi^2)
```

## References

```@bibliography
```