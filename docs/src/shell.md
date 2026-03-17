## 1. Shell formulations

### 1.1 Shell theory comparison

| | Linear KL | Koiter | Reissner–Mindlin | Naghdi |
|---|---|---|---|---|
| **DOFs/node** | 3 (u₁,u₂,u₃) | 3 (u₁,u₂,u₃) | 5 (u₁,u₂,u₃,φ₁,φ₂) | 5 (u₁,u₂,u₃,φ₁,φ₂) |
| **Director** | implicit: **n** = **a**₁×**a**₂/‖·‖ | implicit: **n** = **a**₁×**a**₂/‖·‖ | additive: **d** = **G**₃+φ₁**T**₁+φ₂**T**₂, ‖**d**‖≠1 | Rodrigues: **d** = cos‖φ‖**G**₃+sinc‖φ‖(φ₁**T**₁+φ₂**T**₂), ‖**d**‖=1 |
| **Membrane strain** | linear: ½(**A**_α·**u**,β+**A**_β·**u**,α) | Green–Lagrange: ½(a_αβ−A_αβ) | linear: ½(**A**_α·**u**,β+**A**_β·**u**,α) | Green–Lagrange: ½(a_αβ−A_αβ) |
| **Bending strain** | κ_αβ = −u₃,αβ (flat ref.) | κ_αβ = b_αβ−B_αβ | ½(**A**_α·**d**,β+**A**_β·**d**,α)−B_αβ | ½(**a**_α·**d**,β+**a**_β·**d**,α)−B_αβ |
| **Transverse shear** | γ=0 (Kirchhoff) | γ=0 (Kirchhoff) | γ_α = **A**_α·**d** | γ_α = **a**_α·**d** |
| **Finite rotations** | no | yes | small only (‖φ‖≪1) | yes |
| **C¹ for bending** | yes | yes | no | no |
| **In FerriteShells** | — | ✓ `_KL` functions | — | ✓ `_RM` functions |

The key distinction between RM and Naghdi is which base vectors appear in the strain measures: RM uses the **reference** base vectors **A**_α (linearised around the reference configuration), while Naghdi replaces them with the **current** **a**_α everywhere, giving fully nonlinear strains. The director parametrisation (non-unit additive vs unit Rodrigues) is a separate but related choice — in practice the two always appear together.

Koiter has no director DOFs; the normal is always implicit from the surface geometry, so the Kirchhoff constraint (zero shear) is built in and C¹ continuity is required for bending.

Classical RM (additive director, ‖**d**‖≠1) is not implemented; the `_RM` functions go directly to the geometrically exact Naghdi form via Rodrigues parametrisation.

## 2. Shells kinematics

Curvilinear covariant basis vector (3D)
$$
\bm{g}_i=\frac{∂ \Phi(\xi^1,\xi^2,\xi^3)}{∂ \xi^i}, \quad i\in1,2,3
$$
where the latin indices run from 1 to 3. The **surface** covariant basis vector
$$
\bm{a}_\alpha=\frac{∂ \phi(\xi^1,\xi^2)}{∂ \xi^α}, \quad \alpha\in1,2
$$
with greek indices running from 1 to 2. The first fundamental form or **metric tensor** of the surface and the curvilinear coordinate is then given
$$
a_{\alpha\beta} = \bm{a}_\alpha \cdot \bm{a}_\beta, \quad g_{\alpha\beta} = \bm{g}_\alpha \cdot \bm{g}_\beta
$$
inverse of the **surface** first fundamental form
$$
a^{\alpha\beta} = [a_{\alpha\beta}]^{-1}
$$
which allows to transform covariant quantities into their contravariant form
$$
\bm{a}^α = a^{\alpha\beta}\bm{a}_\beta
$$

The (unit) surface normal is then
$$
\hat{\bm{a}}_3 = \frac{\bm{a}_1 \times \bm{a}_2}{\|\bm{a}_1 \times \bm{a}_2\|}
$$
which satisfies $\bm{a}_3\cdot\bm{a}_\alpha=0$, and thus the second fundamental form of the surface is found
$$
b_{\alpha\beta} = \hat{\bm{a}_3}⋅ \bm{a}_{\alpha,\beta} = -\bm{a}_\alpha⋅ \hat{\bm{a}}_{3,\beta}
$$

#### 2.1 Green-Lagrange strain tensor

The Green-Lagrange strain tensor is given by half the increment in the metric tensor [ref]
$$
e_{ij} = \frac{1}{2}\left(g_{ij} - G_{ji}\right).
$$
Subsituting the definition of $g_{ij}$, we get
$$
e_{ij} = \frac{1}{2}\left(\bm{g}_i\cdot\bm{g}_j - \bm{G}_j\cdot\bm{G}_i\right)
$$

### 2.2 Kirchhoff-Love / Koiter shell

The Kirchhoff-Love kinematic assumption prevents transverse shear strain by constraining the cross-section to remain normal the the shell's midsurface during defomation.
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\hat{\bm{a}}_3(\xi^1,\xi^2)
$$
where $\hat{\bm{a}}_3$ is the unit normal to the midsurface. The surface basis vector are given by
$$
\begin{split}
\bm{g}_\alpha &= \frac{∂ \Phi(\xi^1,\xi^2)}{∂ \xi^α} = \frac{\partial}{\partial\xi^\alpha}\left[\phi(\xi^1,\xi^2) + \xi^3\hat{\bm{a}}_3(\xi^1,\xi^2)\right]\\
&= \bm{a}_\alpha + \xi^3 \hat{\bm{a}}_{3,\alpha} \\
\bm{g}_3 &= \frac{\partial}{\partial\xi^3}\left[\phi(\xi^1,\xi^2) + \xi^3\hat{\bm{a}}_3(\xi^1,\xi^2)\right] =\hat{\bm{a}}_3
\end{split}
$$
using the definition of $\bm{a}_\alpha$. From this, we can get the components of the metric tensor
$$
\begin{split}
g_{\alpha\beta} &= \bm{g}_\alpha \cdot \bm{g}_\beta = (\bm{a}_\alpha + \xi^3 \hat{\bm{a}}_{3,\alpha}) \cdot (\bm{a}_\beta + \xi^3 \hat{\bm{a}}_{3,\beta})\\
 &= a_{\alpha\beta} + \xi^3 (a_\alpha \cdot \hat{\bm{a}}_{3,\beta} + \hat{\bm{a}}_{3,\alpha} \cdot a_\beta) + (\xi^3)^2 \hat{\bm{a}}_{3,\alpha} \cdot \hat{\bm{a}}_{3,\beta}\\
 &= a_{\alpha\beta} - 2\xi^3 b_{\alpha\beta} + (\xi^3)^2 c_{\alpha\beta}\\
g_{α 3} &= g_{3α} = \hat{\bm{a}}_{3}\cdot(\bm{a}_\alpha + \xi^3 \hat{\bm{a}}_{3,\alpha}) = 0\\
g_{33} &= \hat{\bm{a}}_{3} ⋅ \hat{\bm{a}}_{3}= 1.
\end{split}
$$
since $\hat{\bm{a}}_3 \cdot \bm{a}_\alpha = 0$ and $\hat{\bm{a}}_{3}\cdot\hat{\bm{a}}_{3,\alpha} = \frac{1}{2} (\hat{\bm{a}}_{3}\cdot\hat{\bm{a}}_{3})_{,\alpha} = 0$.

A common assumtion made in shells is to ommit the $(\xi^3)^2$ term in $g_{\alpha\beta}$, this assumtpions is called the Love--Kirchhoff strain assumption and requires the smallest radius of curvature of the shell $R_\text{min}>t/2$ there $t$ is the shell's thickness, see Ciarlet [ref].

> ![Warning]
> The Love Kirchhoff **strain** assumption is not to be confused the the Kirchhoff-Love **kinematics** assumption.

Using the components of the metric tensor, we can compute the Gree-Lagrange strain tensor
$$
\begin{split}
e_{\alpha\beta} &= \frac{1}{2} (g_{\alpha\beta} - G_{\alpha\beta}) = \frac{1}{2} (a_{\alpha\beta} - 2\xi^3 b_{\alpha\beta} - A_{\alpha\beta} + 2\xi^3 B_{\alpha\beta})\\
& = \frac{1}{2} (a_{\alpha\beta} - A_{\alpha\beta}) - \xi^3 (b_{\alpha\beta} - B_{\alpha\beta}) \\
& = \gamma_{\alpha\beta} - \xi^3 \kappa_{\alpha\beta} \\
e_{α 3} &= e_{3\alpha} = 0 \\
e_{3,3} &= 0
\end{split}
$$
where we can clearly identify the $\gamma_{\alpha\beta}$ and $\kappa_{\alpha\beta}$ as the membrane and bending strain components, respectively.

> ![Note]
> Something interesting happened, we specialized 3D continuum strains onto the curvilinear coordinate of the shell, the Kirchhoff-Love kinematic and the plane stress assumption result in surface strains only since only $e_{\alpha\beta}$ are non-zero.

#### 2.2.1 Weak form, residual and consistent tangent

The internal energy of the shell is given by
$$
\mathcal{W}_\text{int} =\int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}e_{\gamma\delta} e_{\alpha\beta} \, \sqrt{a}\text{ d}y = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \kappa_{\alpha\beta} \, \sqrt{a}\text{ d}y = \int_\omega p^α\eta_\alpha \, \sqrt{a}\text{ d}y
$$
where $\mathbb{C}^{\alpha\beta\gamma\delta}$ is the contravariant elasticity tensor
$$
\mathbb{C}^{\alpha\beta\gamma\delta} = \frac{4\lambda\mu}{\lambda + 2\mu}a^{\alpha\beta}a^{\gamma\delta} + 2\mu\left( a^{\alpha\gamma}a^{\beta\delta} + a^{\alpha\delta}a^{\beta\gamma} \right)
$$

To obtain the residual equation, we apply the principal of stationnary action in the internal energy of the system
$$
\delta\mathcal{W}_\text{int} = \int_\omega \mathbb{C}^{\alpha\beta\gamma\delta}\gamma_{\gamma\delta} \delta\gamma_{\alpha\beta} +  \frac{t^3}{12} \mathbb{C}^{\alpha\beta\gamma\delta}\kappa_{\gamma\delta} \delta\kappa_{\alpha\beta} \, \sqrt{a}\text{ d}y = \int_\omega \delta p^α\eta_\alpha \, \sqrt{a}\text{ d}y
$$

### 2.3 Reissner-Mindlin / Naghdi shell
The Reissner-Mindlin kinematic assumption
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \bm{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\bm{d}(\xi^1,\xi^2)
$$


## References

- [1] An introduction to differential geometry with applications to elasticity
- [2] The finite-element analysis of shells -- fundamentals
