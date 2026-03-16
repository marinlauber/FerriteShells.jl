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

The key distinction between RM and Naghdi is which base bmtors appear in the strain measures: RM uses the **reference** base bmtors **A**_α (linearised around the reference configuration), while Naghdi replaces them with the **current** **a**_α everywhere, giving fully nonlinear strains. The director parametrisation (non-unit additive vs unit Rodrigues) is a separate but related choice — in practice the two always appear together.

Koiter has no director DOFs; the normal is always implicit from the surface geometry, so the Kirchhoff constraint (zero shear) is built in and C¹ continuity is required for bending.

Classical RM (additive director, ‖**d**‖≠1) is not implemented; the `_RM` functions go directly to the geometrically exact Naghdi form via Rodrigues parametrisation.

## 2. Shells kinematics

Curvilinear covariant basis bmtor (3D)
$$
\bm{g}_i=\frac{∂ \Phi(\xi^1,\xi^2,\xi^3)}{∂ \xi^i}, \quad i\in1,2,3
$$
where the latin indices run from 1 to 3. The **surface** covariant basis bmtor
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

The normal bmtor of the surface
$$
\hat{\bm{a}}_3 = \frac{\bm{a}_1 \times \bm{a}_2}{\|\bm{a}_1 \times \bm{a}_2\|}
$$

which satisfies $\bm{a}_3\cdot\bm{a}_\alpha=0$, and thus the second fundamental form of the surface is found
$$
b_{\alpha\beta} = \hat{\bm{a}_3}⋅ \bm{a}_{\alpha,\beta} = -\bm{a}_\alpha⋅ \hat{\bm{a}}_{3,\beta}
$$


### 2.1 Kirchhoff-Love / Koiter shell

The Kirchhoff-Love kinematic assumption prevents transverse shear strain by constraining the cross-section to remain normal the the shell's midsurface
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\hat{\bm{a}}_3(\xi^1,\xi^2)
$$


### 2.2 Reissner-Mindlin / Naghdi shell
The Reissner-Mindlin kinematic assumption
$$
\Phi(\xi^1,\xi^2,\xi^3) = \phi(\xi^1,\xi^2) + \xi^3\theta^\lambda(\xi^1,\xi^2) \bm{a}_\lambda(\xi^1,\xi^2) = \phi(\xi^1,\xi^2) + \xi^3\bm{d}(\xi^1,\xi^2)
$$


## References

- [1] An introduction to differential geometry with applications to elasticity
- [2] The finite-element analysis of shells -- fundamentals
