
# Linear and non-linear solvers

## Linear analysis

```julia
u = K \ r
```

or even better, by pre-allocating the factorization of the stiffness matrix, and then solving for different load cases

```julia
K_factor = factorize(K)
u = K_factor \ r
```

## Nonlinear analysis

### Newton–Raphson method

```julia
u = zeros(n_dofs)
residual = compute_residual(u)
while norm(residual) > tol
    tangent_stiffness = compute_tangent_stiffness(u)
    du = tangent_stiffness \ residual
    u += du
    residual = compute_residual(u)
end
```

### Load-controlled Newton-Raphson


### Displacement-controlled Newton-Raphson
<!-- https://doi.org/10.1016/j.compstruc.2021.106674 -->

The displacement-controlled Newton-Raphson method uses a bordering technique to enforce prescribed displacements at a selected node in the mesh. The pressure is then treated as an additional constrain on the system. The equilibrium is given by
```math
\begin{split}
\mathbf{r}(\mathbf{u},p) &= \mathbf{r}_\text{int}(\mathbf{u},p) - \lambda_p \mathbf{f}_\text{ext}(\mathbf{u}) = \mathbf{0}\\
u(\mathbf{x}_\text{T}) &= u_\text{target}
\end{split}
```
where ``\lambda_p`` is the load factor associated with the prescribed displacement target ``u_\text{target}`` at node ``\mathbf{x}_\text{T}``. ``K_\text{eff}=K_\text{int}-K_\text{pres}``is the effective stiffness matrix, where the unit-pressure stiffness is given by ``K_\text{pres}=\partial\ f_\text{int}(u)/\partial\mathbf{u}``.

The Newton-Raphson solution to this problem is then obtained in two steps
```math
\begin{split}
\mathbf{v}_1 &= K_\text{eff} \backslash (\lambda_p \mathbf{f}_\text{ext}-\mathbf{r}_\text{int})\\
\mathbf{v}_2 &= K_\text{eff} \backslash \mathbf{f}_\text{ext}
\end{split}
```
where the intermediat vector ``\mathbf{v}_1`` and ``\mathbf{v}_2`` are the equilibrium correction and the load direction vectors. From those, the pressure increment can be found
```math
\delta \lambda_p = \frac{u_\text{target} - u(\mathbf{x}_\text{T}) - \mathbf{v}_1(\mathbf{x}_\text{T})}{\mathbf{v}_2(\mathbf{x}_\text{T})}.
```
Here ``\lambda_p`` plays the role of a Lagrange multiplier — it's the unknown force that enforces the displacement constraint. The bordered 2×2 system:
```math
\begin{bmatrix}K_\text{eff} & -\mathbf{f}_\text{ext} \\ \mathbf{e}^{\top}_{wc} & 0\end{bmatrix}
\begin{bmatrix}\delta\mathbf{u}\\ \delta \lambda_p\end{bmatrix} =
\begin{bmatrix}\lambda_p \mathbf{f}_\text{ext} - \mathbf{r}_\text{int} \\ u_\text{target} - u(\mathbf{x}_\text{T})\end{bmatrix}=
\begin{bmatrix}\mathbf{r}\\ r_c\end{bmatrix}
```
is structurally identical to an augmented Lagrangian system. We can solve this system using Schur complement approach, which we can write explicitly since we have only one unknown. The Schur complement of the (2,2) block (which is 0) with respect to ``K_\text{eff}`` gives:
```math
  S = 0 - \mathbf{e}^{\top}_{wc} \cdot K_\text{eff}^{-1} \cdot (-\mathbf{f}_\text{ext}) = \mathbf{e}^{\top}_{wc} \cdot K_\text{eff}^{-1} \cdot \mathbf{f}_\text{ext} = u(\mathbf{x}_\text{T})
```
which then gives
```math
\begin{split}
\delta \lambda_p &= S^{-1} \cdot (r_c - \mathbf{e}^{\top}_{wc} \cdot K_\text{eff}^{-1} \cdot \mathbf{r}) = (r_c - \mathbf{v}_1(\mathbf{x}_\text{T})) / \mathbf{v}_2(\mathbf{x}_\text{T})\\
\delta \mathbf{u} &= K_\text{eff}^{-1} \cdot (\mathbf{r} + \mathbf{f}_\text{ext} \cdot \delta \lambda_p) = \mathbf{v}_1 + \delta \lambda_p \cdot \mathbf{v}_2
\end{split}
```
The Schur complement reduction costs exactly two triangular solves against the same factorisation —  which is optimal for a rank-1 augmentation.

### Arc-length method

[arc-length pdf](https://img1.wsimg.com/blobby/go/e35e0087-c3c0-4b15-a0c5-d8b4ee6b719d/downloads/ArcLength.pdf?ver=1748029264278#page=13.64)

### Dynamic Relaxation

https://www.sciencedirect.com/science/article/pii/S0263823111001777
https://www.sciencedirect.com/science/article/pii/0045794988903045


DR with kinematic damping approach eliminating the kinetic energy of the system when it reaches a peak. row-lumped mass matrix and a scaling parameters alpha for tunning the speed of convergence of the DR iterations.

## Time-varying analysis

### HHT-α method

Adding inertia M·ü regularizes the problem — the structure accelerates dynamically through the unstable branch rather than Newton stalling at the limit point. The tangent matrix becomes K_eff + (4/Δt²)·M (Newmark), which is better conditioned near the snap-through because the   mass term prevents the stiffness singularity from being reached.

!!! Warning
  the elastic wave speed c ∝ √(E/ρ)/t is very high for thin shells. For explicit time integration (central differences), the
   CFL condition gives a critical time step Δt_crit ~ h·t/L·(1/c) that is extremely small — potentially microseconds for a 2 mm thick shell. You'd need implicit time integration (Newmark/HHT-α) to use physiologically relevant time steps (~1 ms).

### Tip for solving non-convergence issues

The key diagnostic is whether the residual is:
  - Growing → wrong tangent (sign error, missing term)
  - Oscillating → conditioning or over-shooting
  - Slowly decreasing → step size issue (but you've ruled that out)
  - Blowing up on step 1 → issue at reference state

If you have isolated the residual diverging, then check:
  - If the residual is large, figure out which term dominates.
  - Check the tangent at the diverging state.
  - If the error is large at large deformations but small at ``u=0``, the geometric ``\phi\phi`` term (second derivative of Rodrigues director) is the suspect. Its effect is proportional to ``|\phi|`` and ``|f_\text{int}|``, both of which grow with deformation.