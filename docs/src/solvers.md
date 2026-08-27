
# Linear and non-linear solvers

The snippets below show the **structure** of each solver — boundary conditions, the load/time loop, the Newton update and the state update. The element assembly is hidden behind a single `assemble_global!` helper so the focus stays on the solver. Fully runnable versions of every method live under [`examples/`](https://github.com/marinlauber/FerriteShells.jl/tree/master/examples).

## 1.0 Common setup

Every solver shares the same problem definition: a grid, a `DofHandler` with the displacement field `:u` (dim 3) and rotation field `:θ` (dim 2), a `ConstraintHandler` for the Dirichlet BCs, and a preallocated sparse tangent and residual.

```julia
using FerriteShells, LinearAlgebra

# grid, material and shell values (see any example for the details)
mat = LinearElastic(1.2e6, 0.0, 0.1)          # E, ν, thickness
ip  = Lagrange{RefQuadrilateral, 2}()
scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(3), ip, ip; mitc=MITC9)

# degrees of freedom: 3 displacements + 2 rotations per node
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# Dirichlet BCs — clamp one edge (u and θ fixed)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# preallocate the system
K = allocate_matrix(dh)
r = zeros(ndofs(dh))
u = zeros(ndofs(dh))

# fills K (tangent) and r (internal residual) for the current guess `u`
function assemble_global!(K, r, dh, scv, u, mat)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]                 # node-major element DOFs
        # ke, re ← membrane_*_RM!(…) + bending_*_RM!(…)   (see examples)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end
```

!!! note
    `apply_zero!(K, r, ch)` enforces the BCs on the Newton *increment* (zero on prescribed DOFs), while `apply!(u, ch)` writes the prescribed values into the solution. Convergence is measured on the free DOFs, `norm(r[ch.free_dofs])`.

## 1.1 Linear analysis

For a linear problem the tangent is constant and one solve suffices:

```julia
u = K \ r
```

or, when solving several load cases against the same operator, factor once and reuse:

```julia
K_factor = factorize(K)
u = K_factor \ r
```

## 1.2 Nonlinear analysis

### 1.2.1 Newton–Raphson method

The plain Newton loop reassembles the tangent and residual at each iteration and updates until the out-of-balance force vanishes:

```julia
while true
    assemble_global!(K, r, dh, scv, u, mat)   # r = r_int(u)
    r .-= f_ext                                # out-of-balance = r_int − f_ext
    apply_zero!(K, r, ch)                      # BCs on the increment
    norm(r[ch.free_dofs]) < tol && break
    u .-= K \ r                                # Newton update
    apply!(u, ch)                              # keep prescribed DOFs satisfied
end
```

### 1.2.2 Load-controlled Newton-Raphson

For large deformations the external load is ramped through a load factor ``\lambda \in [0,1]``, using the converged solution of one step as the initial guess for the next:

```julia
for λ in 0.2:0.2:1.0                            # ramp the load 0 → 1
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)
        r .-= λ .* f_ext                        # scaled external load
        apply_zero!(K, r, ch)
        norm(r[ch.free_dofs]) < tol && break
        u .-= K \ r
        apply!(u, ch)
    end
    save_step(u, λ)                             # write VTK / record tip deflection
end
```

!!! tip
    Near limit points a full Newton step can overshoot. Scaling the increment with an energy-based Armijo line search (accept the largest ``\alpha\le 1`` for which the total potential ``\Pi = E_\text{int} - \mathbf{f}\cdot\mathbf{u}`` decreases sufficiently) makes the load-controlled solver far more robust, see `examples/RollupCantilever.jl`.

### 1.2.3 Displacement-controlled Newton-Raphson
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
Here ``\lambda_p`` plays the role of a Lagrange multiplier — it's the unknown force that enforces the displacement constraint. The bordered ``2×2`` system:
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

In code, the two triangular solves reuse a single factorization (`ldiv!` against `F_lu`), and the load factor `p` is updated alongside the displacement:

```julia
u = zeros(ndofs(dh)); p = 0.0                   # p = λ_p (unknown load factor)
ctrl = control_dof                              # DOF whose value is prescribed
F_lu = lu(K)                                    # symbolic factorization (pattern)

for step in 1:n_steps
    u_target = step / n_steps * u_max           # prescribed value at `ctrl`
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)             # K_int, r_int
        assemble_pressure_tangent!(K_p, f_ext, scv, u, dh)  # follower-load stiffness
        K_eff.nzval .= K.nzval .- p .* K_p.nzval            # effective tangent
        rhs = p .* f_ext .- r                               # out-of-balance
        apply_zero!(K_eff, rhs, ch)
        (norm(rhs[ch.free_dofs]) < tol && abs(u[ctrl]-u_target) < tol) && break
        lu!(F_lu, K_eff)                        # refactorize in place
        ldiv!(v1, F_lu, rhs)                    # v₁: equilibrium correction
        ldiv!(v2, F_lu, f_ext)                  # v₂: load direction
        δp = (u_target - u[ctrl] - v1[ctrl]) / v2[ctrl]     # Schur complement
        u .+= v1 .+ δp .* v2
        p  += δp
        apply!(u, ch)
    end
end
```

### 1.2.4 Arc-length method

<!-- https://img1.wsimg.com/blobby/go/e35e0087-c3c0-4b15-a0c5-d8b4ee6b719d/downloads/ArcLength.pdf?ver=1748029264278#page=13.64 -->

The arc-length (Riks/Crisfield) method treats the load factor ``\lambda`` as an unknown and constrains the combined step ``(\Delta\mathbf{u},\Delta\lambda)`` to lie on a sphere of radius ``\Delta l``, so the solver can traverse limit points and snap-through where load control fails. Each iteration splits the update into an equilibrium part and a load-direction part, then picks ``\delta\lambda`` to satisfy the constraint:

```julia
u = zeros(ndofs(dh)); λ = 0.0
for step in 1:n_steps
    Δu = zero(u); Δλ = 0.0                       # incremental unknowns for this step
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u .+ Δu, mat)
        g = r .- (λ + Δλ) .* f_ext               # residual at trial state
        apply_zero!(K, g, ch)
        norm(g[ch.free_dofs]) < tol && break
        F = lu(K)
        δu_g = F \ (-g)                           # equilibrium correction
        δu_t = F \ f_ext                          # tangent to the load path
        # spherical constraint ‖Δu+δu‖² + ψ²‖f‖²(Δλ+δλ)² = Δl²  → quadratic in δλ
        δλ = arclength_root(Δu, Δλ, δu_g, δu_t, Δl, ψ)
        Δu .+= δu_g .+ δλ .* δu_t
        Δλ += δλ
    end
    u .+= Δu; λ += Δλ                             # commit the converged increment
end
```

### 1.2.5 Dynamic Relaxation

<!-- https://www.sciencedirect.com/science/article/pii/S0263823111001777 -->
<!-- https://www.sciencedirect.com/science/article/pii/0045794988903045 -->

Dynamic relaxation drives the static solution by integrating a fictitious damped dynamics with a row-lumped (diagonal) mass. The *kinematic damping* variant carries no viscous term: the velocity is reset to zero whenever the kinetic energy peaks, which bleeds energy out of the system and converges to the static equilibrium. A scaling parameter ``\alpha`` tunes the fictitious time step / mass and hence the convergence speed. Only the residual is needed — no tangent factorization:

```julia
u = zeros(ndofs(dh)); v = zero(u)
Mlump = lumped_mass(dh, scv, ρ, mat)             # diagonal, row-summed mass
KE_prev = Inf
for it in 1:max_iter
    assemble_residual!(r, dh, scv, u, mat)       # residual only
    r .-= f_ext
    apply_zero!(r, ch)
    norm(r[ch.free_dofs]) < tol && break
    v .+= α .* Δt .* (-r) ./ Mlump               # explicit velocity update (a = −r/M)
    KE = 0.5 * dot(v .* Mlump, v)                 # kinetic energy
    KE < KE_prev && (v .= 0.0)                    # peak passed → kinetic damping
    u .+= Δt .* v
    apply!(u, ch)
    KE_prev = KE
end
```

## 1.3 Time-varying analysis

### 1.3.1 HHT-α method

Adding inertia ``M·ü`` regularizes the problem — the structure accelerates dynamically through the unstable branch rather than Newton stalling at the limit point. The tangent matrix becomes ``K_eff + (4/Δ t^2)\cdot M`` (Newmark), which is better conditioned near the snap-through because the   mass term prevents the stiffness singularity from being reached.

The HHT-α scheme applies a Newmark predictor, then Newton-corrects the α-weighted residual ``R = M\ddot{u} + (1-\alpha)\,r_\text{int}(u_{n+1}) + \alpha\,r_\text{int}(u_n) - f_\text{ext}``. The mass is assembled once; the effective tangent combines mass and stiffness:

```julia
α = -0.05; γ = 0.5 - α; β = (1 - α)^2 / 4         # HHT-α parameters (α ∈ [−1/3, 0])
u = zeros(ndofs(dh)); v = zero(u); a = zero(u); r_old = zero(u)
assemble_mass!(M, dh, scv, ρ, mat)                # constant → assemble once

for step in 1:n_steps
    # Newmark predictor (advance kinematics without equilibrium)
    ũ = u .+ Δt .* v .+ (Δt^2 * (0.5 - β)) .* a
    ṽ = v .+ (Δt * (1 - γ)) .* a
    u_new = copy(ũ); apply!(u_new, ch)
    for iter in 1:max_iter
        assemble_global!(K, r_int, dh, scv, u_new, mat)
        a_new = (u_new .- ũ) ./ (β * Δt^2)                       # Newmark acceleration
        R = M * a_new .+ (1-α) .* r_int .+ α .* r_old .- f_ext   # HHT residual
        apply_zero!(R, ch)
        norm(R[ch.free_dofs]) < tol && break
        K_eff.nzval .= M.nzval ./ (β * Δt^2) .+ (1-α) .* K.nzval # effective tangent
        apply_zero!(K_eff, R, ch)
        u_new .-= K_eff \ R
        apply!(u_new, ch)
    end
    # commit state: acceleration, velocity, and r_int(u_n) for the next α-weight
    a .= (u_new .- ũ) ./ (β * Δt^2)
    v .= ṽ .+ (Δt * γ) .* a
    r_old .= r_int
    u .= u_new
end
```

!!! note "Why implicit?"
    The scheme above is **implicit** — each step factorizes ``K_\text{eff}`` and Newton-iterates, so it is unconditionally stable and the time step is chosen for accuracy, not stability. This matters for thin shells because the elastic wave speed ``c \propto \sqrt{E/\rho}/t`` is very high: an *explicit* scheme (central differences, no tangent solve) would be limited by the CFL condition to a critical step ``\Delta t_\text{crit} \sim (h\,t/L)/c`` — potentially microseconds for a 2 mm shell. Implicit HHT-α lets you take physiologically relevant steps (``\sim``1 ms) instead.

## 1.4 Tip for solving non-convergence issues

The key diagnostic is whether the residual is:
  - Growing → wrong tangent (sign error, missing term)
  - Oscillating → conditioning or over-shooting
  - Slowly decreasing → step size issue (but you've ruled that out)
  - Blowing up on step 1 → issue at reference state

If you have isolated the residual diverging, then check:
  - If the residual is large, figure out which term dominates.
  - Check the tangent at the diverging state.
  - If the error is large at large deformations but small at ``u=0``, the geometric ``\phi\phi`` term (second derivative of Rodrigues director) is the suspect. Its effect is proportional to ``|\phi|`` and ``|f_\text{int}|``, both of which grow with deformation.
