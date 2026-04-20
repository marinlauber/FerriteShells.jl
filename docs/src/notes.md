# Notes

## Arc-length for nonlinear shells

### Why arc-length can stall when load-controlled NR works

If the arc-length stalls near some ``\lambda`` but Newton–Raphson at fixed ``\lambda`` converges, the problem is usually in the **predictor**, not the corrector.

The predictor computes the tangent direction as

```math
\mathbf{v} = K_\text{eff}^{-1}\,\mathbf{f}_\text{ref}
```

which captures how the **free DOFs** respond to a unit increase in the load parameter, **with constrained DOFs held fixed**. If the boundary conditions also change with ``\lambda`` (e.g. a morphing ramp ``\mathbf{u}_\text{bc}(\lambda) = \lambda\,\Delta\mathbf{x}``), the true tangent is

```math
K_{ff}\,\frac{d\mathbf{u}_\text{free}}{d\lambda} = \mathbf{f}_\text{ref,free} - K_{fc}\,\Delta\mathbf{x}_\text{constrained}
```

The predictor misses the ``-K_{fc}\,\Delta\mathbf{x}`` coupling term. The corrector therefore starts with an ``O(\delta\lambda)`` residual instead of the usual ``O(\delta\lambda^2)`` truncation error. At small ``\lambda`` Newton absorbs this; at larger ``\lambda``, once 40–50 % of the morphing displacement has been applied, the coupling term dominates and the corrector can no longer converge.

NR works because it never makes a predictor assumption: it evaluates ``R(\mathbf{u}_\text{trial})`` at the exact state with all boundary conditions applied, so the coupling is always in the residual and corrected naturally.

### Arc-length variants for nonlinear shells

There are three mechanically distinct cases, differing only in how ``\mathbf{f}_\text{ref}`` is constructed.

| Scenario | ``\mathbf{f}_\text{ref}`` | Recompute each step? |
|---|---|---|
| Dead load (point load, gravity, traction) | ``\mathbf{F}_\text{dead}`` — fixed, state-independent | No |
| Follower load (pressure) | ``\mathbf{F}_\text{plv}(\mathbf{u})`` — state-dependent direction | Yes |
| Prescribed BC ramp | ``-K(\mathbf{u})\,\dot{\mathbf{u}}_\text{bc}`` — state-dependent via ``K`` | Yes |

For the BC ramp case, ``\dot{\mathbf{u}}_\text{bc}`` is the prescribed displacement per unit ``\lambda`` (precomputed once for a linear ramp). The reference load is obtained by forming the coupling contribution **before** applying `apply_zero!`:

```julia
mul!(f_ref, K_int, u_bc_dot)   # K * [0_free; dXs_constrained]
f_ref .*= -1.0                  # −K_{fc} * dXs on free rows
apply_zero!(K_eff, f_ref, ch)   # zero constrained rows
```

The corrector residual is then just the internal force (no external load term):

```julia
rhs .= .-r_int
apply_zero!(K_eff, rhs, ch)
```

Everything else — the two-solve corrector, the arc-length constraint, and the sign control — is identical across all three variants.

### Limit points vs bifurcation points

Arc-length handles **limit points** (snap-through, snap-back) automatically via sign control: the path folds back on itself and the method follows it. A **bifurcation point** is different — multiple equilibrium branches emanate from the same singular ``K``, and arc-length will follow one branch arbitrarily depending on numerical perturbations. Handling bifurcations requires:

1. detecting the singularity (monitor ``\det K`` or the minimum eigenvalue),
2. computing the critical eigenvector,
3. applying a symmetry-breaking perturbation in that direction (branch switching),
4. then re-entering the arc-length on the desired branch.

For typical cardiac shell problems the geometry and boundary conditions break symmetry sufficiently that bifurcation is unlikely; limit points are the relevant case.

### Dynamic stepping as an alternative for morphing

If the morphing phase involves a buckling-type instability (``K_{ff}`` singular), arc-length for the BC-driven problem is the rigorous fix. A practical alternative is to treat the morphing as a **dynamic step**. The dynamic effective tangent

```math
K_\text{eff} = K_\text{int} + \frac{1}{\beta\,\Delta t^2}\,M + \frac{\gamma}{\beta\,\Delta t}\,C
```

remains non-singular even when ``K_\text{int}`` is singular, because the mass matrix ``M`` contributes positive-definite terms. The inertia carries the structure through the limit point; damping dissipates the kinetic energy afterwards. If the ramp rate is slow relative to the natural frequencies and sufficient damping is used, the dynamic solution approximates the quasi-static one.

For shell problems use **implicit** time integration (Newmark, HHT-``\alpha``, generalised-``\alpha``). Explicit methods require a time step satisfying the CFL condition ``\Delta t \sim h\,t / (L\,c)`` where ``c \propto \sqrt{E/\rho}/t``, which for thin shells (``t = 2`` mm) gives microsecond time steps — impractical for physiological ramp times.
