## miniLIMO setups

### 3D–0D Windkessel-coupled beat

The four coupled scripts are named `limo_coupled_<static|dynamic>_<weak|strong>.jl`, spanning a
2×2 design space — **structure inertia** (static equilibrium vs. dynamic HHT-α with `M·ä`) ×
**coupling strength** (weak vs. strong):

|                    | **weak** (Lie–Trotter split, black-box ODE integrator, Plv/Vlv lagged one substep) | **strong** (monolithic Newton on `(u, Plv, Pa, Pv)`, embedded implicit-Euler 0D) |
|--------------------|---|---|
| **static** structure  | `limo_coupled_static_weak.jl`  | `limo_coupled_static_strong.jl` |
| **dynamic** structure | `limo_coupled_dynamic_weak.jl` | `limo_coupled_dynamic_strong.jl` |

All four share the Phase-1 dynamic HHT-α morph (damped) and reload the morphed state from
`limo_dynamic_coupled_u0.jld2`; they differ only in the Phase-2 coupled beat.

1. `limo_coupled_static_weak.jl`: works, static weakly-coupled (Lie–Trotter) beat. Pact max reaches 400 mmHg. Carries `S_min`/`δp` diagnostics (Schur-complement collapse under refinement).
2. `limo_coupled_dynamic_weak.jl`: dynamic weakly-coupled (Lie–Trotter) beat; inertia + damping retained in Phase 2.
3. `limo_coupled_dynamic_strong.jl`: dynamic strongly (monolithically) coupled beat; HHT-α structure + implicit-Euler 0D as one Newton per step.
4. `limo_coupled_static_strong.jl`: static counterpart of `3` — quasi-static equilibrium + implicit-Euler 0D as one Newton (drops `M·ä`/damping, `v2 = K_eff⁻¹F_plv` with no `1−α` factor).

> Note: Godunov splitting ≡ Lie–Trotter. The genuine weak↔strong middle ground would be a
> sub-iterated partitioned scheme (fixed-point Plv↔Vlv, optionally under-relaxed/Aitken) — not yet implemented.

### Other setups

5. `limo_dynamic_kostas_single.jl`: static inflation for Kostas' project. Runs well via command line Pact arguments.
6. `limo_dynamic_full.jl`: same as `limo_coupled_static_weak`, but with the full geometry, no symmetry plane.
7. `util.jl`: shared functions (mesh, edge-morph IC, RM assembly helpers) used across these files.

## old files

```julia
limo_dynamic_actuation.jl
limo_inflation.jl
limo_morph_bypass.jl
limo_ptc_inflation.jl
limo_dynamic_full.jl
limo_dynamic.jl
```