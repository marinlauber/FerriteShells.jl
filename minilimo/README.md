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
5. `limo_coupled_dynamic_strong_physio.jl`: `3` with the 0D coefficients calibrated against
   Regazzoni et al. (JCP 2022, Part II Tab. 4) and the actuation raised to 750 mmHg over a
   shorter systole. Targets physiological pressures and stroke volume — Pao 121/73, MAP 93,
   SV 70 mL — instead of the 101/41, MAP 67, SV 60 the uncalibrated coefficients give.
   Same solver and mesh; only coefficients, actuation and the Phase-2 initial 0D state differ.
   Writes to `minilimo-physio-*`. EF stays ~28% — that is the device's 221 mL unstressed
   volume, not the circulation model.

> Note: Godunov splitting ≡ Lie–Trotter. The genuine weak↔strong middle ground would be a
> sub-iterated partitioned scheme (fixed-point Plv↔Vlv, optionally under-relaxed/Aitken) — not yet implemented.

### Other setups

6. `limo_0d_surrogate.jl`: closed-loop **0D surrogate** of the coupled beat — the 3D shell
   replaced by a two-parameter law fitted to a finished run (`Plv = α·Pact + β·V + γ`).
   Reproduces a coupled run's hemodynamics in ~1 s, so Windkessel coefficients, actuation
   amplitude/timing and the Phase-2 initial state can be chosen *before* committing to a
   coupled run. Carries `fit_device_law` (refit from any `*-positions.jld2`), `run0d`,
   `report`, `si_table` (SI values to paste into the coupled file, next to the Regazzoni
   reference column) and `calibrate` (grid search with a hard *keep Plv positive*
   constraint). Run it directly for a demo: `julia --project=examples minilimo/limo_0d_surrogate.jl`.
   Read the limitations header first — it is quasi-static, linear, reads ~4.8 mmHg
   optimistic at the relaxation trough, and knows nothing about buckling.
7. `limo_dynamic_kostas_single.jl`: static inflation for Kostas' project. Runs well via command line Pact arguments.
8. `limo_dynamic_full.jl`: same as `limo_coupled_static_weak`, but with the full geometry, no symmetry plane.
9. `util.jl`: shared functions (mesh, edge-morph IC, RM assembly helpers) used across these files.

## old files

```julia
limo_dynamic_actuation.jl
limo_inflation.jl
limo_morph_bypass.jl
limo_ptc_inflation.jl
limo_dynamic_full.jl
limo_dynamic.jl
```