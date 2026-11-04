# Nonlinear Reissner–Mindlin shell on Ferrite.jl PR #1544 (Design 3)

Prototype of a geometrically nonlinear RM shell kernel written against the local-frame
derivatives proposed in [Ferrite.jl#1544](https://github.com/Ferrite-FEM/Ferrite.jl/pull/1544),
"Design 3" (`b1616f4`, local frame stored in `CellValues`). It does not use FerriteShells.
The Ferrite version is pinned in `Project.toml` through `[sources]`.

## Formulation (`rm_d3.jl`)

- Fields `:u` and `:ω`, both `ip^3`, in Ferrite's field-blocked DOF layout (no `shelldofs`).
- `CellValues(qr, (u = ip^3, ω = ip^3), ip_geo^3; update_local_gradients = true)`; strains use
  `getlocalframe`, `shape_local_gradient`, `function_local_gradient`.
- `F = E + ∇ₛu`, director `d = Σ N_a R(ω_a)·D_a` (nodal director interpolation, objective),
  `ε = ½(FᵀF − I)`, `κ = sym(Fᵀ∇d) − κ₀`, `γ = Fᵀd − γ₀`, linear-elastic resultants.
- Tensors AD only for `∂d_a/∂ω_a` and `∂²d_a/∂ω_a²` per node; residual and tangent are explicit.
- The drilling rotation is removed by `AffineConstraint`s `ω_a·D_a = 0`, which makes the
  kinematics identical to FerriteShells' two-rotation nodal frames. `D_a` are area-weighted
  averaged nodal normals, built the same way as `NodeFrames`.

## Scripts

Run from this directory after `julia --project=. -e 'using Pkg; Pkg.instantiate()'`:

| Script | Checks |
|---|---|
| `test_fd.jl` | tangent vs central differences, symmetry, residual under a finite rigid rotation (Q4, Q9) |
| `roof_d3.jl` | Scordelis–Lo roof, Q4 and Q9, n = 4…32 |
| `rollup_d3.jl [n_steps]` | roll-up cantilever to a full 2π roll, plain Newton |
| `bench_d3.jl` | one Q9 element, residual + tangent |

`reference/` holds the matching FerriteShells runs; these use the package environment
(`julia --project=. dev/d3_prototype/reference/<script>.jl` from the repository root).

## Results (2026-11-04)

- Tangent matches finite differences to 7e-11 and is symmetric; rigid-rotation residual 1e-14.
- Scordelis–Lo: identical to FerriteShells with `NodeFrames` and no MITC to all 10 printed digits
  (Q9: 0.2653250778, 0.8133858951, 0.9809040622, 0.9953145495 for n = 4, 8, 16, 32).
- Roll-up cantilever, 200 steps: identical to FerriteShells to all 4 printed digits along the
  whole path; 3.7 Newton iterations per step.
- One Q9 element: 75 μs / 32 bytes vs FerriteShells 57 μs / 704 bytes (no MITC), with 54 vs 45 DOFs.
- `element!` is 66 lines.

## Known gaps

- Ferrite's `FacetValues` cannot `reinit!` on embedded surfaces (no `weighted_normal` for a 3×2
  jacobian), so `rollup_d3.jl` integrates the end moment by hand.
- Not yet covered: MITC, hyperelastic material, per-element directors at kinks,
  pressure/follower loads, mass matrix, Kirchhoff–Love.
