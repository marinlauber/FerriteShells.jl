# FerriteShells.jl Memory

## Architecture
- Julia package extending Ferrite.jl for shell elements
- `src/FerriteShells.jl` — module, exports, includes
- `src/shellcellvalues.jl` — ShellCellValues type
- `src/kinematics.jl` — `kinematics(scv, qp, x, u_e::AbstractVector{T})` → flat vector API
- `src/material.jl` — LinearElastic, contravariant_elasticity, contravariant_bending_stiffness
- `src/assembly.jl` — KL and RM membrane/bending/shear residuals and tangents; explicit RM functions (`membrane_residuals_RM_explicit!`, `membrane_tangent_RM_explicit!`, `bending_residuals_RM_explicit!`, `bending_tangent_RM_explicit!`)
- `src/utils.jl` — shell_grid, assemble_traction!, assemble_pressure!, assemble_pressure_tangent!

## Key design decisions
- **Flat vector API**: All element functions take flat `AbstractVector{T}` for `u_e`. KL: 3 DOFs/node `[u₁,u₂,u₃]`. RM: 5 DOFs/node `[u₁,u₂,u₃,φ₁,φ₂]`. Essential for ForwardDiff.
- **KL bending via ForwardDiff**: `bending_residuals_KL!` / `bending_tangent_KL!` use `ForwardDiff.gradient/hessian` on `bending_energy_KL`. No manual tangent.
- **RM formulation**: `membrane_residuals_RM!`, `bending_residuals_RM!` (includes transverse shear) via ForwardDiff on `rm_membrane_energy` / `rm_bending_shear_energy`. Director: `d_I = cos(|φ|)·G₃ + sinc(|φ|)·(φ₁T₁+φ₂T₂)` (geometrically exact Rodrigues, |d_I|=1). Shear correction κ_s = 5/6.
- **Rodrigues director**: Replaces additive `d = G₃+φ₁T₁+φ₂T₂`. Formula: `d_I = cosθ·G₃ + sincθ·(φ₁T₁+φ₂T₂)` where `θ=√(φ₁²+φ₂²)`. Matches additive at first order, unit length exactly. ForwardDiff-safe via `_cos_sinc_sq(θ²)` helper that avoids `norm` (which gives 0/0 gradient at φ=0) by using Taylor series for θ²<1e-6 and `(cos(√θ²), sin(√θ²)/√θ²)` otherwise. Error vs analytical: <0.01% at 20° (vs 5.7% with additive).
- **Function naming**: KL suffix = Kirchhoff-Love (3 DOFs/node, no shear), RM suffix = Reissner-Mindlin (5 DOFs/node, with shear).
- **Q9 for full bending**: Q4 only captures twist (κ₁₂), Q9 gives full curvature tensor. Use `Lagrange{RefQuadrilateral, 2}()` + `QuadraticQuadrilateral` grid.
- **Hessian via reference space**: Uses `reference_shape_hessian_gradient_and_value` (not `shape_hessian` from CellValues, which fails for embedded shells).
- **FacetValues workaround**: `FacetValues` fails for embedded shells (sdim mismatch). Use `assemble_traction!(f, dh, facetset, ip, fqr::FacetQuadratureRule, traction)` instead.
- **Explicit RM membrane residual**: `membrane_residuals_RM_explicit!` computes `r_I = ∫ N^{αβ} ∂N_I^α a_β dΩ` by precomputing `P_α = N^{αβ} a_β` once per QP, avoiding redundant inner loops.
- **Explicit RM membrane tangent**: `membrane_tangent_RM_explicit!` splits into material part `K^mat_IJ = ∂N_I^α ∂N_J^δ M_{αδ}` and geometric part `(∂N_I^α N^{αβ} ∂N_J^β) I₃`. Helper `_frame_stiffness(C, a₁, a₂)` precomputes `M_{αδ} = C^{αβγδ} a_β⊗a_γ` (3 unique `Tensor{2,3}` per QP, with `M₂₁ = transpose(M₁₂)` by C symmetry).
- **B-matrix approach rejected**: The user explicitly prefers the frame-stiffness / M-tensor approach over B-matrix (Voigt) formulations, as it maps directly to index notation without Voigt bookkeeping.
- **Explicit RM bending residual**: `bending_residuals_RM_explicit!` uses displacement DOFs `r_I^u = (∂₁N_I P¹ + ∂₂N_I P²)dΩ` with `P^α = M^{αβ}d_{,β} + Q^α d`; rotation DOFs `r_{I,k}^φ = F_I·dd_{Ik}dΩ` with `F_I = ∂₁N_I S¹ + ∂₂N_I S² + N_I(Q₁a₁+Q₂a₂)`. Rodrigues Jacobian `dd_{Ik}` uses `_cos_sinc_sincc_sq`.
- **Explicit RM bending tangent**: `bending_tangent_RM_explicit!` has 4 blocks: uu uses `frame_stiffness(D, d₁, d₂)` + `q_IJ(d⊗d)`; uφ computed explicitly, φu filled from uφ transpose in same (I,J) iteration; φφ has material part `δF_I·dd_{Ik}` plus geometric part `F_I·∂²d_I/∂φ_k∂φ_l` (only diagonal J=I blocks). Second Rodrigues derivative uses `sccc = (-sinc-3scc)/θ²` (Taylor at θ²→0: 1/15).

## Test files
- `test/runtests.jl` — main test runner
- `test/test_bending.jl` — KL bending tests (Q9, symmetry, FD consistency)
- `test/test_rm.jl` — RM tests (FD consistency, patch, Kirchhoff limit, cantilever, curved geometry, SS plate convergence)
- `test/test_utils.jl` — shelldofs reordering, assemble_traction! regression
- `test/test_plate.jl` — KL bending energy h-convergence (projects sin(πx)sin(πy) mode)
- `test/test_benchmarks.jl` — Scordelis-Lo RM and Pinched cylinder RM convergence tests

## Benchmark results
- **Scordelis-Lo RM** (ref -0.3024): 4×4→-0.080, 8×8→-0.246, 16×16→-0.297 (1.8% error). Rates ≥ 1.5. ✓
- **Pinched cylinder RM** (ref -1.8248e-5): 8×8→-1.03e-5, 16×16→-1.66e-5 (9.2% error), 32×32→-1.82e-5 (99.5%). Requires rotation BCs at symmetry planes.
- **Pinched cylinder rotation BCs**: φ₁=0 at θ=0 and θ=π/2 (sym_theta0, sym_theta90); φ₂=0 at x=L/2 (sym_axial). Without these, diverges non-monotonically.
- **KL on curved shells**: Fails both benchmarks. C0 Q9 bending is mathematically correct per-element but lacks inter-element normal continuity (needs C1/DKQ/NURBS). KL works for flat shells only.
- **Pinched hemisphere RM** (ref |u_x(A)| = 0.0924, P=1): 4→-0.002, 8→-0.021, 16→-0.055, 32→-0.062. Convergence rate ≈ O(h^0.3) — severe membrane locking. Q9 RM without MITC is stalled for this bending-dominated benchmark (t/R=0.004). Confirmed BCs correct (T₂=ê_y at φ=0 → fix φ₂; T₂=−ê_x at φ=π/2 → fix φ₂). MITC required for practical accuracy.

## Utils
- `apply_pointload!(f, dh, nodeset_name, load::Vec{3})` — applies point load to :u DOFs at a named nodeset. Uses getnodes(cell) and tracks processed nodes with Set{Int} to avoid double-counting. Works for single-field and two-field DofHandlers.

## Nonlinear solver notes
- **Energy Armijo for shells**: For geometrically nonlinear shells, residual-norm Armijo fails because the Newton step from a flat reference introduces large nonlinear membrane strains (spurious residual in u_x DOFs) before u_x relaxes. The correct merit function is Π = E_int - F·u (total potential). Newton direction is a descent direction for Π when K is PD. Use slope = du'*rhs = du'*K*du for the sufficient-decrease condition.
- **RM dead-load moment**: Apply as constant force to φ₁ DOFs: `fe[3n+2I-1] -= m*NI*dΓ` (negative sign because φ₁>0 → bending downward; must negate for upward-bending moment). The constant-force approximation matches dead-load moment only at α=0; introduces O(α²) error at large rotations.
- **RM director limitation**: With d = G₃ + φ₁T₁ + φ₂T₂ (additive, not unit-length), the formulation loses accuracy for α > ~10-15°. At 10°: ~1.3% error in u_z; at 20°: ~5.7% error. Geometrically exact (Rodrigues) directors needed for large rotations.
- **Load steps**: Use ≥50 steps for n=50 to keep each increment ~0.4° (Newton converges in 3-4 iterations with α_ls=1). Too few steps (n=20) → 10+ Newton iterations as energy Armijo halves α_ls repeatedly.
- **RM roll-up example**: `examples/RollupCantilever_RM.jl` — Sze 2004 Problem 1, 50 steps to α=20°, energy Armijo, exactly 3 Newton iters/step, <0.03% error at 20° (Rodrigues director). Limit is |φ|<π (180°); total Lagrangian update needed for full 360° roll-up.
- **Square airbag example**: `examples/SquareAirbag_RM.jl` — flat Q9 RM plate [0,L/2]², SS+symmetry BCs, follower pressure. Solved with displacement-controlled bordering method: prescribe w_center = step·Δw, treat p as unknown. Bordering Newton: v₁=K_eff⁻¹(−R), v₂=K_eff⁻¹F_p, δp=(w_target−u[w_c]−v₁[w_c])/v₂[w_c]. Load-controlled NR is infeasible for t/L=10⁻³ (bending-dominated flat start, condition number ~10¹²). Reaches p=500 in 61 steps, 3–5 Newton iters/step.
- **Displacement-controlled bordering**: For follower-pressure problems starting from flat reference, load-controlled NR cannot converge. Bordering (= bordered Newton with displacement constraint) is the correct approach. K_eff = K_int − p·K_pres where K_pres = ∂F_p/∂u is the follower-pressure load-stiffness assembled via ForwardDiff.jacobian.
- **Linear solver pattern**: For Newton on a fixed mesh, use `lu!(F_lu, K_eff)` (3-arg form) to refactorise numerically while reusing the symbolic analysis from the initial `lu(K_eff)`. Update K_eff values in-place: `K_eff.nzval .= K_int.nzval .- p .* K_pres.nzval` (valid because all three share the same sparsity pattern from `allocate_matrix(dh)`). Then `ldiv!(v1, F_lu, rhs1); ldiv!(v2, F_lu, F_p)` for two back-substitutions. Buggy pattern to avoid: `lu!(K_eff)` with discarded return value followed by `ldiv!(v1, K_eff, ...)` — this refactorises on every ldiv! call.
- **Assembly is the bottleneck**: ForwardDiff Hessian/Jacobian calls dominate runtime (~128s for n=8, 61 steps). Linear solves on ~3000 DOFs are negligible. Fix paths: (1) explicit tangent expressions, (2) DiffResults pre-allocated buffers, (3) Enzyme.jl, (4) threaded assembly.

## User preferences
- No separator comments (`# ---`, `# ===`)
- Implicit return, @views, @inbounds where safe, @inline
- Single @testset with multiple @test statements
- Function args on single line unless very long
