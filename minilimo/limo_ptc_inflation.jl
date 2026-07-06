using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))

# Prescribed (pressure-controlled) quasi-static inflation of the miniLIMO device on
# the rectangular multi-surface mesh (`make_minilimo_grid`, same geometry as
# `limo_inflation.jl` / `limo_prescribed_inflation.jl`).  This variant replaces the
# HHT-α dynamic integration with PSEUDO-TRANSIENT CONTINUATION (PTC): each load
# increment is solved as a static equilibrium
#
#     R(u) = r_int(u) − F_ext(u) = 0
#
# by marching the gradient flow  du/dτ = −R(u)  with backward Euler in a fictitious
# pseudo-time τ:
#
#     (M/Δτ + K(uₙ)) δu = −R(uₙ),   uₙ₊₁ = uₙ + δu
#
# The pseudo-mass M (the consistent shell mass matrix) is a positive-definite
# regulariser, NOT physical inertia: there is no velocity/acceleration state, no
# Rayleigh damping, no time-accuracy requirement.  Δτ is adapted by SER (switched
# evolution relaxation): small Δτ early (M/Δτ dominates → robust steepest-descent
# step through the difficult morph, even where K is indefinite), large Δτ late
# (M/Δτ → 0 → full Newton with quadratic convergence).  Unlike a fixed Newton
# direction with a line search, shrinking Δτ both shortens the step AND rotates it
# toward steepest descent, so it traverses the snap-prone morph region more
# robustly than the damped dynamic run.
#
# Three follower-pressure surfaces (as in limo_inflation.jl):
#   SRF_1: endocardium, Plv only (outward)
#   SRF_2: endocardium + actuator footprint, Plv (outward) and Pact (inward, opposing Plv)
#   SRF_3: actuator exterior, Pact only (outward)
#   F_ext(u) = Plv·F_plv + Pact·F_pact − Pact·F_plvpact
#     F_plv     on Plv_srf     = SRF_1 ∪ SRF_2
#     F_pact    on Pact_srf    = SRF_3
#     F_plvpact on PlvPact_srf = SRF_2
#
# Two-phase prescribed loading (the load parameter t drives morph + pressures; it
# is NOT physical time):
#   Phase 1 (t ∈ [0, T_morph]): morph the edge from flat to the elliptic arc while
#     ramping Plv → Plv0 and Pact → Pact0 (both smooth sinusoidal).
#   Phase 2 (t ∈ [T_morph, T_sim]): hold morph + Pact = Pact0 fixed, ramp Plv from
#     Plv0 → Plv1 (slow inflation).

# material
ρ   = 1200.0       # density [kg/m³] — only sets the scale of the pseudo-mass M
mat = LinearElastic(0.35e9, 0.3, 0.0002)

Np = 2
grid = make_minilimo_grid(;
    nx_left=2*3, nx_act=2*10, nx_right=2*3,
    ny_bot=2*1, ny_act=2*14, ny_top=2*2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=Np
)

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# pressure surfaces
Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv (outward)
Pact_srf    = getcellset(grid, "SRF_3")                              # +Pact (outward)
PlvPact_srf = getcellset(grid, "SRF_2")                              # −Pact (opposes Plv)

# pressure schedule (mmHg → Pa)
Pa2mmHg  = 0.00750062
m3_to_ml = 1.0e6
Plv0_mmHg = 6.0    # target Plv at end of phase 1
Pact_mmHg = 6.0   # target Pact at end of phase 1, held constant in phase 2
Plv1_mmHg = 20.0   # target Plv at end of phase 2 (slow ramp)
Plv0 = Plv0_mmHg / Pa2mmHg
Pact = Pact_mmHg / Pa2mmHg
Plv1 = Plv1_mmHg / Pa2mmHg

# two-phase load timeline (t is the load parameter, not physical time)
T_morph = 2.0   # phase 1: morph + reach (Plv0, Pact)
T_sim   = 6.0   # total; phase 2 = (T_morph, T_sim]: ramp Plv0 → Plv1
Δt      = 0.005 # initial load increment

cosramp(s) = 0.5 * (1 - cos(π * clamp(s, 0.0, 1.0)))
morph_ramp(t) = t < T_morph ? cosramp(t / T_morph) : 1.0
plv_schedule(t)  = t < T_morph ? Plv0 * cosramp(t / T_morph) :
                                 Plv0 + (Plv1 - Plv0) * cosramp((t - T_morph) / (T_sim - T_morph))
pact_schedule(t) = t < T_morph ? Pact * cosramp(t / T_morph) : Pact

prescribed_u = generate_boundary_function(grid, "edge"; ramp=morph_ramp)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
free  = ch.free_dofs

# PTC + load-stepping controls
tol      = 1e-4    # static residual tolerance ‖r_int − F_ext‖
max_iter = 60      # max PTC iterations per load step
Δτ0      = 1e-5    # initial pseudo-time step (strong M/Δτ regularisation)
Δτ_min   = 1e-12   # give up the inner Δτ shrink below this
Δτ_max   = 1e30    # effectively recovers Newton (M/Δτ → 0)
Δt_min   = 1e-6    # min load increment before aborting
Δt_max   = 0.5     # max load increment

K_int     = allocate_matrix(dh)
K_eff     = allocate_matrix(dh)
K_plv     = allocate_matrix(dh)
K_pact    = allocate_matrix(dh)
K_plvpact = allocate_matrix(dh)
M         = allocate_matrix(dh)
r_int     = zeros(N_dof)
F_plv     = zeros(N_dof)
F_pact    = zeros(N_dof)
F_plvpact = zeros(N_dof)
res       = zeros(N_dof)
δu        = zeros(N_dof)
u_trial   = zeros(N_dof)
rhs       = zeros(N_dof)
u_new     = zeros(N_dof)

# Precomputed shell-DOF maps (fixed for the run) and reusable element buffers,
# so the per-cell assembly loop allocates nothing.
n_e   = ndofs_per_cell(dh)
ke    = zeros(n_e, n_e)
re    = zeros(n_e)
u_e   = zeros(n_e)
sdofs = Vector{Vector{Int}}(undef, Ferrite.getncells(grid))
for cell in CellIterator(dh)
    sdofs[Ferrite.cellid(cell)] = shelldofs(cell)
end

assemble_mass!(M, dh, scv, ρ, mat)

assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat, sdofs, ke, re, u_e)
K_eff.nzval .= M.nzval ./ Δτ0 .+ K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

bufs = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
          res, rhs, δu, u_trial, F_lu, free, sdofs, ke, re, u_e)

# Pseudo-transient continuation solve of the static equilibrium for one load step.
# Prescribed pressures (p_plv, p_act) and Dirichlet morph are held fixed within the
# step (the inhomogeneous BC is set on `u_new` before the call).  `u_new` is updated
# in place; returns (converged, iters, Δτ).
#   R = r_int − F_ext;   F_ext = p_plv·F_plv + p_act·F_pact − p_act·F_plvpact
#   (M/Δτ + K) δu = −R
# Δτ adapts by SER: it grows by the residual reduction ratio each accepted step
# (→ Newton), and shrinks (more M/Δτ regularisation, steepest-descent step) if a
# trial step fails to reduce the residual.  The full tangent is assembled once per
# iteration; the Δτ sub-iteration re-uses the cheap residual-only assembly.
function ptc_step!(u_new, p_plv, p_act, dh, scv, mat, ch,
                   Plv_srf, Pact_srf, PlvPact_srf, bufs;
                   max_iter=60, tol=1e-4, Δτ0=1e-5, Δτ_min=1e-12, Δτ_max=1e30)
    (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
       res, rhs, δu, u_trial, F_lu, free, sdofs, ke, re, u_e) = bufs
    converged = false; iters = 0; Δτ = Δτ0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv,     F_plv,     dh, scv, u_new, Plv_srf,     sdofs, ke, re, u_e)
        assemble_pressure_region!(K_pact,    F_pact,    dh, scv, u_new, Pact_srf,    sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plvpact, F_plvpact, dh, scv, u_new, PlvPact_srf, sdofs, ke, re, u_e)
        @. res = r_int - (p_plv * F_plv + p_act * F_pact - p_act * F_plvpact)
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        # Δτ control: solve (M/Δτ + K)δu = −R; if the trial step does not reduce
        # the residual, shrink Δτ (more regularisation, step rotates toward
        # steepest descent) and re-solve.  No element re-assembly of the tangent.
        step_ok = false
        for _ in 1:12
            K_eff.nzval .= M.nzval ./ Δτ .+ (K_int.nzval .-
                           p_plv .* K_plv.nzval .- p_act .* K_pact.nzval .+ p_act .* K_plvpact.nzval)
            @. rhs = -res
            apply_zero!(K_eff, rhs, ch)
            lu!(F_lu, K_eff)
            ldiv!(δu, F_lu, rhs)
            @. u_trial = u_new + δu
            apply!(u_trial, ch)
            assemble_residual!(r_int, dh, scv, u_trial, mat, sdofs, re, u_e)
            assemble_pressure_residual!(F_plv,     dh, scv, u_trial, Plv_srf,     sdofs, re, u_e)
            assemble_pressure_residual!(F_pact,    dh, scv, u_trial, Pact_srf,    sdofs, re, u_e)
            assemble_pressure_residual!(F_plvpact, dh, scv, u_trial, PlvPact_srf, sdofs, re, u_e)
            @. res = r_int - (p_plv * F_plv + p_act * F_pact - p_act * F_plvpact)
            apply_zero!(res, ch)
            res_trial = norm(@views res[free])
            if res_trial ≤ res_norm
                Δτ = min(Δτ * (res_norm / max(res_trial, eps())), Δτ_max)  # SER growth
                step_ok = true
                break
            end
            Δτ /= 2  # too aggressive: add regularisation and re-solve
            Δτ < Δτ_min && break
        end
        u_new .= u_trial
        # Even at Δτ→Δτ_min the step did not reduce the residual: bail and let the
        # outer loop reject the load increment (halve Δt) instead of stalling.
        step_ok || break
    end
    return converged, iters, Δτ
end

# Initial state: flat reference geometry
u = zeros(N_dof); apply!(u, ch)

pvd = paraview_collection("minilimo-ptc-inflation")
vtk_step = Ref(0)
d, G3 = director_field(dh, scv, u)
VTKGridFile("minilimo-ptc-inflation-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

@printf("%-6s  %-8s  %-9s  %-10s  %-11s  %-10s  %-6s  %-10s  %-10s\n",
        "step", "t", "λ_morph", "Plv [mmHg]", "Pact [mmHg]", "Vlv [ml]", "iters", "Δt", "Δτ_end")

un = zeros(N_dof)
let t = 0.0; step = 0; Δt_cur = Δt
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_plv = plv_schedule(t_new)
    p_act = pact_schedule(t_new)

    # Predictor: previous converged displacement with the new morph BC applied.
    u_new .= u
    Ferrite.update!(ch, t_new)
    apply!(u_new, ch)

    converged, iters, Δτ_end = ptc_step!(u_new, p_plv, p_act, dh, scv, mat, ch,
                                         Plv_srf, Pact_srf, PlvPact_srf, bufs;
                                         max_iter=max_iter, tol=tol,
                                         Δτ0=Δτ0, Δτ_min=Δτ_min, Δτ_max=Δτ_max)

    if converged
        step += 1
        u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 2 == 0 || t ≥ T_sim - 1e-10 # ensures last step is written regardless
            Vlv = -2compute_volume(dh, scv, u; cellset=Plv_srf) * m3_to_ml
            vtk_step[] += 1
            d, G3 = director_field(dh, scv, u)
            VTKGridFile("minilimo-ptc-inflation-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[t] = vtk
            end
            @printf("%-6d  %-8.3f  %-9.4f  %-10.4f  %-11.4f  %-10.4f  %-6d  %-10.4e  %-10.4e\n",
                    step, t, morph_ramp(t), p_plv * Pa2mmHg, p_act * Pa2mmHg, Vlv, iters, Δt_cur, Δτ_end)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4))")
        @printf("  → step rejected at t=%.3f, Δt → %.4e\n", t, Δt_cur)
    end
end
    un .= u
end
close(pvd)

# using JLD2
# jldsave("minilimo_ptc_inflation.jld2"; u=un)
# Vlv_final = -2compute_volume(dh, scv, un; cellset=Plv_srf) * m3_to_ml
# @printf("PTC inflation complete. Final Vlv = %.4f ml at Plv = %.2f mmHg, Pact = %.2f mmHg\n",
#         Vlv_final, Plv1_mmHg, Pact_mmHg)
# println("Final state saved to minilimo_ptc_inflation.jld2")
