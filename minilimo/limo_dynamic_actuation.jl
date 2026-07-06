using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))

# Two-phase dynamic (HHT-α) miniLIMO simulation on the rectangular multi-surface
# mesh built by `make_minilimo_grid` (same geometry as `limo_dynamic.jl`).  Two
# separate implicit time-integration loops share the carried-over state (u, v, a,
# g_old):
#
#   Phase 1 — morph (identical to `limo_dynamic.jl`):
#       edge nodes are driven from the flat reference to the elliptic arc with a
#       smooth sinusoidal ramp while a follower Plv pressure fills the endocardium
#       (SRF_1 ∪ SRF_2).  End state: fully morphed, Plv = p_hold.
#
#   Phase 2 — actuation:
#       Plv is held constant at p_hold and the actuator pressure Pact is ramped
#       0 → 40 mmHg over T_act = 10 s (smooth sinusoidal).  The follower load uses
#       all three surfaces
#           F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact
#       SRF_1: endocardium, Plv only (outward)
#       SRF_2: endocardium + actuator footprint, Plv (outward) and Pact (inward)
#       SRF_3: actuator exterior, Pact only (outward)
#
# Both phases use the same HHT-α integrator with mass-proportional Rayleigh
# damping C = α_damp·M, adaptive Δt, and a backtracking line search:
#   g(u,v) = C·v + r_int(u) − F_ext(u)
#   R = M·ä_{n+1} + (1−α)·g_{n+1} + α·g_old = 0
#   γ = ½ − α,  β = (1−α)²/4   (2nd-order, unconditionally stable for α ∈ [−⅓,0])
#   K_eff = M·[1/(βΔt²) + (1−α)·α_damp·γ/(βΔt)] + (1−α)·(K_int − ∂F_ext/∂u)

# material
ρ   = 1200.0       # density [kg/m³]
# mat = LinearElastic(0.35e9, 0.3, 0.0002) # nylon-cpated TPU
mat = LinearElastic(20e6, 0.3, 0.001) # soft TPU
@show mat
Np = 3
grid = make_minilimo_grid(;
    nx_left=3*3, nx_act=3*10, nx_right=3*3,
    ny_bot=3*1, ny_act=3*14, ny_top=3*2,
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

# The three follower-pressure surfaces.
# SRF_1: endocardium, Plv only (outward)
# SRF_2: endocardium + actuator, Plv (outward) and Pact (inward, opposing Plv)
# SRF_3: actuator exterior, Pact only (outward)
Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv acts here
Pact_srf    = getcellset(grid, "SRF_3")                              # +Pact
PlvPact_srf = getcellset(grid, "SRF_2")                              # −Pact (opposes Plv)

# Smooth sinusoidal ramp: λ(t) = ½(1 − cos(πt/T_morph)) for t ≤ T_morph, 1 beyond.
T_morph = 2.0   # morphing duration [s]
T_sim   = 2.0   # Phase-1 (morph) simulation duration [s]
Δt      = 0.001 # initial time step [s]
ramp(t) = t < T_morph ? 0.5 * (1 - cos(π * t / T_morph)) : 1.0

prescribed_u = generate_boundary_function(grid, "edge"; ramp=ramp)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
free  = ch.free_dofs

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable)
α_hht   = -0.3
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 10.0    # mass-proportional Rayleigh damping coefficient [1/s]
tol      = 1e-4
max_iter = 50
Δt_min   = 1e-7
Δt_max   = 0.1

Pa2mmHg = 0.00750062       # Pa/mmHg
p_hold   = 6.0  / Pa2mmHg  # Plv held constant in Phase 2 [Pa] (6 mmHg)
Pact_max = 400.0 / Pa2mmHg  # actuator pressure target [Pa] (40 mmHg)
T_act    = 10.0            # actuator ramp duration [s]
ramp_act(t) = t < T_act ? 0.5 * (1 - cos(π * t / T_act)) : 1.0

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
g_old     = zeros(N_dof)
res       = zeros(N_dof)
δu        = zeros(N_dof)
u_trial   = zeros(N_dof)
rhs       = zeros(N_dof)
a_new     = zeros(N_dof)
v_new     = zeros(N_dof)
Ma        = zeros(N_dof)
Mv        = zeros(N_dof)
ũ         = zeros(N_dof)
ṽ         = zeros(N_dof)
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

m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat, sdofs, ke, re, u_e)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

bufs = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
          res, rhs, δu, u_trial, a_new, v_new, Ma, Mv, F_lu, free, g_old,
          sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# HHT-α Newton corrector (with backtracking line search) for one morph time step.
# `u_new` is updated in place; returns (converged, iters). Plv pressure acts on
# `Plv_srf` only.  All vector arithmetic is in-place / mul!-based.
function solve_morph_step!(u_new, ũ, ṽ, p_new, Δt, dh, scv, mat, ch, Plv_srf, bufs; max_iter=20, tol=1e-4)
    (; K_int, r_int, K_plv, F_plv, M, K_eff, res, rhs, δu, u_trial, a_new, v_new,
       Ma, Mv, F_lu, free, g_old, sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv, F_plv, dh, scv, u_new, Plv_srf, sdofs, ke, re, u_e)
        @. a_new = (u_new - ũ) / (β_hht * Δt^2)
        @. v_new = ṽ + (Δt * γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new)
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int - p_new * F_plv) + α_hht * g_old
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        K_eff.nzval .= M.nzval .* mfac .+ (1 - α_hht) .* (K_int.nzval .- p_new .* K_plv.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        α_ls = 1.0; ls_ok = false
        for _ in 1:8
            @. u_trial = u_new + α_ls * δu
            apply!(u_trial, ch)
            assemble_residual!(r_int, dh, scv, u_trial, mat, sdofs, re, u_e)
            assemble_pressure_residual!(F_plv, dh, scv, u_trial, Plv_srf, sdofs, re, u_e)
            @. a_new = (u_trial - ũ) / (β_hht * Δt^2)
            @. v_new = ṽ + (Δt * γ_hht) * a_new
            mul!(Ma, M, a_new); mul!(Mv, M, v_new)
            @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int - p_new * F_plv) + α_hht * g_old
            apply_zero!(res, ch)
            (norm(@views res[free]) ≤ res_norm) && (ls_ok = true; break)
            α_ls /= 2
        end
        u_new .= u_trial
        # Line search stalled (even α_ls=1/256 did not reduce the residual): the
        # step is not making progress, so bail and let the outer loop reject it
        # (halve Δt) instead of burning all max_iter full-tangent sweeps.
        ls_ok || break
    end
    return converged, iters
end

# HHT-α Newton corrector (with backtracking line search) for one actuation time
# step.  Plv is held constant; Pact ramps.  Full three-surface follower load
# F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact acts, with the follower tangent
# K_int − Plv·K_plv − Pact·K_pact + Pact·K_plvpact.  `u_new` updated in place.
function solve_actuation_step!(u_new, ũ, ṽ, Plv, Pact, Δt, dh, scv, mat, ch,
                               Plv_srf, Pact_srf, PlvPact_srf, bufs; max_iter=20, tol=1e-4)
    (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
       res, rhs, δu, u_trial, a_new, v_new, Ma, Mv, F_lu, free, g_old,
       sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv,     F_plv,     dh, scv, u_new, Plv_srf,     sdofs, ke, re, u_e)
        assemble_pressure_region!(K_pact,    F_pact,    dh, scv, u_new, Pact_srf,    sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plvpact, F_plvpact, dh, scv, u_new, PlvPact_srf, sdofs, ke, re, u_e)
        @. a_new = (u_new - ũ) / (β_hht * Δt^2)
        @. v_new = ṽ + (Δt * γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new)
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int - Plv * F_plv - Pact * F_pact + Pact * F_plvpact) + α_hht * g_old
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        K_eff.nzval .= M.nzval .* mfac .+ (1 - α_hht) .* (K_int.nzval .- Plv .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        α_ls = 1.0; ls_ok = false
        for _ in 1:8
            @. u_trial = u_new + α_ls * δu
            apply!(u_trial, ch)
            assemble_residual!(r_int, dh, scv, u_trial, mat, sdofs, re, u_e)
            assemble_pressure_residual!(F_plv,     dh, scv, u_trial, Plv_srf,     sdofs, re, u_e)
            assemble_pressure_residual!(F_pact,    dh, scv, u_trial, Pact_srf,    sdofs, re, u_e)
            assemble_pressure_residual!(F_plvpact, dh, scv, u_trial, PlvPact_srf, sdofs, re, u_e)
            @. a_new = (u_trial - ũ) / (β_hht * Δt^2)
            @. v_new = ṽ + (Δt * γ_hht) * a_new
            mul!(Ma, M, a_new); mul!(Mv, M, v_new)
            @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int - Plv * F_plv - Pact * F_pact + Pact * F_plvpact) + α_hht * g_old
            apply_zero!(res, ch)
            (norm(@views res[free]) ≤ res_norm) && (ls_ok = true; break)
            α_ls /= 2
        end
        u_new .= u_trial
        ls_ok || break
    end
    return converged, iters
end

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

pvd = paraview_collection("minilimo-dynamic-actuation")
vtk_step = Ref(0)
resu = zeros(3, getnnodes(dh.grid))
resθ = zeros(2, getnnodes(dh.grid))

function write_vtk!(pvd, vtk_step, dh, scv, grid, u, res, resu, resθ, t)
    vtk_step[] += 1
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        for (I, nid) in enumerate(cell.nodes)
            resu[:, nid] .= res[sd[5I-4:5I-2]]
            resθ[:, nid] .= res[sd[5I-1:5I  ]]
        end
    end
    d, G3 = director_field(dh, scv, u)
    VTKGridFile("minilimo-dynamic-actuation-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u)
        Ferrite.write_node_data(vtk, resu, "ru")
        Ferrite.write_node_data(vtk, resθ, "rθ")
        Ferrite.write_node_data(vtk, d,  "director")
        Ferrite.write_node_data(vtk, G3, "G3")
        for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
        pvd[t] = vtk
    end
end

d, G3 = director_field(dh, scv, u)
VTKGridFile("minilimo-dynamic-actuation-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, resu, "ru")
    Ferrite.write_node_data(vtk, resθ, "rθ")
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

println("PHASE 1 — dynamic HHT-α morph (Plv → $(round(p_hold*Pa2mmHg, digits=2)) mmHg)")
@printf("%-6s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "λ", "Plv [mmHg]", "iters", "Δt")

let t = 0.0; step = 0; Δt_cur = Δt; p = 0.0
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_new = p_hold * ramp(t_new)

    @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
    @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

    u_new .= ũ
    Ferrite.update!(ch, t_new * 5)
    apply!(u_new, ch)

    converged, iters = solve_morph_step!(u_new, ũ, ṽ, p_new, Δt_cur, dh, scv, mat, ch, Plv_srf, bufs;
                                         max_iter=max_iter, tol=tol)

    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v); @. g_old = α_damp * Mv + r_int - p_new * F_plv
        p = p_new; u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 4 == 0
            write_vtk!(pvd, vtk_step, dh, scv, grid, u, res, resu, resθ, t)
            @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d  %-10.4e\n", step, t, ramp(t), p * Pa2mmHg, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
    end
end
end

# The morph is complete: freeze the edge constraint at the fully-morphed arc so
# ch is time-independent for the whole actuation phase.
Ferrite.update!(ch, T_morph)
apply!(u, ch)

println("\nPHASE 2 — dynamic HHT-α actuation (Plv held at $(round(p_hold*Pa2mmHg, digits=2)) mmHg, Pact → $(round(Pact_max*Pa2mmHg, digits=1)) mmHg over $(T_act) s)")
@printf("%-6s  %-8s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "τ", "Plv [mmHg]", "Pact [mmHg]", "iters", "Δt")

un = zeros(N_dof)
let τ = 0.0; step = 0; Δt_cur = Δt; Plv = p_hold
@time while τ < T_act - 1e-10
    τ_new    = min(τ + Δt_cur, T_act)
    Pact_new = Pact_max * ramp_act(τ_new)

    @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
    @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

    u_new .= ũ
    apply!(u_new, ch)

    converged, iters = solve_actuation_step!(u_new, ũ, ṽ, Plv, Pact_new, Δt_cur, dh, scv, mat, ch,
                                             Plv_srf, Pact_srf, PlvPact_srf, bufs;
                                             max_iter=max_iter, tol=tol)

    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v)
        @. g_old = α_damp * Mv + r_int - Plv * F_plv - Pact_new * F_pact + Pact_new * F_plvpact
        u .= u_new; τ = τ_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 5 == 0
            write_vtk!(pvd, vtk_step, dh, scv, grid, u, res, resu, resθ, T_sim + τ)
            @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-8.4f  %-6d  %-10.4e\n",
                    step, T_sim + τ, ramp_act(τ), Plv * Pa2mmHg, Pact_new * Pa2mmHg, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at τ=$(round(τ, digits=4)) s")
    end
end
    un .= u
end
write_vtk!(pvd, vtk_step, dh, scv, grid, un, res, resu, resθ, T_sim + T_act)
close(pvd)

# using JLD2
# jldsave("minilimo_dynamic_actuation.jld2"; u=un)
# println("Dynamic actuation complete; final state saved to minilimo_dynamic_actuation.jld2")
