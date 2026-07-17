using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK, JLD2
include(joinpath(@__DIR__, "util.jl"))

# Pressure-controlled quasi-static inflation of the miniLIMO device that BYPASSES
# the fragile incremental morphing phase.  Instead of slowly ramping the edge from
# flat to the elliptic arc (which drives a snap-through of the flat membrane), we
# build an approximate morphed configuration directly by extrapolating the known
# edge morph displacement into the interior, and start the solve from it.
#
# Extrapolation (transfinite y-blend of the boundary motion):  the morph is
# prescribed only on the `edge` (y=0) as (Δx(x), Δz(x)).  For an interior node at
# (x,y) we set
#     u(x,y) = ( φ(y)·Δx(x),  0,  φ(y)·Δz(x) ),   φ(y) = ½(1+cos(π y/Hy))
# so the full arc appears at y=0 (φ=1) and decays to flat at y=Hy (φ=0, respecting
# the `sym` top u_z=0).  This carries the +z arc sign into the interior, so the
# guess is on the correct post-snap branch — the snap is bypassed because we start
# past the singular point rather than path-following through it.
#
# Director seed:  the reference mesh is flat (z=0), so the shell centroid frame is
# the global axes (T₁=ê_x, T₂=ê_y, G₃=ê_z) and the Rodrigues director
#     d = cos|φ|·ê_z + sinc|φ|·(φ₁ ê_x + φ₂ ê_y)
# inverts analytically from the guessed surface normal n = ∂ₓp × ∂_y p:
#     |φ| = acos(n_z),   φ₁ = n_x·|φ|/√(n_x²+n_y²),   φ₂ = n_y·|φ|/√(n_x²+n_y²)
# with n in closed form from the blend (no per-element frame needed).
#
# The morph BC is then frozen at full morph and only the follower pressures ramp
# (the gentle part).  The static equilibrium is solved per pressure increment by
# pseudo-transient continuation (PTC): (M/Δτ + K)δu = −R with SER-adapted Δτ
# (small → robust steepest-descent step, large → Newton).  The very first solve
# settles the morphed shape from the guess at ~zero pressure.
#
# Three follower-pressure surfaces (as in limo_inflation.jl):
#   SRF_1: endocardium, Plv only (outward)
#   SRF_2: endocardium + actuator footprint, Plv (outward) and Pact (inward)
#   SRF_3: actuator exterior, Pact only (outward)
#   F_ext(u) = Plv·F_plv + Pact·F_pact − Pact·F_plvpact

# `morph_edge_data` and `build_morph_guess` (transfinite y-blend morph seed +
# analytic Rodrigues director) now live in util.jl.

# material
ρ   = 1200.0       # density [kg/m³] — only sets the scale of the pseudo-mass M
mat = LinearElastic(0.35e7, 0.3, 0.0002)   # E_true target (zero-pressure morph shape is E-independent)

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
Plv0_mmHg = 6.0    # Plv at end of the (bypassed) morph phase
Pact_mmHg = 6.0    # Pact reached during phase 1, held constant in phase 2
Plv1_mmHg = 20.0   # target Plv at end of phase 2 (slow ramp)
Plv0 = Plv0_mmHg / Pa2mmHg
Pact = Pact_mmHg / Pa2mmHg
Plv1 = Plv1_mmHg / Pa2mmHg

# load timeline.  Morph is NOT ramped (the guess provides it); the two phases now
# only ramp the pressures: phase 1 raises (0,0)→(Plv0,Pact), phase 2 Plv0→Plv1.
T_morph = 2.0
T_sim   = 0.05     # pressures are off → only need to settle the guess (no idle inflation phase)
Δt      = 0.01     # initial load (pressure) increment

cosramp(s) = 0.5 * (1 - cos(π * clamp(s, 0.0, 1.0)))
plv_schedule(t)  = t < T_morph ? Plv0 * cosramp(t / T_morph) :
                                 Plv0 + (Plv1 - Plv0) * cosramp((t - T_morph) / (T_sim - T_morph))
pact_schedule(t) = t < T_morph ? Pact * cosramp(t / T_morph) : Pact

prescribed_u = generate_boundary_function(grid, "edge")

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
close!(ch); Ferrite.update!(ch, 0.0)   # full morph (time-independent)

N_dof = ndofs(dh)
free  = ch.free_dofs

# PTC + load-stepping controls
tol      = 1e-3
max_iter = 200     # generous: the first solve settles the full morph from the guess
Δτ0      = 1e-9    # initial pseudo-time step: M/Δτ0 ≈ K so the first steps are
                   # gradient-descent-like (avoids overshoot into the snap); SER grows it
Δτ_min   = 1e-12
Δτ_max   = 1e30
Δt_min   = 1e-6
Δt_max   = 0.5

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
#   R = r_int − F_ext;   F_ext = p_plv·F_plv + p_act·F_pact − p_act·F_plvpact
#   (M/Δτ + K) δu = −R,  Δτ adapted by SER (grow → Newton; shrink → steepest descent).
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
            Δτ /= 2
            Δτ < Δτ_min && break
        end
        u_new .= u_trial
        step_ok || break
    end
    return converged, iters, Δτ
end

# Initial state: the extrapolated morphed shape (bypasses the incremental morph).
u = build_morph_guess(dh, grid)
apply!(u, ch)   # enforce exact edge morph + sym on the guess

pvd = paraview_collection("minilimo-morph-bypass")
vtk_step = Ref(0)
function write_vtk(name, u, t)
    Vlv = -2compute_volume(dh, scv, u; cellset=Plv_srf) * m3_to_ml
    d, G3 = director_field(dh, scv, u)
    VTKGridFile(name, dh) do vtk
        write_solution(vtk, dh, u)
        Ferrite.write_node_data(vtk, d,  "director")
        Ferrite.write_node_data(vtk, G3, "G3")
        for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
        pvd[t] = vtk
    end
    return Vlv
end

# step 0: the raw extrapolated guess (before any solve) for inspection
write_vtk("minilimo-morph-bypass-0", u, 0.0)

@printf("%-6s  %-8s  %-10s  %-11s  %-10s  %-6s  %-10s  %-10s\n",
        "step", "t", "Plv [mmHg]", "Pact [mmHg]", "Vlv [ml]", "iters", "Δt", "Δτ_end")

un = zeros(N_dof)
let t = 0.0; step = 0; Δt_cur = Δt
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_plv = 0.0 # plv_schedule(t_new)
    p_act = 0.0 # pact_schedule(t_new)

    @show t_new
    # Predictor: previous configuration with the (constant) full-morph BC applied.
    u_new .= u
    apply!(u_new, ch)

    converged, iters, Δτ_end = ptc_step!(u_new, p_plv, p_act, dh, scv, mat, ch,
                                         Plv_srf, Pact_srf, PlvPact_srf, bufs;
                                         max_iter=max_iter, tol=tol,
                                         Δτ0=Δτ0, Δτ_min=Δτ_min, Δτ_max=Δτ_max)
    @show converged, iters
    if converged
        step += 1
        u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 2 == 0 || t ≥ T_sim - 1e-10
            vtk_step[] += 1
            Vlv = write_vtk("minilimo-morph-bypass-$(vtk_step[])", u, t)
            @printf("%-6d  %-8.3f  %-10.4f  %-11.4f  %-10.4f  %-6d  %-10.4e  %-10.4e\n",
                    step, t, p_plv * Pa2mmHg, p_act * Pa2mmHg, Vlv, iters, Δt_cur, Δτ_end)
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

# Settled zero-pressure morphed equilibrium. Shape is E-independent (no external load),
# so this loads directly into limo_dynamic at E_true for the option-1 inflation start.
jldsave("minilimo_morph_stiff.jld2"; u=un)
println("Settled morphed state saved to minilimo_morph_stiff.jld2 (", length(un), " dofs).")

# using JLD2
# jldsave("minilimo_morph_bypass.jld2"; u=un)
