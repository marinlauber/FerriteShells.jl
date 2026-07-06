using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))

# Prescribed (pressure-controlled) dynamic inflation of the miniLIMO device on the
# rectangular multi-surface mesh (`make_minilimo_grid`, same geometry as
# `limo_inflation.jl`).  Loading is slow + heavily damped so the response is
# quasi-static, integrated with HHT-α.
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
# Two-phase prescribed loading:
#   Phase 1 (t ∈ [0, T_morph]): morph the edge from flat to the elliptic arc while
#     ramping Plv → Plv0 and Pact → Pact0 (both smooth sinusoidal).
#   Phase 2 (t ∈ [T_morph, T_sim]): hold morph + Pact = Pact0 fixed, ramp Plv from
#     Plv0 → Plv1 (slow inflation).
#
# HHT-α with mass-proportional Rayleigh damping C = α_damp·M:
#   g(u,v) = C·v + r_int(u) − F_ext(u)
#   R = M·ä + (1−α)·g(u_{n+1},v_{n+1}) + α·g_old = 0
#   K_eff = M·[1/(βΔt²) + (1−α)·α_damp·γ/(βΔt)]
#         + (1−α)·(K_int − Plv·K_plv − Pact·K_pact + Pact·K_plvpact)

# Seed a small smooth out-of-plane geometric imperfection into the REFERENCE mesh.
# The perfectly flat membrane makes the snap-through bifurcation singular (no
# preferred ±z side), which stalls the solver right at the snap; a tiny +z bump
# breaks the symmetry and turns the bifurcation into a smooth equilibrium path.
# The half-wave  sin(π y/Hy)·cos(π x/2Lx)  is exactly zero on y=0 (edge),
# y=Hy (sym top) and |x|=Lx (sym sides), so only the free interior is perturbed —
# the boundaries (where u_z is prescribed/fixed to 0) stay flat.  The +z sign
# biases the imperfection toward the Plv inflation direction.
function seed_imperfection!(grid; amp=1e-4)
    coords = get_node_coordinate.(getnodes(grid))
    Lx = maximum(abs(c[1]) for c in coords)
    Hy = maximum(c[2]      for c in coords)
    for (i, node) in enumerate(grid.nodes)
        x = get_node_coordinate(node)
        (x[2] ≈ 0.0 || x[2] ≈ Hy || abs(x[1]) ≈ Lx) && continue
        dz = amp * sin(π * x[2] / Hy) * cos(π * x[1] / (2Lx))
        grid.nodes[i] = Node(Vec{3}((x[1], x[2], x[3] + dz)))
    end
    return grid
end

# material
ρ   = 1200.0       # density [kg/m³]
mat = LinearElastic(0.267e8, 0.3, 0.0002)

Np = 2
grid = make_minilimo_grid(;
    nx_left=2*3, nx_act=2*10, nx_right=2*3,
    ny_bot=2*1, ny_act=2*14, ny_top=2*2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=Np
)

# small geometric imperfection (≈ half the shell thickness) to break the flat
# symmetry and help the solver pass through the snap; set amp=0 to disable
seed_imperfection!(grid; amp=1e-4)

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

# two-phase timeline (slow loading)
T_morph = 2.0   # phase 1: morph + reach (Plv0, Pact)
T_sim   = 6.0   # total; phase 2 = (T_morph, T_sim]: ramp Plv0 → Plv1
Δt      = 0.005

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

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable)
α_hht   = -0.3
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 100.0    # mass-proportional Rayleigh damping coefficient [1/s]
tol      = 1e-3
max_iter = 20
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
          res, rhs, δu, u_trial, a_new, v_new, Ma, Mv, F_lu, free,
          g_old, sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# HHT-α Newton corrector (with backtracking line search) for one time step.
# Prescribed pressures (p_plv, p_act) are held fixed within the step.
# `u_new` is updated in place; returns (converged, iters).  The expensive
# MITC/ForwardDiff element tangent is assembled only for the Newton direction;
# the line search uses residual-only assembly. All vector arithmetic is
# in-place / mul!-based.
#   g = C·v + r_int − F_ext;  F_ext = p_plv·F_plv + p_act·F_pact − p_act·F_plvpact
function solve_step!(u_new, ũ, ṽ, p_plv, p_act, Δt, dh, scv, mat, ch,
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
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int -
                 (p_plv * F_plv + p_act * F_pact - p_act * F_plvpact)) + α_hht * g_old
        apply_zero!(res, ch)
        res_norm = norm(@views res[free])
        res_norm < tol && (converged = true; break)
        K_eff.nzval .= M.nzval .* mfac .+ (1 - α_hht) .* (K_int.nzval .-
                       p_plv .* K_plv.nzval .- p_act .* K_pact.nzval .+ p_act .* K_plvpact.nzval)
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
            @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int -
                     (p_plv * F_plv + p_act * F_pact - p_act * F_plvpact)) + α_hht * g_old
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

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

pvd = paraview_collection("minilimo-prescribed-inflation")
vtk_step = Ref(0)
d, G3 = director_field(dh, scv, u)
VTKGridFile("minilimo-prescribed-inflation-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

@printf("%-6s  %-8s  %-9s  %-10s  %-11s  %-10s  %-6s  %-10s\n",
        "step", "t [s]", "λ_morph", "Plv [mmHg]", "Pact [mmHg]", "Vlv [ml]", "iters", "Δt")

un = zeros(N_dof)
let t = 0.0; step = 0; Δt_cur = Δt
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_plv = plv_schedule(t_new)
    p_act = pact_schedule(t_new)

    @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
    @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

    u_new .= ũ
    Ferrite.update!(ch, 5t_new)
    apply!(u_new, ch)

    converged, iters = solve_step!(u_new, ũ, ṽ, p_plv, p_act, Δt_cur, dh, scv, mat, ch,
                                   Plv_srf, Pact_srf, PlvPact_srf, bufs;
                                   max_iter=max_iter, tol=tol)

    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v)
        @. g_old = α_damp * Mv + r_int - (p_plv * F_plv + p_act * F_pact - p_act * F_plvpact)
        u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 2 == 0 || t ≥ T_sim - 1e-10 # ensures last step is written regardless
            Vlv = -2compute_volume(dh, scv, u; cellset=Plv_srf) * m3_to_ml
            vtk_step[] += 1
            d, G3 = director_field(dh, scv, u)
            VTKGridFile("minilimo-prescribed-inflation-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[t] = vtk
            end
            @printf("%-6d  %-8.3f  %-9.4f  %-10.4f  %-11.4f  %-10.4f  %-6d  %-10.4e\n",
                    step, t, morph_ramp(5t), p_plv * Pa2mmHg, p_act * Pa2mmHg, Vlv, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
        @printf("  → step rejected at t=%.3f, Δt → %.4e\n", t, Δt_cur)
    end
end
    un .= u
end
close(pvd)

# using JLD2
# jldsave("minilimo_prescribed_inflation.jld2"; u=un)
# Vlv_final = -2compute_volume(dh, scv, un; cellset=Plv_srf) * m3_to_ml
# @printf("Prescribed inflation complete. Final Vlv = %.4f ml at Plv = %.2f mmHg, Pact = %.2f mmHg\n",
#         Vlv_final, Plv1_mmHg, Pact_mmHg)
# println("Final state saved to minilimo_prescribed_inflation.jld2")
