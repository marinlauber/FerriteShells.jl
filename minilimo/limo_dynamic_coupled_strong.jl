using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))

# Strongly (monolithically) coupled dynamic miniLIMO: HHT-α structure + 0D Windkessel
# solved together in ONE Newton system per time step.  Counterpart of the weakly-coupled
# `limo_dynamic_coupled_transient.jl` (Lie–Trotter split with a black-box ODE integrator).
#
#   PHASE 1 — dynamic morph  (t ∈ [0, T_sim], HHT-α)   [identical to the transient file]
#     Morph the edge onto the elliptic arc + fill Plv → morphed state u = un, v, a, Plv=p_max.
#
#   PHASE 2 — monolithic strong coupling  (HHT-α, Δt = dt_cpl)
#     The Windkessel is discretized by implicit Euler and embedded in the structural Newton;
#     the black-box ODE integrator is gone.  The LV chamber volume is NOT an independent 0D
#     state — it is V_LV(u) = 2·V₃D(u); Plv is the pressure that closes the flow balance.
#     Unknowns per step: (u ∈ Rᴺ, Plv, Pa, Pv).  Residuals:
#       R_s(u,Plv) = M·ä + (1−α)(α_damp·Mv + r_int − F_ext(u,Plv,Pact)) + α·g_old   [N]
#       R_vol      = (V_LV(u) − V_LVₙ)/Δt − (Qmv − Qao)                              [1]
#       R_art      = (Pa − Paₙ)/Δt − (Qao/Ca + (Pv − Pa)/(Rp·Ca))                    [1]
#       R_ven      = (Pv − Pvₙ)/Δt − ((Pa − Pv)/(Rp·Cv) − Qmv/Cv)                    [1]
#     with F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact, V_LV = 2·V₃D = −2·compute_volume,
#     valves Qmv(Plv,Pv), Qao(Plv,Pa) (diodes).  Bordered/condensed Newton:
#       v1 = K_eff⁻¹(−R_s),  v2 = (1−α)·K_eff⁻¹F_plv,  δu = v1 + δPlv·v2,
#       aᵀ = (1/Δt)·∂V_LV/∂u = −(2/Δt)·dVdu  (dVdu = ∂compute_volume/∂u = −∂V₃D/∂u),
#       3×3 solve for (δPlv, δPa, δPv) with the volume row augmented by aᵀv2 / aᵀv1.
#     Everything in SI (Pa, m³, s); pressures reported in mmHg, volumes in ml.
#     Three follower surfaces: SRF_1 Plv, SRF_2 Plv−Pact, SRF_3 Pact.

# Valve diodes (SI: Pa, m³/s).  Return (Q, ∂Q/∂P_first, ∂Q/∂P_second) on the active branch.
# Mitral: venous → LV filling, opens when Pv ≥ Plv.  Args (Plv, Pv).
@inline function mitral_flow(Plv, Pv, Rv; R_closed=1e10)
    Pv ≥ Plv ? ((Pv - Plv)/Rv,       -1/Rv,       1/Rv) :
               ((Plv - Pv)/R_closed,  1/R_closed, -1/R_closed)   # (Q, ∂/∂Plv, ∂/∂Pv)
end
# Aortic: LV → arterial ejection, opens when Plv ≥ Pa.  Args (Plv, Pa).
@inline function aortic_flow(Plv, Pa, Ra; R_closed=1e10)
    Plv ≥ Pa ? ((Plv - Pa)/Ra,        1/Ra,       -1/Ra) :
               ((Pa - Plv)/R_closed, -1/R_closed,  1/R_closed)   # (Q, ∂/∂Plv, ∂/∂Pa)
end

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

# The three follower-pressure surfaces (used in the coupled phase; Phase 1 uses Plv_srf).
Plv_srf     = getcellset(grid, "SRF_1") ∪ getcellset(grid, "SRF_2")  # Plv acts here
Pact_srf    = getcellset(grid, "SRF_3")                              # +Pact
PlvPact_srf = getcellset(grid, "SRF_2")                              # −Pact (opposes Plv)

# Smooth sinusoidal ramp: λ(t) = ½(1 − cos(πt/T_morph)) for t ≤ T_morph, 1 beyond.
T_morph = 2.0   # morphing duration [s]
T_sim   = 2.0   # Phase-1 simulation duration [s]
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

# Pressure ramp: same sinusoidal profile as morphing, up to p_max [Pa]
Pa2mmHg = 0.00750062 # Pa/mmHg
m3_to_ml = 1.0e6     # m³ → ml
p_max   = 6.0 / Pa2mmHg   # 6 mmHg → Pa

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
v1        = zeros(N_dof)   # coupled-phase Schur vectors
v2        = zeros(N_dof)
dVdu      = zeros(N_dof)
a_new     = zeros(N_dof)
v_new     = zeros(N_dof)
Ma        = zeros(N_dof)
Mv        = zeros(N_dof)
ũ         = zeros(N_dof)
ṽ         = zeros(N_dof)
u_new     = zeros(N_dof)

# Precomputed shell-DOF maps (fixed for the run) and reusable element buffers.
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

bufs_morph = (; K_int, r_int, K_plv, F_plv, M, K_eff, res, rhs, δu, u_trial, a_new, v_new,
                Ma, Mv, F_lu, free, g_old, sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# HHT-α Newton corrector (with backtracking line search) for one morph time step.
# `u_new` is updated in place; returns (converged, iters). Plv pressure acts on `Plv_srf`.
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
        ls_ok || break
    end
    return converged, iters
end

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

pvd = paraview_collection("minilimo-dynamic-coupled-strong")
vtk_step = Ref(0)
resu = zeros(3, getnnodes(dh.grid))
resθ = zeros(2, getnnodes(dh.grid))
d, G3 = director_field(dh, scv, u)
VTKGridFile("minilimo-dynamic-coupled-strong-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, resu, "ru")
    Ferrite.write_node_data(vtk, resθ, "rθ")
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

# println("PHASE 1 — dynamic HHT-α morph")
# @printf("%-6s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "λ", "p [mmHg]", "iters", "Δt")

un = zeros(N_dof)
# let t = 0.0; step = 0; Δt_cur = Δt; p = 0.0
# @time while t < T_sim - 1e-10
#     t_new = min(t + Δt_cur, T_sim)
#     p_new = p_max * ramp(t_new)

#     @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
#     @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

#     u_new .= ũ
#     Ferrite.update!(ch, t_new * 5)
#     apply!(u_new, ch)

#     converged, iters = solve_morph_step!(u_new, ũ, ṽ, p_new, Δt_cur, dh, scv, mat, ch, Plv_srf, bufs_morph;
#                                          max_iter=max_iter, tol=tol)

#     if converged
#         step += 1
#         @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
#         @. v = ṽ + (Δt_cur * γ_hht) * a
#         mul!(Mv, M, v); @. g_old = α_damp * Mv + r_int - p_new * F_plv
#         p = p_new; u .= u_new; t = t_new
#         Δt_cur = min(Δt_cur * 1.2, Δt_max)
#         if step % 4 == 0
#             vtk_step[] += 1
#             for cell in CellIterator(dh)
#                 sd = shelldofs(cell)
#                 for (I, nid) in enumerate(cell.nodes)
#                     resu[:, nid] .= res[sd[5I-4:5I-2]]
#                     resθ[:, nid] .= res[sd[5I-1:5I  ]]
#                 end
#             end
#             d, G3 = director_field(dh, scv, u)
#             VTKGridFile("minilimo-dynamic-coupled-strong-$(vtk_step[])", dh) do vtk
#                 write_solution(vtk, dh, u)
#                 Ferrite.write_node_data(vtk, resu, "ru")
#                 Ferrite.write_node_data(vtk, resθ, "rθ")
#                 Ferrite.write_node_data(vtk, d,  "director")
#                 Ferrite.write_node_data(vtk, G3, "G3")
#                 for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
#                 pvd[t] = vtk
#             end
#             @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d  %-10.4e\n", step, t, ramp(t), p * Pa2mmHg, iters, Δt_cur)
#         end
#     else
#         Δt_cur /= 2
#         Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
#     end
# end
#     un .= u
# end

using JLD2
# jldsave("limo_dynamic_coupled_u0.jld2"; u=un)
# reload if done already
un .= load("limo_dynamic_coupled_u0.jld2")["u"]

# Freeze the fully-morphed edge configuration (t·5 ≥ T_morph → ramp = 1) for the coupled
# phase; the Dirichlet morph is held constant from here on (u, v, a carried forward).
Ferrite.update!(ch, T_sim * 5)
apply!(u, ch)

# actuation waveform (normalized to [0,1])
ϕᵢ(t; tC=0.10, tR=0.25, TC=0.15, TR=0.45) =
    0.0 <= (t-tC)%1 <= TC ? 0.5*(1 - cos(π*((t-tC)%1)/TC)) :
    (0.0 <= (t-tR)%1 <= TR ? 0.5*(1 + cos(π*((t-tR)%1)/TR)) : 0.0)

# Windkessel parameters, SI (Pa, m³, s) — Plv, Pa, Pv all carried in Pa.
Ra = 8.0e6    # aortic resistance   [Pa·s/m³]
Rp = 1.0e8    # peripheral resist.  [Pa·s/m³]
Rv = 5.0e5    # mitral resistance   [Pa·s/m³]
Ca = 8.0e-9   # arterial compliance [m³/Pa]
Cv = 5.0e-8   # venous compliance   [m³/Pa]
wk = (; Ra, Rp, Rv, Ca, Cv, Pscale = p_max)

# coupling controls
tol_cpl  = 1e-4
max_iter = 20
dt_cpl   = 0.01   # doubles as the HHT-α time step in the coupled phase
T_beat   = 4.0    # total coupled duration [s]

# storages
vols = Float64[]; pres = Float64[]; pact = Float64[]
paos = Float64[]; pvns = Float64[]

bufs_cpl = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
              res, rhs, v1, v2, dVdu, a_new, v_new, Ma, Mv, F_lu, free, g_old,
              sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# Monolithic strong-coupling step: solve the HHT-α structure AND the implicit-Euler 0D
# Windkessel simultaneously for (u_new, Plv, Pa, Pv).  `u_new` updated in place; the LV
# chamber volume is V_LV(u) = 2·V₃D(u), Plv is the coupling multiplier.  V_LVₙ/Paₙ/Pvₙ are
# the previous converged 0D state.  Returns (Plv, Pa, Pv, iters, converged, V₃D).
function solve_coupled_strong_step!(u_new, ũ, ṽ, Plv, Pa, Pv, V_LVₙ, Paₙ, Pvₙ,
                                    Pact, Δt, dh, scv, mat, ch,
                                    Plv_srf, Pact_srf, PlvPact_srf, wk, bufs;
                                    max_iter=20, tol=1e-4, verbose=false)
    (; Ra, Rp, Rv, Ca, Cv, Pscale) = wk
    (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
       res, rhs, v1, v2, dVdu, a_new, v_new, Ma, Mv, F_lu, free, g_old,
       sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    converged = false; n_iter = 0; V₃D = 0.0
    scaleV = max(abs(V_LVₙ), 1e-12) / Δt
    scaleP = Pscale / Δt
    for iter in 1:max_iter
        n_iter = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat, sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plv,     F_plv,     dh, scv, u_new, Plv_srf,     sdofs, ke, re, u_e)
        assemble_pressure_region!(K_pact,    F_pact,    dh, scv, u_new, Pact_srf,    sdofs, ke, re, u_e)
        assemble_pressure_region!(K_plvpact, F_plvpact, dh, scv, u_new, PlvPact_srf, sdofs, ke, re, u_e)
        @. a_new = (u_new - ũ) / (β_hht * Δt^2)
        @. v_new = ṽ + (Δt * γ_hht) * a_new
        mul!(Ma, M, a_new); mul!(Mv, M, v_new)
        @. res = Ma + (1 - α_hht) * (α_damp * Mv + r_int -
                 (Plv * F_plv + Pact * F_pact - Pact * F_plvpact)) + α_hht * g_old
        apply_zero!(res, ch)
        # chamber volume (full LV) and its gradient; dVdu = ∂compute_volume/∂u = −∂V₃D/∂u
        V₃D  = -compute_volume(dh, scv, u_new; cellset=Plv_srf)
        V_LV = 2 * V₃D
        volume_gradient!(dVdu, dh, scv, u_new; cellset=Plv_srf)
        dVdu[ch.prescribed_dofs] .= 0.0
        # 0D residuals (implicit Euler); flows on the active valve branch
        Qmv, dQmv_dPlv, dQmv_dPv = mitral_flow(Plv, Pv, Rv)
        Qao, dQao_dPlv, dQao_dPa = aortic_flow(Plv, Pa, Ra)
        R_vol = (V_LV - V_LVₙ) / Δt - (Qmv - Qao)
        R_art = (Pa - Paₙ) / Δt - (Qao / Ca + (Pv - Pa) / (Rp * Ca))
        R_ven = (Pv - Pvₙ) / Δt - ((Pa - Pv) / (Rp * Cv) - Qmv / Cv)
        res_norm = norm(@views res[free])
        ok_c = abs(R_vol)/scaleV < tol && abs(R_art)/scaleP < tol && abs(R_ven)/scaleP < tol
        verbose && @printf("    it %2d | |res|=%.2e Rv=%+.2e Ra=%+.2e Rν=%+.2e\n",
                           iter, res_norm, R_vol, R_art, R_ven)
        (res_norm < tol && ok_c && iter != 1) && (converged = true; n_iter = iter - 1; break)
        # structural tangent + two back-substitutions (shared with the volume row)
        K_eff.nzval .= M.nzval .* mfac .+ (1 - α_hht) .* (K_int.nzval .-
                       Plv .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(v1, F_lu, rhs)
        ldiv!(v2, F_lu, F_plv); v2 .*= (1 - α_hht)   # ∂R_s/∂Plv = −(1−α)·F_plv → δu = v1 + δPlv·v2
        # aᵀ = (1/Δt)·∂V_LV/∂u = −(2/Δt)·dVdu  (volume-row coupling to δu)
        aTv1 = -(2 / Δt) * dot(dVdu, v1)
        aTv2 = -(2 / Δt) * dot(dVdu, v2)
        # 3×3 condensed 0D Jacobian J·(δPlv,δPa,δPv) = rhs_c  (column-major Tensor)
        j11 = (-dQmv_dPlv + dQao_dPlv) + aTv2
        j12 = dQao_dPa
        j13 = -dQmv_dPv
        j21 = -dQao_dPlv / Ca
        j22 = 1/Δt - dQao_dPa / Ca + 1/(Rp*Ca)
        j23 = -1/(Rp*Ca)
        j31 = dQmv_dPlv / Cv
        j32 = -1/(Rp*Cv)
        j33 = 1/Δt + 1/(Rp*Cv) + dQmv_dPv / Cv
        Jt  = Tensor{2,3}((j11, j21, j31, j12, j22, j32, j13, j23, j33))
        bt  = Vec{3}((-R_vol - aTv1, -R_art, -R_ven))
        δc  = inv(Jt) ⋅ bt
        δPlv, δPa, δPv = δc[1], δc[2], δc[3]
        @. u_new = u_new + v1 + δPlv * v2
        Plv += δPlv; Pa += δPa; Pv += δPv
        apply!(u_new, ch)
    end
    return Plv, Pa, Pv, n_iter, converged, V₃D
end

# initial (full-LV) cavity volume from the morphed state
V_LV0 = -2 * compute_volume(dh, scv, u; cellset=Plv_srf)   # m³
println("Initial volume of the device: ", round(V_LV0 * m3_to_ml; digits=4), " ml")

println("\nPHASE 2 — monolithic strong 3D-0D coupling (dt_cpl=$(dt_cpl) s)")
println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")

# initial 0D state (Pa): filled ventricle, arterial at 80 mmHg
@time let V_LVₙ = V_LV0, Paₙ = Pa, Pvₙ = Pv, t_cpl = 0.0, Plv=p_max, Pa = 80.0 / Pa2mmHg, Pv = p_max
    step = 0
    while t_cpl < T_beat - dt_cpl / 2
        step += 1
        t_cpl += dt_cpl

        Pact_mmHg = 200 * ϕᵢ(t_cpl; tC=0.1, tR=0.4, TC=0.3, TR=0.3)
        Pact = Pact_mmHg / Pa2mmHg

        # HHT-α predictors (Δt = dt_cpl), morph BC frozen.
        @. ũ = u + dt_cpl * v + (dt_cpl^2 * (0.5 - β_hht)) * a
        @. ṽ = v + (dt_cpl * (1 - γ_hht)) * a
        u_new .= ũ
        apply!(u_new, ch)

        Plv, Pa, Pv, n_iter, converged, V₃D = solve_coupled_strong_step!(
            u_new, ũ, ṽ, Plv, Pa, Pv, V_LVₙ, Paₙ, Pvₙ, Pact, dt_cpl, dh, scv, mat, ch,
            Plv_srf, Pact_srf, PlvPact_srf, wk, bufs_cpl; max_iter=max_iter, tol=tol_cpl, verbose=false)

        # commit dynamic structural state + advance 0D history
        @. a = (u_new - ũ) / (β_hht * dt_cpl^2)
        @. v = ṽ + (dt_cpl * γ_hht) * a
        mul!(Mv, M, v)
        @. g_old = α_damp * Mv + r_int - (Plv * F_plv + Pact * F_pact - Pact * F_plvpact)
        u .= u_new
        V_LVₙ = 2 * V₃D; Paₙ = Pa; Pvₙ = Pv

        vtk_step[] += 1
        for cell in CellIterator(dh)
            sd = shelldofs(cell)
            for (I, nid) in enumerate(cell.nodes)
                resu[:, nid] .= res[sd[5I-4:5I-2]]
                resθ[:, nid] .= res[sd[5I-1:5I  ]]
            end
        end
        d, G3 = director_field(dh, scv, u)
        VTKGridFile("minilimo-dynamic-coupled-strong-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u)
            Ferrite.write_node_data(vtk, resu, "ru")
            Ferrite.write_node_data(vtk, resθ, "rθ")
            Ferrite.write_node_data(vtk, d,  "director")
            Ferrite.write_node_data(vtk, G3, "G3")
            for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
            pvd[T_sim + t_cpl] = vtk
        end
        @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n",
                t_cpl, Plv * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)

        !converged && (@warn "coupling step $step (t=$(t_cpl)) did not converge"; break)

        push!(vols, 2V₃D * m3_to_ml)   # full LV volume [ml]
        push!(pres, Plv * Pa2mmHg)     # LV pressure [mmHg]
        push!(pact, Pact_mmHg)
        push!(paos, Pa * Pa2mmHg)
        push!(pvns, Pv * Pa2mmHg)
    end
end
close(pvd)

using Plots
times = collect(dt_cpl:dt_cpl:dt_cpl*length(pres))
p1 = plot(times, [vols, pres, pact, paos, pvns], xlabel="Time [s]",
          label=["Vlv" "Plv" "Pact" "Pao" "Pv"], lw=2, legend=:right)
p2 = plot(vols, pres, label=:none, xlim=extrema(vols).+(-10,10), ylims=(0, 100),
          xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2, linez=times./maximum(times))
plot(p1, p2)
# savefig("minilimo-dynamic-coupled-strong-N$Np.png")
