using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))
import OrdinaryDiffEq as ODE

# Fully-dynamic (HHT-α throughout) morphing + 3D-0D Windkessel-coupled beat of the
# miniLIMO, on the rectangular multi-surface mesh built by `make_minilimo_grid`.  This is
# the transient counterpart of `limo_dynamic_coupled.jl`: there the coupled phase is a
# quasi-static Schur solve; here inertia and damping are retained in the coupled phase too.
#
#   PHASE 1 — dynamic morph  (t ∈ [0, T_sim], HHT-α)
#     Edge nodes are driven onto the elliptic arc while a follower Plv (same ramp) fills the
#     endocardium (SRF_1 ∪ SRF_2).  End state: u = un, Plv = p_max, with velocity/accel v, a.
#
#   PHASE 2 — dynamic 3D-0D coupling  (HHT-α, Δt = dt_cpl, Lie–Trotter split)
#     The 0D Windkessel is advanced by dt_cpl with Plv held; its LV volume sets a target the
#     3D shell must enclose.  Each step is a DYNAMIC bordered-Newton solve for (u, Plv):
#       res(u,p) = M·ä + (1−α)(α_damp·Mv + r_int(u) − F_ext(u,p)) + α·g_old = 0
#       r_V(u)   = V₃D(u) − V_target = 0            (Plv is the multiplier)
#     with F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact.  Bordered system:
#       ∂res/∂u = K_eff = M·mfac + (1−α)(K_int − p·K_plv − Pact·K_pact + Pact·K_plvpact)
#       ∂res/∂p = −(1−α)·F_plv,   ∂r_V/∂u = −dVdu   (dVdu = ∂compute_volume/∂u = −∂V₃D/∂u)
#     Schur complement: v1 = K_eff⁻¹(−res), v2 = (1−α)·K_eff⁻¹F_plv,
#       δp = (−r_V + dot(dVdu,v1)) / (−dot(dVdu,v2)),  δu = v1 + δp·v2.
#     The converged Plv is fed back into the ODE.  Three follower surfaces:
#       SRF_1: endocardium, Plv only (outward)
#       SRF_2: endocardium + actuator footprint, Plv (outward) and Pact (inward)
#       SRF_3: actuator exterior, Pact only (outward)

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
# SRF_1: endocardium, Plv only (outward)
# SRF_2: endocardium + actuator, Plv (outward) and Pact (inward, opposing Plv)
# SRF_3: actuator exterior, Pact only (outward)
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

pvd = paraview_collection("minilimo-dynamic-coupled-transient")
vtk_step = Ref(0)
resu = zeros(3, getnnodes(dh.grid))
resθ = zeros(2, getnnodes(dh.grid))
d, G3 = director_field(dh, scv, u)
VTKGridFile("minilimo-dynamic-coupled-transient-0", dh) do vtk
    write_solution(vtk, dh, u)
    Ferrite.write_node_data(vtk, resu, "ru")
    Ferrite.write_node_data(vtk, resθ, "rθ")
    Ferrite.write_node_data(vtk, d,  "director")
    Ferrite.write_node_data(vtk, G3, "G3")
    for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
    pvd[0.0] = vtk
end

println("PHASE 1 — dynamic HHT-α morph")
@printf("%-6s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "λ", "p [mmHg]", "iters", "Δt")

un = zeros(N_dof)
let t = 0.0; step = 0; Δt_cur = Δt; p = 0.0
@time while t < T_sim - 1e-10
    t_new = min(t + Δt_cur, T_sim)
    p_new = p_max * ramp(t_new)

    @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
    @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

    u_new .= ũ
    Ferrite.update!(ch, t_new * 5)
    apply!(u_new, ch)

    converged, iters = solve_morph_step!(u_new, ũ, ṽ, p_new, Δt_cur, dh, scv, mat, ch, Plv_srf, bufs_morph;
                                         max_iter=max_iter, tol=tol)

    if converged
        step += 1
        @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
        @. v = ṽ + (Δt_cur * γ_hht) * a
        mul!(Mv, M, v); @. g_old = α_damp * Mv + r_int - p_new * F_plv
        p = p_new; u .= u_new; t = t_new
        Δt_cur = min(Δt_cur * 1.2, Δt_max)
        if step % 4 == 0
            vtk_step[] += 1
            for cell in CellIterator(dh)
                sd = shelldofs(cell)
                for (I, nid) in enumerate(cell.nodes)
                    resu[:, nid] .= res[sd[5I-4:5I-2]]
                    resθ[:, nid] .= res[sd[5I-1:5I  ]]
                end
            end
            d, G3 = director_field(dh, scv, u)
            VTKGridFile("minilimo-dynamic-coupled-transient-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, resu, "ru")
                Ferrite.write_node_data(vtk, resθ, "rθ")
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[t] = vtk
            end
            @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d  %-10.4e\n", step, t, ramp(t), p * Pa2mmHg, iters, Δt_cur)
        end
    else
        Δt_cur /= 2
        Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
    end
end
    un .= u
end

# Freeze the fully-morphed edge configuration (t·5 ≥ T_morph → ramp = 1) for the coupled
# phase; the Dirichlet morph is held constant from here on (u, v, a carried forward).
Ferrite.update!(ch, T_sim * 5)
apply!(u, ch)

# Windkessel open-loop 0D model (Kasra's parameters, mmHg/ml units).
function Windkessel!(du, u, p, t)
    (Vlv, Pa, Pv, Plv) = u
    (Ra, Ca, Rv, Cv, Rp) = p
    Qmv = Pv ≥ Plv ? (Pv - Plv)/Rv : (Plv - Pv)/1e10
    Qao = Plv ≥ Pa ? (Plv - Pa)/Ra : (Pa - Plv)/1e10
    du[1] = Qmv - Qao                 # dVlv/dt
    du[2] = Qao/Ca + (Pv-Pa)/(Rp*Ca)  # dPa/dt
    du[3] = (Pa-Pv)/(Rp*Cv) - Qmv/Cv  # dPv/dt
    du[4] = 0.0                       # Plv held fixed within the ODE substep
end

# actuation waveform (normalized to [0,1])
ϕᵢ(t; tC=0.10, tR=0.25, TC=0.15, TR=0.45) =
    0.0 <= (t-tC)%1 <= TC ? 0.5*(1 - cos(π*((t-tC)%1)/TC)) :
    (0.0 <= (t-tR)%1 <= TR ? 0.5*(1 + cos(π*((t-tR)%1)/TR)) : 0.0)

# Kasra's parameters (converted Pa·s/m³ → mmHg·s/ml, m³/Pa → ml/mmHg)
Ra = 8.0e6*Pa2mmHg/m3_to_ml
Rp = 1.0e8*Pa2mmHg/m3_to_ml
Rv = 5.0e5*Pa2mmHg/m3_to_ml
Ca = 8.0e-9*m3_to_ml/Pa2mmHg
Cv = 5.0e-8*m3_to_ml/Pa2mmHg
Pv = p_max * Pa2mmHg

# initial (full-LV) cavity volume from the morphed state, in ml
vol = -2compute_volume(dh, scv, u; cellset=Plv_srf) * m3_to_ml
println("Initial volume of the device: ", round(vol; digits=4), " ml")

u₀     = [vol, 80.0, Pv, Pv]   # [Vlv, Pa, Pv, Plv]
tspan  = (0.0, 4.0)
params = (Ra, Ca, Rv, Cv, Rp)
prob   = ODE.ODEProblem(Windkessel!, u₀, tspan, params)
integrator = ODE.init(prob, ODE.Tsit5(), reltol=1e-6, abstol=1e-9, save_everystep=false)

# coupling controls
tol_cpl  = 1e-4
max_iter = 50
dt_cpl   = 0.01   # doubles as the HHT-α time step in the coupled phase

# storages
vols = Float64[]; pres = Float64[]; pact = Float64[]
paos = Float64[]; pvns = Float64[]; vtarget = Float64[]

bufs_cpl = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
              res, rhs, v1, v2, dVdu, a_new, v_new, Ma, Mv, F_lu, free, g_old,
              sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# Dynamic (HHT-α) volume-controlled bordered-Newton solve for one coupling step: find
# (u_new, p=Plv) enforcing both the HHT-α equation of motion and V₃D(u_new) = V_target,
# with Plv the multiplier.  `u_new` updated in place; returns (p, iters, converged, V₃D).
# F_ext = p·F_plv + Pact·F_pact − Pact·F_plvpact.
function solve_coupled_dyn_step!(u_new, ũ, ṽ, p, Pact, V_target, Δt, dh, scv, mat, ch,
                                 Plv_srf, Pact_srf, PlvPact_srf, bufs; max_iter=20, tol=1e-4, verbose=false)
    (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
       res, rhs, v1, v2, dVdu, a_new, v_new, Ma, Mv, F_lu, free, g_old,
       sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp) = bufs
    mfac = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)
    converged = false; n_iter = 0; V₃D = 0.0
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
                 (p * F_plv + Pact * F_pact - Pact * F_plvpact)) + α_hht * g_old
        apply_zero!(res, ch)
        # volume_residual returns −val → compute_volume < 0 for outward (+z) inflation.
        V₃D = -compute_volume(dh, scv, u_new; cellset=Plv_srf)
        volume_gradient!(dVdu, dh, scv, u_new; cellset=Plv_srf)
        dVdu[ch.prescribed_dofs] .= 0.0
        r_V = V₃D - V_target
        res_norm = norm(@views res[free])
        verbose && @printf("    iter %2d | r_V=%+.3e | |res|=%.3e\n", iter, r_V, res_norm)
        if res_norm < tol && abs(r_V) < tol * max(1.0, abs(V_target)) && iter != 1
            converged = true; n_iter = iter - 1; break
        end
        K_eff.nzval .= M.nzval .* mfac .+ (1 - α_hht) .* (K_int.nzval .-
                       p .* K_plv.nzval .- Pact .* K_pact.nzval .+ Pact .* K_plvpact.nzval)
        @. rhs = -res
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(v1, F_lu, rhs)
        ldiv!(v2, F_lu, F_plv); v2 .*= (1 - α_hht)   # ∂res/∂p = −(1−α)·F_plv
        # Schur complement (dVdu = ∂(compute_volume)/∂u = −∂V₃D/∂u):
        S  = -dot(dVdu, v2)
        δp = (-r_V + dot(dVdu, v1)) / S
        @. u_new = u_new + v1 + δp * v2
        p += δp
        apply!(u_new, ch)
    end
    return p, n_iter, converged, V₃D
end

dt_cpl   = 0.005

println("\nPHASE 2 — dynamic HHT-α 3D-0D coupling (dt_cpl=$(dt_cpl) s)")
println("      t [s] |  p [mmHg]   |  Vlv_full [ml]  |  Pact [mmHg]  | iters")

@time let p = p_max
    step = 0
    while integrator.t < tspan[2] - dt_cpl / 2
        step += 1

        # advance Windkessel by dt_cpl (Plv = integrator.u[4] held fixed).
        ODE.step!(integrator, dt_cpl, true)

        # target half-model volume (m³) from the full-LV volume the ODE tracks.
        V_target = 0.5 * integrator.u[1] / m3_to_ml
        push!(vtarget, integrator.u[1])

        # actuator pressure at this time [mmHg] → Pa
        Pact_mmHg = 200 * ϕᵢ(integrator.t; tC=0.1, tR=0.4, TC=0.3, TR=0.3)
        Pact = Pact_mmHg / Pa2mmHg

        # HHT-α predictors for this step (Δt = dt_cpl), morph BC frozen.
        @. ũ = u + dt_cpl * v + (dt_cpl^2 * (0.5 - β_hht)) * a
        @. ṽ = v + (dt_cpl * (1 - γ_hht)) * a
        u_new .= ũ
        apply!(u_new, ch)

        p, n_iter, converged, V₃D = solve_coupled_dyn_step!(u_new, ũ, ṽ, p, Pact, V_target, dt_cpl,
                                        dh, scv, mat, ch, Plv_srf, Pact_srf, PlvPact_srf, bufs_cpl;
                                        max_iter=max_iter, tol=tol_cpl, verbose=false)

        # commit dynamic state (velocity/accel updates + HHT history g_old).
        @. a = (u_new - ũ) / (β_hht * dt_cpl^2)
        @. v = ṽ + (dt_cpl * γ_hht) * a
        mul!(Mv, M, v)
        @. g_old = α_damp * Mv + r_int - (p * F_plv + Pact * F_pact - Pact * F_plvpact)
        u .= u_new

        if step%50 == 0
            vtk_step[] += 1
            for cell in CellIterator(dh)
                sd = shelldofs(cell)
                for (I, nid) in enumerate(cell.nodes)
                    resu[:, nid] .= res[sd[5I-4:5I-2]]
                    resθ[:, nid] .= res[sd[5I-1:5I  ]]
                end
            end
            d, G3 = director_field(dh, scv, u)
            VTKGridFile("minilimo-dynamic-coupled-transient-$(vtk_step[])", dh) do vtk
                write_solution(vtk, dh, u)
                Ferrite.write_node_data(vtk, resu, "ru")
                Ferrite.write_node_data(vtk, resθ, "rθ")
                Ferrite.write_node_data(vtk, d,  "director")
                Ferrite.write_node_data(vtk, G3, "G3")
                for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
                pvd[T_sim + integrator.t] = vtk
            end
        end
        @printf("  %9.4f | %11.4f | %14.4f | %14.4f | %d\n",
                integrator.t, p * Pa2mmHg, 2V₃D * m3_to_ml, Pact_mmHg, n_iter)

        !converged && (@warn "coupling step $step (t=$(integrator.t)) did not converge"; break)

        # feed the converged LV pressure back into the ODE state.
        integrator.u[4] = p * Pa2mmHg
        ODE.u_modified!(integrator, true)

        push!(vols, 2V₃D * m3_to_ml)   # full LV volume [ml]
        push!(pres, p * Pa2mmHg)       # LV pressure [mmHg]
        push!(pact, Pact_mmHg)
        push!(paos, integrator.u[2])
        push!(pvns, integrator.u[3])
    end
end
close(pvd)

using Plots
times = collect(0:dt_cpl:integrator.t)[1:length(pres)]
p1 = plot(times, [vols, pres, pact, paos, pvns], xlabel="Time [s]",
          label=["Vlv" "Plv" "Pact" "Pao" "Pv"], lw=2, legend=:right)
p2 = plot(vols, pres, label=:none, xlim=extrema(vols).+(-10,10), ylims=(0, 100),
          xlabel="Volume [ml]", ylabel="Pressure [mmHg]", lw=2, linez=times./maximum(times))
plot(p1, p2)
# savefig("minilimo-dynamic-coupled-transient-N$Np.png")
