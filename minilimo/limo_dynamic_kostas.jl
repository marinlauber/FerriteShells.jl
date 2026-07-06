using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK
include(joinpath(@__DIR__, "util.jl"))

# Dynamic (HHT-α) miniLIMO simulation on the rectangular multi-surface mesh built
# by `make_minilimo_grid` (same geometry as `limo_dynamic.jl`).  A single implicit
# time-integration loop advances the whole protocol; a `schedule(t)` maps global
# time to the edge-morph ramp argument and the two chamber pressures, so the morph,
# actuation and pressurization stages are just successive segments of one loop:
#
#   [0, T_sim]        morph   : edge → elliptic arc, Plv ramps 0 → p_hold
#   [T_sim, +T_act]   actuate : Plv held, Pact ramps 0 → Pact_max
#   [+T_act, +T_pres] pressur : Pact held, Plv ramps → Plv_max
#
# The three follower surfaces give F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact
# (SRF_1 Plv; SRF_2 Plv−Pact; SRF_3 Pact).  One HHT-α corrector `solve_step!` with
# mass-proportional Rayleigh damping C = α_damp·M, adaptive Δt and a backtracking
# line search serves every stage (morph is just Pact = 0):
#   g(u,v) = C·v + r_int(u) − F_ext(u)
#   R = M·ä_{n+1} + (1−α)·g_{n+1} + α·g_old = 0
#   γ = ½ − α,  β = (1−α)²/4   (2nd-order, unconditionally stable for α ∈ [−⅓,0])
#   K_eff = M·[1/(βΔt²) + (1−α)·α_damp·γ/(βΔt)] + (1−α)·(K_int − ∂F_ext/∂u)

# material
ρ   = 1200.0       # density [kg/m³]
# mat = LinearElastic(350e6, 0.3, 0.0002) # nylon-cpated TPU
# mat = LinearElastic(20e6, 0.3, 0.001) # soft TPU
mat = LinearElastic(80e6, 0.3, 0.0008) # soft TPU
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

# Smooth sinusoidal ramp: ½(1 − cos(πt/T)) for t ≤ T, 1 beyond.
T_morph = 2.0   # morphing duration [s]
T_sim   = 2.0   # morph stage duration [s]
T_act   = 2.0   # actuation ramp duration [s]
T_pres  = 2.0   # pressurization ramp duration [s]
T_total = T_sim + T_act + T_pres
Δt      = 0.001 # initial time step [s]
smoothramp(t, T) = t < T ? 0.5 * (1 - cos(π * t / T)) : 1.0
ramp(t) = smoothramp(t, T_morph)

prescribed_u = generate_boundary_function(grid, "edge"; ramp=ramp)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0, [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# allocate arrays
N_dof = ndofs(dh)
free  = ch.free_dofs
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

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable).
# These are needed by `m_fac`, the initial `K_eff` fill, and `bufs` below, so
# they live at global scope (the per-case `let` block only carries the loop).
α_hht   = -0.3
γ_hht   = 0.5 - α_hht
β_hht   = (1 - α_hht)^2 / 4
α_damp  = 10.0    # mass-proportional Rayleigh damping coefficient [1/s]

m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat, sdofs, ke, re, u_e)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

bufs = (; K_int, r_int, K_plv, F_plv, K_pact, F_pact, K_plvpact, F_plvpact, M, K_eff,
          res, rhs, δu, u_trial, a_new, v_new, Ma, Mv, F_lu, free, g_old,
          sdofs, ke, re, u_e, α_hht, γ_hht, β_hht, α_damp)

# HHT-α Newton corrector (with backtracking line search) for one time step, valid
# for every stage.  The follower load F_ext = Plv·F_plv + Pact·F_pact − Pact·F_plvpact
# and its tangent K_int − Plv·K_plv − Pact·K_pact + Pact·K_plvpact are assembled on
# the three surfaces; the morph stage is recovered by passing Pact = 0.  `u_new` is
# updated in place; returns (converged, iters).  All vector arithmetic is in-place.
function solve_step!(u_new, ũ, ṽ, Plv, Pact, Δt, dh, scv, mat, ch,
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
        # Line search stalled (even α_ls=1/256 did not reduce the residual): bail
        # and let the outer loop reject the step (halve Δt) instead of burning all
        # max_iter full-tangent sweeps.
        ls_ok || break
    end
    return converged, iters
end

# Actuator-pressure targets to sweep [mmHg]; reused by the plotting section below.
pact_cases = 0:400:400

let
tol      = 1e-4
max_iter = 50
Δt_min   = 1e-7
Δt_max   = 0.1

# scaling and pressure
Pa2mmHg  = 0.00750062      # Pa/mmHg
m3_to_ml = 1.0e6           # m³ → ml
p_hold   = 0.0  / Pa2mmHg  # Plv held during morph + actuation [Pa]
Plv_max  = 150.0 / Pa2mmHg  # final ventricular pressure after ramp [Pa]
save_vtk = false

# run a few actuator-pressure targets [mmHg].  `Pact_mmHg` is the case identifier;
# the scheduled actuator pressure inside the loop is a separate `Pact` [Pa].
for Pact_mmHg in pact_cases
    Pact_max = Pact_mmHg / Pa2mmHg  # actuator pressure target [Pa]
    # schedule(t) → (morph_arg, Plv, Pact) at global time t.  The edge morph is driven
    # by `ramp(5t)` (completes early, well within the morph stage) then frozen.
    function schedule(t)
        if t < T_sim
            return (5t, p_hold * ramp(t), 0.0)
        elseif t < T_sim + T_act
            return (T_morph, p_hold, Pact_max * smoothramp(t - T_sim, T_act))
        else
            return (T_morph, Plv_max * smoothramp(t - T_sim - T_act, T_pres), Pact_max)
        end
    end

    # Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0).
    # g_old is a shared global buffer, so it must be re-zeroed for each Pact case.
    u = zeros(N_dof); apply!(u, ch)
    v = zeros(N_dof)
    a = zeros(N_dof)
    fill!(g_old, 0.0)

    pvd = save_vtk ? paraview_collection("minilimo-dynamic-actuation-pact$(Pact_mmHg)") : nothing
    vtk_step = Ref(-1)
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
        VTKGridFile("minilimo-dynamic-actuation-pact$(Pact_mmHg)-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u)
            Ferrite.write_node_data(vtk, resu, "ru")
            Ferrite.write_node_data(vtk, resθ, "rθ")
            Ferrite.write_node_data(vtk, d,  "director")
            Ferrite.write_node_data(vtk, G3, "G3")
            for ID in 1:3; color(vtk, grid, "SRF_$ID"); end
            pvd[t] = vtk
        end
    end

    save_vtk && write_vtk!(pvd, vtk_step, dh, scv, grid, u, res, resu, resθ, 0.0)

    println("minilimo inflation for Pact $Pact_mmHg mmHg")
    @printf("%-6s  %-8s  %-8s  %-8s  %-6s  %-10s\n", "step", "t [s]", "Plv [mmHg]", "Pact [mmHg]", "iters", "Δt")

    # Pressurization-stage history: (t, cavity volume, Plv, Pact) rows, cavity volume
    # measured over the endocardium as in the 3D–0D coupling.  Written to CSV at the end.
    hist = NTuple{4,Float64}[]

    un = zeros(N_dof)
    let t = 0.0; step = 0; Δt_cur = Δt
    @time while t < T_total - 1e-10
        t_new = min(t + Δt_cur, T_total)
        marg, Plv, Pact = schedule(t_new)

        @. ũ = u + Δt_cur * v + (Δt_cur^2 * (0.5 - β_hht)) * a
        @. ṽ = v + (Δt_cur * (1 - γ_hht)) * a

        u_new .= ũ
        Ferrite.update!(ch, marg)
        apply!(u_new, ch)

        converged, iters = solve_step!(u_new, ũ, ṽ, Plv, Pact, Δt_cur, dh, scv, mat, ch,
                                       Plv_srf, Pact_srf, PlvPact_srf, bufs; max_iter=max_iter, tol=tol)

        if converged
            step += 1
            @. a = (u_new - ũ) / (β_hht * Δt_cur^2)
            @. v = ṽ + (Δt_cur * γ_hht) * a
            mul!(Mv, M, v)
            @. g_old = α_damp * Mv + r_int - Plv * F_plv - Pact * F_pact + Pact * F_plvpact
            u .= u_new; t = t_new
            if t ≥ T_sim + T_act - 1e-10   # pressurization stage: log cavity volume + pressures
                Vlv = -2 * compute_volume(dh, scv, u; cellset=Plv_srf) * m3_to_ml
                push!(hist, (t, Vlv, Plv * Pa2mmHg, Pact * Pa2mmHg))
            end
            Δt_cur = min(Δt_cur * 1.2, Δt_max)
            if step % 4 == 0
                save_vtk && write_vtk!(pvd, vtk_step, dh, scv, grid, u, res, resu, resθ, t)
                @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d  %-10.4e\n",
                        step, t, Plv * Pa2mmHg, Pact * Pa2mmHg, iters, Δt_cur)
            end
        else
            Δt_cur /= 2
            Δt_cur < Δt_min && error("minimum Δt reached at t=$(round(t, digits=4)) s")
        end
    end
        un .= u
        save_vtk && write_vtk!(pvd, vtk_step, dh, scv, grid, un, res, resu, resθ, T_total)
    end
    save_vtk && close(pvd)

    open("minilimo_pressurization_pact_$(Pact_mmHg).csv", "w") do io
        println(io, "t_s,Vlv_ml,Plv_mmHg,Pact_mmHg")
        for (t, V, Plv, Pact) in hist
            @printf(io, "%.6f,%.6f,%.6f,%.6f\n", t, V, Plv, Pact)
        end
    end
end
end

# Pressure–volume (Plv vs Vlv) curves from the pressurization stage of each Pact
# case, read back from the CSVs written above.
using Plots

# Read the (Vlv_ml, Plv_mmHg) columns of one pressurization CSV; skips the header.
function read_pv(path)
    Vlv = Float64[]; Plv = Float64[]
    for line in Iterators.drop(eachline(path), 1)
        cols = split(line, ',')
        push!(Vlv, parse(Float64, cols[2]))
        push!(Plv, parse(Float64, cols[3]))
    end
    return Vlv, Plv
end

pv = plot(xlabel="Vlv [ml]", ylabel="Plv [mmHg]", legend=:topleft, lw=2)
for Pact_mmHg in pact_cases
    fname = "minilimo_pressurization_pact_$(Pact_mmHg).csv"
    isfile(fname) || (@warn "missing $fname, skipping"; continue)
    Vlv, Plv = read_pv(fname)
    plot!(pv, Vlv, Plv, label="Pact = $(Pact_mmHg) mmHg", lw=2, marker=:circle, ms=3)
end
# savefig(pv, "minilimo_pressurization_pv.png")