using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK

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

function color(vtk, grid, cellset)
    z = zeros(Ferrite.getncells(grid))
    z[collect(Ferrite.getcellset(grid, cellset))] .= 1.0
    write_cell_data(vtk, z, cellset)
end

function bisect(f, θ_lo, θ_hi; tolerance=1e-8)
    θ_mid = (θ_lo + θ_hi) / 2
    while θ_hi - θ_lo > tolerance
        θ_mid = (θ_lo + θ_hi) / 2
        f(θ_mid) * f(θ_lo) < 0 ? (θ_hi = θ_mid) : (θ_lo = θ_mid)
    end
    return θ_mid
end

function find_points(x, y, A, B, L)
    N = length(x)
    x_new = similar(x); y_new = similar(y)
    x_min = minimum(x)
    for i in (1, N)
        θ = (x[i] - x_min) * π / L
        x_new[i] = -A * cos(θ); y_new[i] = -B * sin(θ)
    end
    lengths = @views sqrt.((x[2:end] .- x[1:end-1]).^2 .+ (y[2:end] .- y[1:end-1]).^2)
    θ0 = 0.0
    for i in 1:N-2
        x0, y0, d = x_new[N-i+1], y_new[N-i+1], lengths[N-i]
        θ0 = bisect(θ0, π) do θ
            sqrt((A*cos(θ)-x0)^2 + (B*sin(θ)-y0)^2) - d
        end
        x_new[N-i] = A*cos(θ0); y_new[N-i] = B*sin(θ0)
    end
    x_new, y_new
end

function map_initial(x, y, Ar)
    L = maximum(x) - minimum(x)
    ds(θ, a) = sqrt(a^2*sin(θ)^2 + (a/Ar)^2*cos(θ)^2)
    find_a(a) = quadgk(θ -> ds(θ, a), 0, π)[1] - L
    a0 = bisect(find_a, 0.0, L)
    a = bisect(0.98*a0, 1.08*a0) do a
        xi, yi = find_points(x, y, a, a/Ar, L)
        @views sum(sqrt.((xi[2:end].-xi[1:end-1]).^2 .+ (yi[2:end].-yi[1:end-1]).^2)) - L
    end
    find_points(x, y, a, a/Ar, L)
end

# Rectangular approximation of the miniLIMO geometry without rounded edges.
function make_minilimo_grid(;
    nx_left=3, nx_act=10, nx_right=3,
    ny_bot=1, ny_act=14, ny_top=2,
    W=0.10118, H=0.109, x_act=0.035, y_lo=0.004, y_hi=0.09,
    Np=1)

    @assert nx_act % Np == 0 "nx_act ($nx_act) must be divisible by Np ($Np)"
    nx_per_pouch = nx_act ÷ Np

    Lx = W / 2
    xs = vcat(range(-Lx,   -x_act, nx_left + 1),
              range(-x_act,  x_act, nx_act  + 1)[2:end],
              range( x_act,    Lx,  nx_right + 1)[2:end])
    ys = vcat(range(0.0,  y_lo, ny_bot + 1),
              range(y_lo, y_hi, ny_act + 1)[2:end],
              range(y_hi,   H,  ny_top + 1)[2:end])
    nx = length(xs) - 1;  ny = length(ys) - 1

    ins(v) = (w = similar(v, 2length(v)-1); w[1:2:end]=v; w[2:2:end]=(v[1:end-1].+v[2:end])./2; w)
    xs_f = ins(xs);  ys_f = ins(ys)

    endo_node(px, py) = py * (2nx + 1) + px + 1
    n_endo = (2nx + 1) * (2ny + 1)
    endo_coords = [Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)) for py in 0:2ny for px in 0:2nx]

    py_lo = 2ny_bot;  py_hi = 2(ny_bot + ny_act)

    act_nodes = [Dict{Tuple{Int,Int},Int}() for _ in 1:Np]
    act_coords = Vec{3,Float64}[]
    for p in 1:Np
        px_lo_p = 2(nx_left + (p-1)*nx_per_pouch)
        px_hi_p = 2(nx_left + p*nx_per_pouch)
        for py in py_lo:py_hi, px in px_lo_p:px_hi_p
            if px == px_lo_p || px == px_hi_p || py == py_lo || py == py_hi
                act_nodes[p][(px, py)] = endo_node(px, py)
            else
                push!(act_coords, Vec{3}((xs_f[px+1], ys_f[py+1], 0.0)))
                act_nodes[p][(px, py)] = n_endo + length(act_coords)
            end
        end
    end

    function q9_endo(px, py)
        QuadraticQuadrilateral((
            endo_node(px,   py),   endo_node(px+2, py),
            endo_node(px+2, py+2), endo_node(px,   py+2),
            endo_node(px+1, py),   endo_node(px+2, py+1),
            endo_node(px+1, py+2), endo_node(px,   py+1),
            endo_node(px+1, py+1)))
    end
    function q9_act(p, px, py)
        QuadraticQuadrilateral((
            act_nodes[p][(px,   py)],   act_nodes[p][(px+2, py)],
            act_nodes[p][(px+2, py+2)], act_nodes[p][(px,   py+2)],
            act_nodes[p][(px+1, py)],   act_nodes[p][(px+2, py+1)],
            act_nodes[p][(px+1, py+2)], act_nodes[p][(px,   py+1)],
            act_nodes[p][(px+1, py+1)]))
    end

    srf1   = Int[]
    srf2_k = [Int[] for _ in 1:Np]
    endo_cells = QuadraticQuadrilateral[]
    for iy in 0:ny-1, ix in 0:nx-1
        push!(endo_cells, q9_endo(2ix, 2iy))
        cid    = length(endo_cells)
        ix_rel = ix - nx_left
        if iy >= ny_bot && iy < ny_bot + ny_act && ix_rel >= 0 && ix_rel < nx_act
            push!(srf2_k[ix_rel ÷ nx_per_pouch + 1], cid)
        else
            push!(srf1, cid)
        end
    end
    n_ec = length(endo_cells)

    srf3_k = [Int[] for _ in 1:Np]
    act_cells = QuadraticQuadrilateral[]
    for p in 1:Np
        for k in 1:nx_per_pouch
            ix = nx_left + (p-1)*nx_per_pouch + k - 1
            px = 2ix
            for iy in ny_bot:ny_bot+ny_act-1
                push!(act_cells, q9_act(p, px, 2iy))
                push!(srf3_k[p], n_ec + length(act_cells))
            end
        end
    end

    grid = Grid(vcat(endo_cells, act_cells), Node.(vcat(endo_coords, act_coords)))
    addcellset!(grid, "SRF_1", Set(srf1))
    srf2_all = Int[];  srf3_all = Int[]
    for k in 1:Np
        addcellset!(grid, "SRF_2_$k", Set(srf2_k[k]))
        addcellset!(grid, "SRF_3_$k", Set(srf3_k[k]))
        append!(srf2_all, srf2_k[k]);  append!(srf3_all, srf3_k[k])
    end
    addcellset!(grid, "SRF_2", Set(srf2_all))
    addcellset!(grid, "SRF_3", Set(srf3_all))
    addnodeset!(grid, "edge", x -> x[2] ≈ 0.0)
    addfacetset!(grid, "sym", x -> x[2] ≈ H || abs(x[1]) ≈ Lx)
    return grid
end

function assemble_all!(K_int, r_int, dh, scv, u, mat, sdofs, ke, re, u_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(ke, 0.0); fill!(re, 0.0)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        membrane_tangent_RM!(ke, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        assemble!(asm_i, sd, ke, re)
    end
end

# Residual-only assembly (no tangent) for the Δτ control sub-iteration — the
# expensive MITC/ForwardDiff element tangent is only needed to build the PTC
# direction, not to evaluate the residual at a trial pseudo-step.
function assemble_residual!(r_int, dh, scv, u, mat, sdofs, re, u_e)
    fill!(r_int, 0.0)
    for cell in CellIterator(dh)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(re, 0.0)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        @views r_int[sd] .+= re
    end
end

function assemble_mass!(M, dh, scv, ρ, mat)
    n_e = ndofs_per_cell(dh)
    me  = zeros(n_e, n_e)
    asm = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(me, 0.0)
        reinit!(scv, cell)
        mass_matrix!(me, scv, ρ, mat)
        assemble!(asm, shelldofs(cell), me)
    end
end

# Follower-pressure load vector + tangent (unit pressure) restricted to `cellset`.
function assemble_pressure_region!(K_p, F_p, dh, scv, u, cellset, sdofs, ke, re, u_e; Pᵢ=1.0)
    asm = start_assemble(K_p)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(ke, 0.0); fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, Pᵢ)
        assemble_pressure_tangent!(ke, scv, u_e, Pᵢ)
        assemble!(asm, sd, ke)
        @views F_p[sd] .+= re
    end
end

# Pressure residual only (no follower tangent) for the Δτ control sub-iteration.
function assemble_pressure_residual!(F_p, dh, scv, u, cellset, sdofs, re, u_e; Pᵢ=1.0)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh, cellset)
        sd = sdofs[Ferrite.cellid(cell)]
        reinit!(scv, cell)
        @views u_e .= u[sd]
        fill!(re, 0.0)
        assemble_pressure!(re, scv, u_e, Pᵢ)
        @views F_p[sd] .+= re
    end
end

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

function generate_boundary_function(grid, nodeset)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    Ar = 80.2 / 55.2
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Xs = vcat(x', y'); dXs = vcat(x_new' .- x', y_new')
    return function prescribed_u(x, t)
        idx = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        return morph_ramp(t) .* dXs[:, idx]
    end
end

prescribed_u = generate_boundary_function(grid, "edge")

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
