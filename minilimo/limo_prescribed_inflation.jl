using FerriteShells, LinearAlgebra, Printf, WriteVTK, QuadGK

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

# Residual-only assembly (no tangent) for the backtracking line search — the
# expensive MITC/ForwardDiff element tangent is only needed for the Newton
# direction, not for evaluating the residual at a trial step.
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

# Pressure residual only (no follower tangent) for the line search.
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
