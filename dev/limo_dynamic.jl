using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK
using QuadGK

# HHT-α implicit time integration for the limo morphing step.
#
# Edge nodes are driven from flat to target elliptic arc geometry over [0, T_morph]
# using a smooth sinusoidal ramp.  A follower pressure (same ramp) is applied
# simultaneously.  Mass-proportional Rayleigh damping C = α_damp·M is included.
#
# HHT-α with damping and follower pressure:
#   g(u,v,p) = C·v + r_int(u) − p·F_p(u)
#   R = M·ä_{n+1} + (1−α)·g(u_{n+1},v_{n+1},p_{n+1}) + α·g_old = 0
#   γ = ½ − α,  β = (1−α)²/4   (2nd-order, unconditionally stable for α ∈ [−⅓,0])
#   K_eff = M·[1/(βΔt²) + (1−α)·α_damp·γ/(βΔt)] + (1−α)·(K_int − p_{n+1}·K_plv)
#   Predictor:  ũ = u_n + Δt v_n + Δt²(½−β) a_n
#               ṽ = v_n + Δt(1−γ) a_n
#   Corrector:  solve K_eff δu = −R, then u ← u + δu, apply BCs

function make_quarter_pillow_grid(n; primitive=Quadrilateral)
    corners = [Vec{2}((-0.05058799, 0.000)), Vec{2}(( 0.05058799, 0.000)),
               Vec{2}(( 0.05058799, 0.109)), Vec{2}((-0.05058799, 0.109))]
    grid = shell_grid(generate_grid(primitive, (n, n), corners))
    return grid
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

function assemble_all!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble!(asm_i, sd, ke_i, re_i)
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

function assemble_pressure_all!(K_p, F_p, dh, scv, u)
    n_e  = ndofs_per_cell(dh)
    ke_p = zeros(n_e, n_e); re_p = zeros(n_e)
    asm  = start_assemble(K_p)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        assemble_pressure!(re_p, scv, u_e, 1.0)
        assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm, sd, ke_p)
        @views F_p[sd] .+= re_p
    end
end

# Material: silicone rubber (E=0.35 MPa, ν=0.3, t=2 mm)
const ρ   = 1200.0       # density [kg/m³]
mat = LinearElastic(0.35e6, 0.3, 0.002)

# fname = "/home/marin/Workspace/HHH/code/miniLIMO/p6/geom_julia.inp"
# grid  = get_ferrite_grid(fname)
grid = make_quarter_pillow_grid(32; primitive=Quadrilateral)

addnodeset!(grid, "edge", x -> x[2] ≈ 0)
addfacetset!(grid, "sym", x -> (x[2] ≈ 0.109) || (abs(x[1]) ≈ 0.05058799))

ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# Smooth sinusoidal ramp: λ(t) = ½(1 − cos(πt/T_morph)) for t ≤ T_morph, 1 beyond.
# Zero velocity at t=0 and t=T_morph avoids impulsive loads.
const T_morph = 10.0   # morphing duration [s]
const T_sim   = 10.0   # total simulation  [s]
const Δt      = 0.001  # time step         [s]
const n_steps = round(Int, T_sim / Δt)

ramp(t) = t < T_morph ? 0.5 * (1 - cos(π * t / T_morph)) : 1.0

function generate_boundary_function(grid, nodeset)
    top_nodes = get_node_coordinate.(getnodes(grid, nodeset))
    idx = sortperm(top_nodes)
    node_sorted = top_nodes[idx]
    Ar = 80.2 / 55.2
    x, y = getindex.(node_sorted, 1), getindex.(node_sorted, 2)
    x_new, y_new = map_initial(x, y, Ar)
    Xs  = vcat(x', y')
    dXs = vcat(x_new' .- x', y_new')
    (x, t) -> begin
        i = findmin(dropdims(sum(abs2, Xs .- [x[1], x[2]], dims=1), dims=1))[2]
        ramp(t) .* dXs[:, i]
    end
end

prescribed_u = generate_boundary_function(grid, "edge")

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), (x,t) -> prescribed_u(x, t), [1,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0,      [2]))
add!(ch, Dirichlet(:θ, getnodeset(grid, "edge"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "sym"), x -> 0.0,      [3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

const N_dof = ndofs(dh)
const free  = ch.free_dofs

# HHT-α parameters  (α = −0.3: strong high-frequency damping, still stable)
const α_hht   = -0.3
const γ_hht   = 0.5 - α_hht
const β_hht   = (1 - α_hht)^2 / 4
const α_damp  = 2.0    # mass-proportional Rayleigh damping coefficient [1/s]
const tol      = 1e-6
const max_iter = 20

# Pressure ramp: same sinusoidal profile as morphing, up to p_max [Pa]
const Pa2mmHg = 0.00750062
const p_max   = 6.0 / Pa2mmHg   # 6 mmHg → Pa

K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
K_plv = allocate_matrix(dh)
M     = allocate_matrix(dh)
r_int = zeros(N_dof)
F_plv = zeros(N_dof)
g_old = zeros(N_dof)   # (1−α) term from previous step: C·v_n + r_int(u_n) − p_n·F_p(u_n)
R     = zeros(N_dof)
δu    = zeros(N_dof)

assemble_mass!(M, dh, scv, ρ, mat)

# Effective mass factor (M coefficient in K_eff, updated when Δt changes)
m_fac(Δt) = 1 / (β_hht * Δt^2) + (1 - α_hht) * α_damp * γ_hht / (β_hht * Δt)

# Initial symbolic LU factorisation (reused each Newton step via lu!)
assemble_all!(K_int, r_int, dh, scv, zeros(N_dof), mat)
K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* K_int.nzval
let tmp = zeros(N_dof); apply_zero!(K_eff, tmp, ch); end
F_lu = lu(K_eff)

# Initial state: at rest, flat reference geometry; g_old = 0 (u=v=0, p=0)
u = zeros(N_dof); apply!(u, ch)
v = zeros(N_dof)
a = zeros(N_dof)

pvd = paraview_collection("limo_dynamic")
vtk_step = Ref(0)
VTKGridFile("limo_dynamic-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

@printf("%-6s  %-8s  %-8s  %-8s  %-6s\n", "step", "t [s]", "λ", "p [mmHg]", "iters")

p = 0.0   # current pressure [Pa]

for step in 1:n_steps
    t_new = step * Δt
    p_new = p_max * ramp(t_new)

    # Predictor
    ũ = u .+ Δt .* v .+ (Δt^2 * (0.5 - β_hht)) .* a
    ṽ = v .+ (Δt * (1 - γ_hht)) .* a

    u_new = copy(ũ)
    Ferrite.update!(ch, t_new)
    apply!(u_new, ch)

    converged = false; iters = 0

    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u_new, mat)
        assemble_pressure_all!(K_plv, F_plv, dh, scv, u_new)

        a_new = (u_new .- ũ) ./ (β_hht * Δt^2)
        v_new = ṽ .+ (Δt * γ_hht) .* a_new

        # HHT residual: M ä + (1−α)[C v + r_int − p F_p] + α g_old = 0
        R .= M * a_new .+ (1 - α_hht) .* (α_damp .* (M * v_new) .+ r_int .- p_new .* F_plv) .+ α_hht .* g_old
        apply_zero!(R, ch)

        norm(@views R[free]) < tol && (converged = true; break)

        K_eff.nzval .= M.nzval .* m_fac(Δt) .+ (1 - α_hht) .* (K_int.nzval .- p_new .* K_plv.nzval)
        rhs = .-R
        apply_zero!(K_eff, rhs, ch)
        lu!(F_lu, K_eff)
        ldiv!(δu, F_lu, rhs)
        u_new .+= δu
        apply!(u_new, ch)
    end

    !converged && @warn "Step $step (t=$(round(t_new, digits=3)) s): no convergence in $max_iter iters"

    a .= (u_new .- ũ) ./ (β_hht * Δt^2)
    v .= ṽ .+ (Δt * γ_hht) .* a
    # store g_old = C·v + r_int(u) − p·F_p(u) for the α-offset in next step
    g_old .= α_damp .* (M * v) .+ r_int .- p_new .* F_plv
    p  = p_new
    u .= u_new

    vtk_step[] += 1
    VTKGridFile("limo_dynamic-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t_new] = vtk
    end

    step % 10 == 0 && @printf("%-6d  %-8.3f  %-8.4f  %-8.4f  %-6d\n",
                               step, t_new, ramp(t_new), p_new * Pa2mmHg, iters)
end
close(pvd)
