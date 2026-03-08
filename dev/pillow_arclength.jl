using FerriteShells
using OrdinaryDiffEq
using LinearAlgebra
using Printf

# Square pillow clamped on all four edges, inflated by follower pressure.
# Exploit double symmetry: model the quarter domain [0,L/2]×[0,L/2].
# Symmetry BCs: u_x=0 on x=0, u_y=0 on y=0.
#
# Arc-length (Riks) method via OrdinaryDiffEq.
# State: z = [u; λ]  (N+1 dimensional), independent variable: arc-length s.
#
# Equilibrium: R_int(u) = λ·F(u)  →  differentiating w.r.t. s:
#   K_eff · du/ds = F(u) · dλ/ds,  K_eff = K_mem − λ·K_pres
# Let du/ds = v · dλ/ds  →  v = K_eff⁻¹ · F(u).
# Arc-length constraint: ‖du/ds‖² + (dλ/ds)² = 1
#   →  dλ/ds = 1/√(‖v‖²+1),  du/ds = v · dλ/ds.

function make_quarter_pillow_grid(n; L=1.0)
    nodes = [Vec{3}((L/2*i/n, L/2*j/n, 0.0)) for j in 0:n for i in 0:n]
    cells = [Quadrilateral((j*(n+1)+i+1, j*(n+1)+i+2, (j+1)*(n+1)+i+2, (j+1)*(n+1)+i+1))
             for j in 0:n-1 for i in 0:n-1]
    grid = Grid(cells, Node.(nodes))
    addnodeset!(grid, "clamped", x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addnodeset!(grid, "sym_x",   x -> isapprox(x[1], 0.0,  atol=1e-10))
    addnodeset!(grid, "sym_y",   x -> isapprox(x[2], 0.0,  atol=1e-10))
    return grid
end

function assemble_global!(K, r, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e)
    re  = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = reinterpret(Vec{3, Float64}, u[celldofs(cell)])
        membrane_tangent!(ke, scv, x, u_e, mat)
        membrane_residuals!(re, scv, x, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
end

function assemble_pressure_global!(r, dh, scv, u, p)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = reinterpret(Vec{3, Float64}, u[celldofs(cell)])
        assemble_pressure!(re, scv, x, u_e, p)
        r[celldofs(cell)] .+= re
    end
end

function assemble_pressure_tangent_global!(K, dh, scv, u, p)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e)
    asm = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = reinterpret(Vec{3, Float64}, u[celldofs(cell)])
        assemble_pressure_tangent!(ke, scv, x, u_e, p)
        assemble!(asm, celldofs(cell), ke)
    end
end

n   = 8
L   = 1.0
mat = LinearElastic(1.0e6, 0.3, 1e-3)

grid = make_quarter_pillow_grid(n; L)
ip   = Lagrange{RefQuadrilateral, 1}()
qr   = QuadratureRule{RefQuadrilateral}(2)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "clamped"), x -> 0.0, [1, 2, 3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_x"),   x -> 0.0, [1]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_y"),   x -> 0.0, [2]))
close!(ch)
Ferrite.update!(ch, 0.0)

# Initial z-perturbation vanishing on all four boundary edges of the quarter domain.
u0 = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x    = getcoordinates(cell)
    dofs = celldofs(cell)
    for i in eachindex(x)
        u0[dofs[3i]] = 1e-4 * sin(π * x[i][1] / (L/2)) * sin(π * x[i][2] / (L/2))
    end
end
apply!(u0, ch)

N     = ndofs(dh)
p_max = 500.0

function arclength_ode!(dz, z, _, s)
    u = collect(@view z[1:N])   # copy so reinterpret works on a plain Vector
    λ = z[N+1]

    K_mem = allocate_matrix(dh)
    r_int = zeros(N)
    assemble_global!(K_mem, r_int, dh, scv, u, mat)

    K_pres = allocate_matrix(dh)
    assemble_pressure_tangent_global!(K_pres, dh, scv, u, λ)

    K_eff = K_mem - K_pres

    F = zeros(N)
    assemble_pressure_global!(F, dh, scv, u, 1.0)   # unit-pressure tangent vector

    apply_zero!(K_eff, F, ch)   # enforce dλ/ds·du/ds = 0 at constrained DOFs

    v    = K_eff \ F
    dλds = 1.0 / sqrt(dot(v, v) + 1.0)

    dz[1:N] .= v .* dλds
    dz[N+1]  = dλds
end

# Terminate when the load parameter reaches p_max.
condition(z, s, integrator) = z[N+1] - p_max
affect!(integrator)          = terminate!(integrator)
cb = ContinuousCallback(condition, affect!)

z0   = [u0; 0.0]
prob = ODEProblem(arclength_ode!, z0, (0.0, 1e6))
sol  = solve(prob, Rodas5(), reltol=1e-4, abstol=1e-6, callback=cb)

ph = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])

println("Arc-length (Riks) method (quarter symmetry)")
println("  pressure | deflection")
for i in eachindex(sol.t)
    u_sol = sol.u[i][1:N]
    λ_sol = sol.u[i][N+1]
    w     = evaluate_at_points(ph, dh, u_sol, :u)[1][3]
    @printf("  %8.2f | %10.4e\n", λ_sol, w)
end
