using FerriteShells
using OrdinaryDiffEq
using LinearAlgebra

# Square membrane clamped on all four edges, inflated by follower pressure.
# The equilibrium path is traced by the ODE:
#   K_eff(u, p) · du/dp = F(u)
# where p is pressure (the "time"), K_eff = K_membrane - K_pressure_tangent,
# and F(u) is the unit-pressure follower force. This is obtained by
# differentiating R_int(u) = p·F(u) w.r.t. p.

function make_pillow_grid(n; L=1.0)
    nodes = [Vec{3}((L*(i/n - 0.5), L*(j/n - 0.5), 0.0)) for j in 0:n for i in 0:n]
    cells = [Quadrilateral((j*(n+1)+i+1, j*(n+1)+i+2, (j+1)*(n+1)+i+2, (j+1)*(n+1)+i+1))
             for j in 0:n-1 for i in 0:n-1]
    grid = Grid(cells, Node.(nodes))
    addnodeset!(grid, "boundary",
        x -> isapprox(abs(x[1]), L/2, atol=1e-10) || isapprox(abs(x[2]), L/2, atol=1e-10))
    return grid
end

# Global pressure residual: r += ∫ p N_I (a₁ × a₂) dξ  (element-level → scattered)
function assemble_pressure_global!(r, dh, scv, u, p)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        FerriteShells.reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        assemble_pressure!(re, scv, x, u_e, p)
        r[celldofs(cell)] .+= re
    end
end

# Global pressure tangent: K += p · ∂F/∂u  (via ForwardDiff element-by-element)
function assemble_pressure_tangent_global!(K, dh, scv, u, p)
    ke        = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        FerriteShells.reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        assemble_pressure_tangent!(ke, scv, x, u_e, p)
        assemble!(assembler, celldofs(cell), ke)
    end
end

function assemble_tangent_and_residual!(K, r, dh, scv, u, mat)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    re  = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        x = getcoordinates(cell)
        fill!(ke, 0.0); fill!(re, 0.0)
        FerriteShells.reinit!(scv, cell) # prepares reference geometry
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, x, u_e, mat)
        membrane_residuals_KL!(re, scv, x, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

# Setup
n   = 8
L   = 1.0
mat = LinearElastic(1.0e6, 0.3, 1e-3)   # E, ν, thickness

grid = make_pillow_grid(n; L)
ip   = Lagrange{RefQuadrilateral, 1}()
qr   = QuadratureRule{RefQuadrilateral}(2)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"), x -> zero(x), [1, 2, 3]))
close!(ch)
Ferrite.update!(ch, 0.0)

# Initial condition: small z-perturbation that satisfies the clamped BCs.
# sin(π(x+L/2)/L)·sin(π(y+L/2)/L) is zero on all four edges.
# Without this, K_eff is singular at u=0 (flat membrane has no z-stiffness).
u0 = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x    = getcoordinates(cell)
    dofs = celldofs(cell)
    for i in eachindex(x)
        u0[dofs[3i]] = 1e-4 * sin(π*(x[i][1] + L/2)/L) * sin(π*(x[i][2] + L/2)/L)
    end
end
apply!(u0, ch)   # zero out any boundary DOFs touched by the loop above

# ODE: du/dp = K_eff(u, p)⁻¹ · F(u)
# K_eff = K_membrane(u) − K_pressure_tangent(u, p)
# F(u)  = unit-pressure follower force at current configuration
function ode!(du, u, _, p)
    N = ndofs(dh)

    K_mem = allocate_matrix(dh)
    r     = zeros(N)
    assemble_tangent_and_residual!(K_mem, r, dh, scv, u, mat)

    K_pres = allocate_matrix(dh)
    assemble_pressure_tangent_global!(K_pres, dh, scv, u, p)

    K_eff = K_mem - K_pres          # effective tangent (sparse subtraction)

    F = zeros(N)
    assemble_pressure_global!(F, dh, scv, u, 1.0)   # unit-pressure RHS

    apply_zero!(K_eff, F, ch)       # enforce du_boundary/dp = 0

    du .= K_eff \ F
end

p_max = 500.0
prob  = ODEProblem(ode!, u0, (0.0, p_max))
sol   = solve(prob, Rodas5(), reltol=1e-4, abstol=1e-6)

# Post-process: central z-deflection vs pressure
ph       = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])
p_vals   = sol.t
w_center = [evaluate_at_points(ph, dh, sol(p), :u)[1][3] for p in p_vals]

println("pressure | central deflection")
for (p, w) in zip(p_vals, w_center)
    @printf("  %6.1f | %.4e\n", p, w)
end
