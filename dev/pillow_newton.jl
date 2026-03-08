using FerriteShells
using LinearAlgebra
using Printf

# Square pillow clamped on all four edges, inflated by follower pressure.
# Exploit double symmetry: model the quarter domain [0,L/2]×[0,L/2].
# Symmetry BCs: u_x=0 on x=0, u_y=0 on y=0.
# Solve with load-stepping Newton-Raphson.

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

# Initial z-perturbation vanishing on all four boundary edges of the quarter domain:
# sin(πx/(L/2)) * sin(πy/(L/2)) = 0 at x=0, x=L/2, y=0, y=L/2.
u = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x    = getcoordinates(cell)
    dofs = celldofs(cell)
    for i in eachindex(x)
        u[dofs[3i]] = 1e-4 * sin(π * x[i][1] / (L/2)) * sin(π * x[i][2] / (L/2))
    end
end
apply!(u, ch)

ph       = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])
N        = ndofs(dh)
p_max    = 500.0
n_steps  = 50
tol      = 1e-8
max_iter = 20

println("Load-stepping Newton-Raphson (quarter symmetry)")
println("  pressure | deflection | iters")
for p in range(0.0, p_max; length=n_steps+1)[2:end]
    converged = false
    n_iter    = 0
    for iter in 1:max_iter
        K_mem = allocate_matrix(dh)
        r_int = zeros(N)
        assemble_global!(K_mem, r_int, dh, scv, u, mat)

        F_ext = zeros(N)
        assemble_pressure_global!(F_ext, dh, scv, u, p)

        K_pres = allocate_matrix(dh)
        assemble_pressure_tangent_global!(K_pres, dh, scv, u, p)

        K_eff = K_mem - K_pres
        rhs   = F_ext .- r_int   # out-of-balance force

        apply_zero!(K_eff, rhs, ch)

        n_iter = iter
        if norm(rhs) < tol
            converged = true
            n_iter    = iter - 1
            break
        end

        u .+= K_eff \ rhs
    end
    converged || @warn "p=$p did not converge in $max_iter iterations"
    w = evaluate_at_points(ph, dh, u, :u)[1][3]
    @printf("  %8.2f | %10.4e | %d\n", p, w, n_iter)
end
