using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Cantilever beam (L×W flat plate, clamped at x=0) under tip shear in z.
# Membrane + KL bending (Q9). Geometrically nonlinear, large deflections.
# Solved with load-stepping Newton-Raphson + Armijo backtracking.
#
# Clamped BC for KL elements: fixing only w=0 at x=0 leaves a zero-curvature
# rigid-rotation mode w=α·x in the null space.  Fix: also clamp the entire
# first element column (x=0 … Δx), which enforces dw/dx = 0 at x=0.

function make_cantilever(dims; L=10.0, W=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L, 0.0)), Vec{2}((L, W)), Vec{2}((0.0, W))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, dims, corners))
    Δx = L / dims[1]
    addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
    addfacetset!(grid, "traction", x -> x[1] ≈ L)
    addnodeset!(grid, "clamped_col", x -> x[1] ≤ Δx + 1e-10)
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
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, x, u_e, mat)
        membrane_residuals_KL!(re, scv, x, u_e, mat)
        bending_tangent_KL!(ke, scv, x, u_e, mat)
        bending_residuals_KL!(re, scv, x, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
end

function assemble_residual_only!(r, dh, scv, u, mat)
    fill!(r, 0.0)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        membrane_residuals_KL!(re, scv, x, u_e, mat)
        bending_residuals_KL!(re, scv, x, u_e, mat)
        r[celldofs(cell)] .+= re
    end
end

dims = (10, 1)
L    = 10.0
W    = 1.0
mat  = LinearElastic(1.2e6, 0.0, 0.1)

grid = make_cantilever(dims; L, W)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
fqr  = FacetQuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(dh.grid, "clamped"),    x -> zero(x), [1,2,3]))
add!(ch, Dirichlet(:u, getnodeset(dh.grid, "clamped_col"), x -> 0.0,     [3]))
close!(ch)
Ferrite.update!(ch, 0.0)

N = ndofs(dh)
K = allocate_matrix(dh)
r = zeros(N)
r_trial = zeros(N)
rhs = zeros(N)

# Dead-load traction (constant, assembled once)
F_ext = zeros(N)
assemble_traction!(F_ext, dh, getfacetset(grid, "traction"), ip, fqr, (0.0, 0.0, 4.0))

n_steps  = 20
tol      = 1e-6
max_iter = 30

ph  = PointEvalHandler(grid, [Vec{3}((L, W/2, 0.0))])
pvd = paraview_collection("cantilever_newton")
println("Cantilever Newton (membrane + KL bending, large deflection)")
println("  load | tip w | iters")

u = zeros(N)
for step in 1:n_steps
    λ      = step / n_steps
    F      = λ .* F_ext
    u_prev = copy(u)
    converged = false
    n_iter = 0
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)
        @. rhs = F - r
        apply_zero!(K, rhs, ch)
        rhs_norm = norm(rhs)
        if rhs_norm < tol
            converged = true
            n_iter = iter - 1
            break
        end
        n_iter = iter
        du = K \ rhs
        α = 1.0
        for _ in 1:8
            assemble_residual_only!(r_trial, dh, scv, u .+ α .* du, mat)
            @. rhs = F - r_trial
            for dof in ch.prescribed_dofs; rhs[dof] = 0.0; end
            norm(rhs) < rhs_norm && break
            α /= 2
        end
        u .+= α .* du
    end
    if !converged
        u .= u_prev
        @warn "step $step (λ=$λ) did not converge, rolling back"
    end
    w = evaluate_at_points(ph, dh, u, :u)[1][3]
    @printf("  %.2f | %10.4e | %d\n", λ, w, n_iter)
    VTKGridFile("cantilever-$step", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[step] = vtk
    end
end
vtk_save(pvd)
