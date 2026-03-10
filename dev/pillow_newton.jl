using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Square pillow clamped on all four edges, inflated by follower pressure.
# Exploit double symmetry: model the quarter domain [0,L/2]×[0,L/2].
# Symmetry BCs: u_x=0 on x=0, u_y=0 on y=0.
# Solve with load-stepping Newton-Raphson + Armijo backtracking line search.
# Membrane + KL bending (Q9) for positive-definite stiffness at the flat state.

function make_quarter_pillow_grid(n; L=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n), corners))
    addnodeset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addnodeset!(grid, "sym_x", x -> isapprox(x[1], 0.0,  atol=1e-10))
    addnodeset!(grid, "sym_y", x -> isapprox(x[2], 0.0,  atol=1e-10))
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

function assemble_residual_only!(r_int, F_ext, dh, scv, u, mat, p)
    fill!(r_int, 0.0); fill!(F_ext, 0.0)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
        membrane_residuals_KL!(re, scv, x, u_e, mat)
        bending_residuals_KL!(re, scv, x, u_e, mat)
        r_int[celldofs(cell)] .+= re
        fill!(re, 0.0)
        assemble_pressure!(re, scv, x, u_e, p)
        F_ext[celldofs(cell)] .+= re
    end
end

function assemble_pressure_global!(r, dh, scv, u, p)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = u[celldofs(cell)]
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
        u_e = u[celldofs(cell)]
        assemble_pressure_tangent!(ke, scv, x, u_e, p)
        assemble!(asm, celldofs(cell), ke)
    end
end

n   = 8
L   = 1.0
mat = LinearElastic(1.0e6, 0.3, 1e-3)

grid = make_quarter_pillow_grid(n; L)
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"), x -> 0.0, [3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_x"), x -> 0.0, [1]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_y"), x -> 0.0, [2]))
close!(ch)
Ferrite.update!(ch, 0.0)

# Initial z-perturbation vanishing on boundary edges.
u = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x    = getcoordinates(cell)
    dofs = celldofs(cell)
    for i in eachindex(x)
        u[dofs[3i]] = 1e-2 * sin(π * x[i][1] / (L/2)) * sin(π * x[i][2] / (L/2))
    end
end
apply!(u, ch)

ph       = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])
N        = ndofs(dh)
p_max    = 500.0
n_steps  = 50
tol      = 1e-6
max_iter = 30

K_mem  = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
r_int  = zeros(N)
F_ext  = zeros(N)
rhs    = zeros(N)
r_trial = zeros(N)
F_trial = zeros(N)

pvd = paraview_collection("pillow_newton")
println("Load-stepping Newton-Raphson (quarter symmetry, membrane + KL bending)")
println("  pressure | deflection | iters")
for (t,p) in enumerate(range(0.0, p_max; length=n_steps+1)[2:end])
    u_prev    = copy(u)
    converged = false
    n_iter    = 0
    for iter in 1:max_iter
        fill!(F_ext, 0.0)
        assemble_global!(K_mem, r_int, dh, scv, u, mat)
        assemble_pressure_global!(F_ext, dh, scv, u, p)
        assemble_pressure_tangent_global!(K_pres, dh, scv, u, p)

        K_eff = K_mem - K_pres
        @. rhs = F_ext - r_int

        apply_zero!(K_eff, rhs, ch)

        n_iter = iter
        rhs_norm = norm(rhs)
        if rhs_norm < tol
            converged = true
            n_iter    = iter - 1
            break
        end

        du = K_eff \ rhs

        # Armijo backtracking: find α such that ||rhs(u+α·du)|| < ||rhs(u)||
        α = 1.0
        for _ in 1:8
            u_trial = u .+ α .* du
            assemble_residual_only!(r_trial, F_trial, dh, scv, u_trial, mat, p)
            @. rhs = F_trial - r_trial
            # zero constrained DOFs (apply_zero! on rhs only via prescribed DOFs)
            for dof in ch.prescribed_dofs
                rhs[dof] = 0.0
            end
            norm(rhs) < rhs_norm && break
            α /= 2
        end
        u .+= α .* du
    end
    VTKGridFile("pillow_newton-$t", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
    if !converged
        u .= u_prev
        @warn "p=$p did not converge in $max_iter iterations; rolling back"
    end
    w = evaluate_at_points(ph, dh, u, :u)[1][3]
    @printf("  %8.2f | %10.4e | %d\n", p, w, n_iter)
end
vtk_save(pvd);
