using FerriteShells
using ForwardDiff
using LinearAlgebra
using Printf
using WriteVTK
using TimerOutputs
# Square airbag: flat square plate [0,L]×[0,L], simply-supported at edges (u_z=0, φ=0),
# inflated by follower pressure. Quarter-domain symmetry model [0,L/2]×[0,L/2].
# Reissner–Mindlin (5 DOF/node) with geometrically exact Rodrigues directors.
#
# Displacement-controlled path following (bordering method):
#   Prescribe w_center = step·Δw; treat pressure p as an additional unknown.
#   Equilibrium: R(u,p) = R_int(u) − p·F_p(u) = 0   (F_p: unit-pressure follower force)
#   Constraint:  u[w_c] = w_target
#   Bordering Newton: v₁ = K_eff⁻¹(p·F_p − R_int),  v₂ = K_eff⁻¹·F_p
#     δp = (w_target − u[w_c] − v₁[w_c]) / v₂[w_c],   δu = v₁ + δp·v₂
#   K_eff = K_int − p·K_pres,   K_pres = ∂F_p/∂u  (follower pressure load-stiffness).
#
# Load-controlled Newton is infeasible for t/L = 10⁻³: from a flat reference state,
# K_eff is bending-dominated (λ_min ≈ 10⁻³) and the Newton step is O(10³ m), far
# beyond the radius of convergence of any line search. Displacement control avoids
# this by keeping the linearisation always at a membrane-dominated inflated state.

function make_quarter_pillow_grid(n; L=1.0)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L/2, 0.0)), Vec{2}((L/2, L/2)), Vec{2}((0.0, L/2))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n, n), corners))
    addfacetset!(grid, "edge",  x -> isapprox(x[1], L/2, atol=1e-10) || isapprox(x[2], L/2, atol=1e-10))
    addfacetset!(grid, "sym_x", x -> isapprox(x[1], 0.0, atol=1e-10))
    addfacetset!(grid, "sym_y", x -> isapprox(x[2], 0.0, atol=1e-10))
    addnodeset!(grid, "center", x -> norm(x) < 1e-10)
    return grid
end

# Assemble K_int, R_int, K_pres and F_p (all for unit pressure p=1) in one cell loop.
function assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    ke_p = zeros(n_e, n_e); re_p = zeros(n_e)
    asm_i = start_assemble(K_int, r_int)
    asm_p = start_assemble(K_pres)
    fill!(F_p, 0.0)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        fill!(ke_p, 0.0); fill!(re_p, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        @timeit "membrane tangent" membrane_tangent_RM!(ke_i, scv, u_e, mat)
        @timeit "membrane residual" membrane_residuals_RM!(re_i, scv, u_e, mat)
        @timeit "bending tangent" bending_tangent_RM!(ke_i, scv, u_e, mat)
        @timeit "bending residual" bending_residuals_RM!(re_i, scv, u_e, mat)
        @timeit "pressure tangent" assemble_pressure!(re_p, scv, u_e, 1.0)
        @timeit "pressure residual" assemble_pressure_tangent!(ke_p, scv, u_e, 1.0)
        assemble!(asm_i, sd, ke_i, re_i)
        assemble!(asm_p, sd, ke_p)
        F_p[sd] .+= re_p
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
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "edge"),  x -> 0.0, [3]))          # u_z=0 at boundary
add!(ch, Dirichlet(:θ, getfacetset(grid, "edge"),  x -> zeros(2), [1,2]))   # φ=0 at boundary
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # u_x=0 at x=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_x"), x -> 0.0, [1]))          # φ₁=0 at x=0 (∂w/∂x=0)
add!(ch, Dirichlet(:u, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # u_y=0 at y=0
add!(ch, Dirichlet(:θ, getfacetset(grid, "sym_y"), x -> 0.0, [2]))          # φ₂=0 at y=0 (∂w/∂y=0)
close!(ch); Ferrite.update!(ch, 0.0)

# Global DOF index for u_z at the center node (free DOF — used as displacement control).
center_node = only(getnodeset(grid, "center"))
w_center_dof = let
    dof = 0
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            if gid == center_node
                dof = celldofs(cell)[3I]; break
            end
        end
        dof > 0 && break
    end
    dof
end
@assert w_center_dof > 0 "center u_z DOF not found"

# Displacement steps: trace p vs w_center from w=0 up to p=p_max.
# Membrane theory: w ~ (p·L⁴/(E·t))^(1/3) → at p=500: w ≈ 0.63 m.
p_max   = 500.0
w_max   = 0.8         # upper bound for w_center (membrane theory at p_max ≈ 0.63 m)
n_steps = 160
Δw      = w_max / n_steps   # = 0.005 m = 5·t per step
tol     = 1e-6
max_iter = 20

N = ndofs(dh)

K_int  = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
K_eff  = allocate_matrix(dh)   # preallocated; values updated in-place each Newton step
r_int  = zeros(N)
F_p    = zeros(N)
v1     = zeros(N)
v2     = zeros(N)

println("Square airbag RM (Q9, n=$n, p_max=$p_max, Δw=$(round(Δw;digits=4)) m)")
println("  step |    p    | w_center | iters")

pvd = paraview_collection("square_airbag")

# Use a let block so that p and u are unambiguously local (avoids Julia soft-scope warning
# when assigning to variables that also exist at global scope).
reset_timer!()
let u = zeros(N), p = 0.0
    VTKGridFile("square_airbag-0", dh) do vtk
        write_solution(vtk, dh, u); pvd[0.0] = vtk
    end

    # Initialise symbolic LU factorisation from the linearised system at u=0, p=0.
    # lu!(F_lu, K) reuses this symbolic analysis (sparsity pattern) and only redoes
    # the numeric phase, which is the dominant cost for Newton on a fixed mesh.
    @timeit "assembly" assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
    K_eff.nzval .= K_int.nzval
    let rhs_dummy = zeros(N); apply_zero!(K_eff, rhs_dummy, ch); end
    F_lu = lu(K_eff)

    for step in 1:n_steps
        w_target = step * Δw

        converged = false; n_iter = 0
        for iter in 1:max_iter
            @timeit "assembly" assemble_all!(K_int, r_int, K_pres, F_p, dh, scv, u, mat)
            # Update K_eff values in-place (same sparsity pattern as K_int and K_pres).
            K_eff.nzval .= K_int.nzval .- p .* K_pres.nzval
            rhs1 = p .* F_p .- r_int           # −R(u,p): negative equilibrium residual
            apply_zero!(K_eff, rhs1, ch)        # zero BC rows/cols of K_eff, BC entries of rhs1
            if norm(rhs1) < tol && abs(u[w_center_dof] - w_target) < tol
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter
            @timeit "linear solve" begin
                @timeit "lu " lu!(F_lu, K_eff)        # numeric refactorisation only; reuses symbolic analysis
                @timeit "ldiv!-1" ldiv!(v1, F_lu, rhs1)   # equilibrium correction
                @timeit "ldiv!-2" ldiv!(v2, F_lu, F_p)    # load-direction vector
            end
            δp = (w_target - u[w_center_dof] - v1[w_center_dof]) / v2[w_center_dof]
            u .+= v1 .+ δp .* v2
            p  += δp
            apply!(u, ch)                       # reset BC DOFs to zero
        end

        if !converged
            @warn "step $step (w_target=$w_target) did not converge after $max_iter iters (p=$p)"
            break
        end

        VTKGridFile("square_airbag-$step", dh) do vtk
            write_solution(vtk, dh, u); pvd[float(step)] = vtk
        end
        @printf("  %4d | %7.2f | %8.4e | %d\n", step, p, u[w_center_dof], n_iter)
        p ≥ p_max && (@printf("  Reached p_max = %.1f at step %d (w = %.4f m).\n", p_max, step, u[w_center_dof]); break)
    end
end
vtk_save(pvd);
print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)