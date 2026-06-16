using FerriteShells,LinearAlgebra,Printf,WriteVTK

# helper for the mesh
function annular_plate_grid(n_θ, n_r;a=6.0,b=10.0,slit_gap=1e-3)
    g = shell_grid(
        generate_grid(QuadraticQuadrilateral, (n_θ, n_r),
                      Vec{2}((0.0, a)), Vec{2}((2π - slit_gap, b)));
        map = nd -> (nd.x[2]*cos(nd.x[1]), nd.x[2]*sin(nd.x[1]), 0.0))
    addfacetset!(g, "clamped", x -> abs(x[2]) < 1e-8 && x[1] > 0.5a)
    addfacetset!(g, "free",    x -> x[2] < -1e-4 && x[2] > -0.05 && x[1] > 0.5a)
    addnodeset!(g, "tip_outer", x -> abs(norm(x[1:2]) - b) < 0.15 && x[2] < -1e-4 && x[2] > -0.05)
    return g
end

function assemble_global!(K, r, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh); ke = zeros(n_e, n_e); re = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        membrane_tangent_RM!(ke, scv, u_e, mat)
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

# mesh dimensions and material
a = 6.0
b = 10.0
slit_gap = 1e-3
t = 0.03
mat = LinearElastic(2.1e7, 0.0, t)

# grid and interpolation space
grid  = annular_plate_grid(20, 8; a, b, slit_gap)
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degrees of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# Ndofs and evaluation points
N_dofs = ndofs(dh)
point_A = Vec{3}((b*cos(2π - slit_gap), b*sin(2π - slit_gap), 0.0))  # outer free edge
point_B = Vec{3}((a*cos(2π - slit_gap), a*sin(2π - slit_gap), 0.0))  # inner free edge
ph = PointEvalHandler(grid, [point_A, point_B])

# Reference (full) edge traction; the actual load is λ·F_ext with λ the unknown load factor.
F_ext = zeros(N_dofs)
assemble_traction!(F_ext, dh, getfacetset(grid, "free"), ip, fqr, Vec{3}((0.0,0.0,0.8)))

# Global u_z DOF at the outer corner of the loaded edge (point A) — the control DOF.
tip_node = only(getnodeset(grid, "tip_outer"))
w_dof = let dof = 0
    for cell in CellIterator(dh)
        for (I, gid) in enumerate(getnodes(cell))
            gid == tip_node && (dof = celldofs(cell)[3I]; break)
        end
        dof > 0 && break
    end
    dof
end
@assert w_dof > 0 "control u_z DOF (point A) not found"

# Displacement-controlled path following (bordering method).
#   Equilibrium: R(u) − λ·F_ext = 0,   constraint: u[w_dof] = w_target
#   v₁ = K⁻¹(λ·F_ext − R_int),  v₂ = K⁻¹·F_ext
#   δλ = (w_target − u[w_dof] − v₁[w_dof]) / v₂[w_dof],   δu = v₁ + δλ·v₂
# Load control fails here: from the flat reference the thin plate (t/(b−a)=0.0075) has a
# near-singular bending-dominated tangent, so a load increment overshoots equilibrium and
# the residual climbs instead of dropping. Displacement control keeps the linearisation on
# the equilibrium path. λ is traced from 0 up to the full load (λ=1).
w_max   = 20.0          # upper bound on u_z(A); λ=1 is reached well before this
n_steps = 200
Δw      = w_max / n_steps
tol     = 1e-6
max_iter = 20

K  = allocate_matrix(dh)
r_int = zeros(N_dofs)
v1 = zeros(N_dofs)
v2 = zeros(N_dofs)

pvd = paraview_collection("slit_annular_plate")
println("Slit annular plate (displacement control, n=$(getncells(grid)) cells)")
println("  step |   λ    |  u_z(A,outer) | u_z(B,inner) | iters")

VTKGridFile("slit_annular_plate-0", dh) do vtk
    write_solution(vtk, dh, zeros(N_dofs)); pvd[0.0] = vtk
end

let u = zeros(N_dofs), λ = 0.0
    # Symbolic LU from the linearised system at u=0; lu! reuses it numerically each step.
    assemble_global!(K, r_int, dh, scv, u, mat)
    apply_zero!(K, r_int, ch)
    F_lu = lu(K)

    for step in 1:n_steps
        w_target = step * Δw
        converged = false; n_iter = 0
        for iter in 1:max_iter
            assemble_global!(K, r_int, dh, scv, u, mat)
            rhs1 = λ .* F_ext .- r_int          # −R(u,λ)
            apply_zero!(K, rhs1, ch)
            if norm(rhs1) < tol && abs(u[w_dof] - w_target) < tol
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter
            lu!(F_lu, K)
            ldiv!(v1, F_lu, rhs1)               # equilibrium correction
            ldiv!(v2, F_lu, F_ext)              # load-direction vector
            δλ = (w_target - u[w_dof] - v1[w_dof]) / v2[w_dof]
            u .+= v1 .+ δλ .* v2
            λ += δλ
            apply!(u, ch)
        end
        !converged && (@warn "step $step (w_target=$(round(w_target;digits=3))) did not converge (λ=$λ)"; break)

        u_pts = evaluate_at_points(ph, dh, u, :u)
        @printf("  %4d | %.4f | %13.4f | %12.4f | %d\n", step, λ, u_pts[1][3], u_pts[2][3], n_iter)
        VTKGridFile("slit_annular_plate-$step", dh) do vtk
            write_solution(vtk, dh, u); pvd[float(step)] = vtk
        end
        λ ≥ 1.0 && (@printf("  Reached full load λ=1 at step %d (u_z(A)=%.3f).\n", step, u[w_dof]); break)
    end
end
vtk_save(pvd)
