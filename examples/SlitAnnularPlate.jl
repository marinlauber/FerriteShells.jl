using FerriteShells, LinearAlgebra, Printf, WriteVTK

# Slit annular plate roll-up — Reissner–Mindlin (5 DOF/node).
# Flat annular plate with inner radius a=6, outer radius b=10, slit at θ=0.
# Clamped at θ=0; dead-load moment M (per unit radial length) at θ≈2π.
# Parameters: t=0.04, E=2.1×10⁷, ν=0.  Bending stiffness D = Et³/12 ≈ 112.
# Full-circle moment: M_full = D/R_mean ≈ 14 (plate forms a torus at mean radius).
#
# Director: T₁=ê_θ, T₂=ê_r, G₃=−ê_z (flat plate, downward normal).
# Rodrigues limit |φ| < π → director at free end limited to ~180° rotation.
#
# LOCKING ANALYSIS (Section 1):
#   The plate starts flat → initial K is purely bending → same membrane locking as
#   pinched hemisphere (t/R_mean ≈ 0.005). Linear convergence study quantifies this.
#
# NEWTON CONVERGENCE PARADOX (Section 2):
#   Locked meshes (coarse) converge easily because the artificially large K gives a
#   small Newton step, staying in the quadratic regime of the Armijo check.
#   Unlocked meshes (fine) have the physically correct, softer K; their first Newton
#   step from the flat reference is geometrically huge and needs many more line-search
#   halvings to satisfy Armijo. With 50 halvings the fine mesh also converges.

const a        = 6.0
const b        = 10.0
const R_mean   = 0.5*(a + b)
const t        = 0.03
const mat      = LinearElastic(21.0e6, 0.0, t)
const slit_gap = 1e-3   # radians; free face at θ=2π−slit_gap → y ≈ −r·slit_gap

function annular_plate_grid(n_θ, n_r)
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
        membrane_tangent_RM_FD!(ke, scv, u_e, mat)
        membrane_residuals_RM_FD!(re, scv, u_e, mat)
        bending_tangent_RM_FD!(ke, scv, u_e, mat)
        bending_residuals_RM_FD!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

function strain_energy(dh, scv, u, mat)
    E = 0.0
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        E += FerriteShells.membrane_energy_RM(u_e, scv, mat)
        E += FerriteShells.bending_shear_energy_RM(u_e, scv, mat)
    end
    return E
end

potential(dh, scv, u, mat, F) = strain_energy(dh, scv, u, mat) - dot(F, u)

println("Slit annular plate RM (Q9): a=$a, b=$b, t=$t")

n_steps = 1000
n_halvings = 50
tol      = 1e-6
max_iter = 20
armijo_c = 1e-4

grid  = annular_plate_grid(10, 4)
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
fqr = FacetQuadratureRule{RefQuadrilateral}(3)
scv   = ShellCellValues(qr, ip, ip; mitc=MITC9)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dofs = ndofs(dh)
point_A = Vec{3}((b*cos(2π - slit_gap), b*sin(2π - slit_gap), 0.0))  # outer free edge
point_B = Vec{3}((a*cos(2π - slit_gap), a*sin(2π - slit_gap), 0.0))  # inner free edge
ph = PointEvalHandler(grid, [point_A, point_B])
K = allocate_matrix(dh); r_vec = zeros(N_dofs); rhs = zeros(N_dofs)
F_ext = zeros(N_dofs)

assemble_traction!(F_ext, dh, getfacetset(grid, "free"), ip, fqr, Vec{3}((0.0,0.0,0.8)))

pvd = paraview_collection("slit_annular_plate")
println("  step |    F    |  u_z(A,outer) | u_z(B,inner) | iters")
u = zeros(N_dofs)
for step in 1:n_steps
    λ = step / n_steps
    F = λ .* F_ext
    u_prev = copy(u)
    converged = false; n_iter = 0
    for iter in 1:max_iter
        assemble_global!(K, r_vec, dh, scv, u, mat)
        @. rhs = F - r_vec; apply_zero!(K, rhs, ch)
        norm(rhs) < tol && (converged = true; n_iter = iter-1; break)
        n_iter = iter
        du = K \ rhs;
        slope = dot(rhs, du)
        Π0 = potential(dh, scv, u, mat, F)
        α_ls = 1.0
        for _ in 1:n_halvings
            u_trial = u .+ α_ls .* du
            Π_trial=  potential(dh, scv, u_trial, mat, F)
            Π_trial ≤ Π0 - armijo_c * α_ls * slope && break
            α_ls /= 2
        end
        u .+= α_ls .* du
    end
    !converged && (@warn "step $step (M=$(round(λ;digits=3))) did not converge; rolling back"; break)
    if step % max(1, n_steps ÷ 10) == 0
        u_pts = evaluate_at_points(ph, dh, u, :u)
        @printf("  %4d | %7.4f | %13.4f | %12.4f | %d\n", step, λ, u_pts[1][3], u_pts[2][3], n_iter)
    end
    # save results
    VTKGridFile("slit_annular_plate-$(step)", dh) do vtk
        write_solution(vtk, dh, u); pvd[step] = vtk
    end
end
close(pvd)
