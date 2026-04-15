using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Prestressed shear wrinkling of a square membrane (380×380 mm, t=0.025 mm).
#
# Two-step procedure:
#   Phase 1 (t ∈ [0,1]): uniform vertical prestress at top edge, u_y = 0.05 mm
#   Phase 2 (t ∈ [1,2]): horizontal shear at top edge, u_x = 3·(t−1)·x/380 mm
#                          (0 at x=0, up to 3 mm at x=380 at full load)
#
# Initial geometric imperfection:  w = 0.01·cos(2x/20)·cos(2y/20)  [mm]
# This seeds the wrinkling mode without requiring eigenvector perturbations.

function make_grid(nx, ny; primitive=Quadrilateral)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((380.0, 0.0)),
               Vec{2}((380.0, 128.0)), Vec{2}((0.0, 128.0))]
    grid2D = generate_grid(primitive, (nx, ny), corners)
    grid = shell_grid(grid2D; map = n -> (n.x[1], n.x[2], 0.01 * cos(2*n.x[1]/20) * cos(2*n.x[2]/20)))
    addfacetset!(grid, "top", x -> isapprox(x[2], 128.0, atol=1e-10))
    addfacetset!(grid, "bottom", x -> isapprox(x[2], 0.0,   atol=1e-10))
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

mat  = LinearElastic(3500.0, 0.31, 0.025)
grid = make_grid(150, 50; primitive=Quadrilateral)

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# Phase 1 (t ≤ 1): uniform vertical stretch u_y = 0.05·t, no horizontal motion.
# Phase 2 (t > 1): maintain u_y = 0.05, add horizontal shear u_x = 3·(t−1)·x/380.
function top_disp(x, t)
    t ≤ 1.0 ? Vec{3}((0.0, 0.05*t, 0.0)) : Vec{3}((3.0*(t-1.0)*x[1]/380.0, 0.05, 0.0))
end

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), x -> zero(x), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "bottom"), x -> zeros(2), [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x,t) -> top_disp(x, t), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "top"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

const N_dof = ndofs(dh)
const free  = ch.free_dofs

K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N_dof)
Δu    = zeros(N_dof)
u     = zeros(N_dof); apply!(u, ch)

# Initial symbolic LU factorisation
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
let tmp = zeros(N_dof); apply_zero!(K_eff, tmp, ch); end
F_lu = lu(K_eff)

pvd = paraview_collection("membrane_shear")
vtk_step = Ref(0)
VTKGridFile("membrane_shear-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

const tol      = 1e-6
const max_iter = 20

@printf("%-6s  %-10s  %-8s  %-6s\n", "step", "phase", "t", "iters")

for (label, t_vals) in [("prestress", range(0.0, 1.0, 11)[2:end]),
                         ("shear",    range(1.0, 2.0, 61)[2:end])]
    for t in t_vals
        Ferrite.update!(ch, t)
        apply!(u, ch)

        converged = false; iters = 0
        for iter in 1:max_iter
            iters = iter
            assemble_all!(K_int, r_int, dh, scv, u, mat)
            K_eff.nzval .= K_int.nzval
            rhs = .-r_int
            apply_zero!(K_eff, rhs, ch)
            norm(@views rhs[free]) < tol && (converged = true; break)
            lu!(F_lu, K_eff)
            ldiv!(Δu, F_lu, rhs)
            u .+= Δu
            apply!(u, ch)
        end

        !converged && @warn "t=$(round(t, digits=3)): no convergence in $max_iter iters"

        vtk_step[] += 1
        VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u); pvd[t] = vtk
        end

        @printf("%-6d  %-10s  %-8.3f  %-6d\n", vtk_step[], label, t, iters)
    end
end
close(pvd)
