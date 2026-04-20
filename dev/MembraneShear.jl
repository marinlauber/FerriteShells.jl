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
grid = make_grid(150, 50; primitive=QuadraticQuadrilateral)

ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC9)

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

N_dof = ndofs(dh)
free  = ch.free_dofs
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N_dof)
Δu    = zeros(N_dof)
u     = zeros(N_dof); apply!(u, ch)

# Initial symbolic LU factorisation
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# Positions of diagonal entries in K_eff.nzval (used by viscous stabilization)
diag_idx = let
    idx = Int[]
    for j in 1:N_dof
        for k in K_eff.colptr[j]:K_eff.colptr[j+1]-1
            K_eff.rowval[k] == j && (push!(idx, k); break)
        end
    end
    idx
end

pvd = paraview_collection("membrane_shear")
vtk_step = Ref(0)
VTKGridFile("membrane_shear-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

tol      = 1e-6
max_iter = 25

@printf("%-6s  %-10s  %-8s  %-6s  %-10s\n", "step", "phase", "t", "iters", "c_visc")

# Newton iteration with optional diagonal viscous stabilization.
# When c_visc > 0, adds c_visc*(u − u_ref) to the residual and c_visc to K diagonal.
# This makes K positive-definite through bifurcations at the cost of a small residual
# imbalance c_visc*||u − u_ref|| that decays as u_ref tracks the solution.
function newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, tol, max_iter, diag_idx;
                 c_visc=0.0, u_ref=u)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        K_eff.nzval .= K_int.nzval
        rhs = .-r_int
        if c_visc > 0.0
            rhs .-= c_visc .* (u .- u_ref)
            @views K_eff.nzval[diag_idx] .+= c_visc
        end
        apply_zero!(K_eff, rhs, ch)
        norm(@views rhs[free]) < tol && (converged = true; break)
        lu!(F_lu, K_eff); ldiv!(Δu, F_lu, rhs)
        u .+= Δu; apply!(u, ch)
    end
    converged, iters
end

# Phase 1: prestress (fixed steps, always converges without stabilization)
for t in range(0.0, 1.0, 11)[2:end]
    Ferrite.update!(ch, t); apply!(u, ch)
    converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, tol, max_iter, diag_idx)
    !converged && error("Prestress failed at t=$t")
    vtk_step[] += 1
    VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t] = vtk
    end
    @printf("%-6d  %-10s  %-8.3f  %-6d  %-10s\n", vtk_step[], "prestress", t, iters, "-")
end

# Phase 2: shear with adaptive stepping and viscous stabilization.
# Near wrinkling bifurcations K becomes indefinite; plain Newton diverges.
# Stabilization levels: c = 0 (exact) → c_base → 10c_base → 100c_base (progressively more damped).
# If all stabilization levels fail, cut back Δt by half and retry.
# c_base ~ 1e-4 × max(diag K) so the parasitic residual stays small relative to tol at moderate Δu.
assemble_all!(K_int, r_int, dh, scv, u, mat)
c_base  = 1e-4 * maximum(@views K_int.nzval[diag_idx])
c_levels = (0.0, c_base, 10*c_base, 100*c_base)

let
u_conv  = copy(u)
t       = 1.0
Δt      = 0.002
Δt_min  = 1e-5
Δt_max  = 0.01
t_end   = 2.0

while t < t_end - Δt_min/2
    Δt    = min(Δt, t_end - t)
    t_try = t + Δt
    Ferrite.update!(ch, t_try)

    converged = false; iters = 0; c_used = 0.0
    for c_visc in c_levels
        u .= u_conv; apply!(u, ch)
        converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, tol, max_iter, diag_idx;
                                   c_visc, u_ref=u_conv)
        converged && (c_used = c_visc; break)
    end

    if converged
        u_conv .= u; t = t_try
        Δt = min(Δt * (iters ≤ 4 ? 1.5 : 1.0), Δt_max)
        vtk_step[] += 1
        VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
            write_solution(vtk, dh, u); pvd[t] = vtk
        end
        c_str = c_used > 0 ? @sprintf("%.2e", c_used) : "-"
        @printf("%-6d  %-10s  %-8.4f  %-6d  %-10s\n", vtk_step[], "shear", t, iters, c_str)
    else
        Δt *= 0.5
        if Δt < Δt_min
            @warn "Step too small (Δt=$Δt) at t=$t — aborting"
            break
        end
        @printf("  cut-back at t=%.4f → Δt=%.6f\n", t, Δt)
    end
end
close(pvd)
end
