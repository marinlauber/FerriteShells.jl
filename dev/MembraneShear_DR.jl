using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Prestressed shear wrinkling of a square membrane (380×380 mm, t=0.025 mm).
#
# Two-phase solver:
#   Phase 1 (t ∈ [0,1]): Newton-Raphson — uniform prestress, smooth path, quadratic convergence.
#   Phase 2 (t ∈ [1,2]): Dynamic Relaxation with kinetic-energy (KE) peak damping.
#     Near wrinkling bifurcations K becomes indefinite and Newton diverges. DR avoids the
#     tangent entirely: it integrates the artificial dynamics M·ü + R_int(u) = 0 with an
#     explicit scheme and resets velocities to zero whenever kinetic energy peaks. The system
#     dissipates its way to equilibrium regardless of snap-throughs or mode switches.
#   Final step: one Newton solve at t=2 to verify (and tighten) the DR solution.
#
# Artificial mass: M_I = K_II (stiffness-proportional lumped mass).
#   All modes get the same natural frequency ω = 1 rad/s → Δt_crit = 2.
#   Use Δt_dr = 0.9 for a comfortable stability margin.

function make_grid(nx, ny; primitive=Quadrilateral)
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((380.0, 0.0)),
               Vec{2}((380.0, 128.0)), Vec{2}((0.0, 128.0))]
    grid2D = generate_grid(primitive, (nx, ny), corners)
    grid = shell_grid(grid2D; map = n -> (n.x[1], n.x[2], 0.01 * cos(2*n.x[1]/20) * cos(2*n.x[2]/20)))
    addfacetset!(grid, "top",    x -> isapprox(x[2], 128.0, atol=1e-10))
    addfacetset!(grid, "bottom", x -> isapprox(x[2],   0.0, atol=1e-10))
end

function assemble_all!(K_int, r_int, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke_i = zeros(n_e, n_e); re_i = zeros(n_e)
    asm  = start_assemble(K_int, r_int)
    for cell in CellIterator(dh)
        fill!(ke_i, 0.0); fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        membrane_tangent_RM!(ke_i, scv, u_e, mat)
        bending_tangent_RM!(ke_i, scv, u_e, mat)
        assemble!(asm, sd, ke_i, re_i)
    end
end

# Residual-only assembly — skips tangent computation, used by DR inner loop.
function assemble_residual!(r_int, dh, scv, u, mat)
    fill!(r_int, 0.0)
    n_e  = ndofs_per_cell(dh)
    re_i = zeros(n_e)
    for cell in CellIterator(dh)
        fill!(re_i, 0.0)
        reinit!(scv, cell)
        sd  = shelldofs(cell)
        u_e = u[sd]
        membrane_residuals_RM!(re_i, scv, u_e, mat)
        bending_residuals_RM!(re_i, scv, u_e, mat)
        @views r_int[sd] .+= re_i
    end
end

mat  = LinearElastic(3500.0, 0.31, 0.025)
grid = make_grid(150, 50; primitive=Quadrilateral)

ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

function top_disp(x, t)
    t ≤ 1.0 ? Vec{3}((0.0, 0.05*t, 0.0)) : Vec{3}((3.0*(t-1.0), 0.05, 0.0))
end

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), x -> zero(x),   [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "bottom"), x -> zeros(2),  [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x,t) -> top_disp(x,t), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "top"), x -> zeros(2),     [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dof = ndofs(dh)
free  = ch.free_dofs
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N_dof)
Δu    = zeros(N_dof)
u     = zeros(N_dof); apply!(u, ch)

assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# Positions of diagonal entries in nzval — used for mass extraction.
diag_idx = let
    idx = Int[]
    for j in 1:N_dof
        for k in K_eff.colptr[j]:K_eff.colptr[j+1]-1
            K_eff.rowval[k] == j && (push!(idx, k); break)
        end
    end
    idx
end

pvd = paraview_collection("membrane_shear_dr")
vtk_step = Ref(0)
VTKGridFile("membrane_shear_dr-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

@printf("%-6s  %-12s  %-8s  %-8s  %-8s\n", "step", "phase", "t", "iters", "resets")

function newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, tol, max_iter)
    converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        K_eff.nzval .= K_int.nzval
        rhs = .-r_int; apply_zero!(K_eff, rhs, ch)
        norm(@views rhs[free]) < tol && (converged = true; break)
        lu!(F_lu, K_eff); ldiv!(Δu, F_lu, rhs)
        u .+= Δu; apply!(u, ch)
    end
    converged, iters
end

# Dynamic Relaxation with KE peak damping.
# Integrates M·ü + R_int = 0 explicitly and resets v = 0 whenever KE peaks.
# M_free: lumped mass for free DOFs. Δt_dr: explicit time step (< Δt_crit = 2√(M/K_max)).
function dynamic_relaxation!(u, r_int, dh, scv, mat, ch, free, M_free, Δt_dr, tol, max_iter)
    v      = zeros(length(free))
    F      = zeros(length(free))
    KE_old = Inf
    nresets = 0; converged = false; iters = 0
    for iter in 1:max_iter
        iters = iter
        assemble_residual!(r_int, dh, scv, u, mat)
        @views @. F = -r_int[free]
        norm(F) < tol && (converged = true; break)
        @. v += Δt_dr * F / M_free
        KE = 0.5 * sum(M_free .* v.^2)
        if KE < KE_old
            nresets += 1; fill!(v, 0.0)
        else
            @views u[free] .+= Δt_dr .* v
            apply!(u, ch)
        end
        KE_old = KE
    end
    converged, iters, nresets
end

# Phase 1: Newton-Raphson prestress (t ∈ [0,1]) — smooth path, no bifurcation.
for t in range(0.0, 1.0, 11)[2:end]
    Ferrite.update!(ch, t); apply!(u, ch)
    converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, 1e-6, 20)
    !converged && error("Prestress Newton failed at t=$t")
    vtk_step[] += 1
    VTKGridFile("membrane_shear_dr-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t] = vtk
    end
    @printf("%-6d  %-12s  %-8.3f  %-8d  %-8s\n", vtk_step[], "prestress(NR)", t, iters, "-")
end

# Compute stiffness-proportional lumped mass from K at t = 1 (end of prestress).
# M_I = K_II for free DOFs → ω_I = √(K_II/M_II) = 1 for all modes → Δt_crit = 2.
assemble_all!(K_int, r_int, dh, scv, u, mat)
M_free = max.((@views K_int.nzval[diag_idx][free]), 1e-10)
Δt_dr  = 0.9   # stable: Δt_dr = 0.9 < Δt_crit = 2.0

# Phase 2: Dynamic Relaxation shear (t ∈ [1,2]).
# KE peak resets dissipate snap-through energy and guide the solution through mode switches.
tol_dr      = 1e-3
max_iter_dr = 5000
for t in range(1.0, 2.0, 101)[2:end]
    Ferrite.update!(ch, t); apply!(u, ch)
    converged, iters, nresets = dynamic_relaxation!(u, r_int, dh, scv, mat, ch, free, M_free, Δt_dr, tol_dr, max_iter_dr)
    !converged && @warn "DR not converged at t=$(round(t,digits=3)) after $iters iters ($nresets resets)"
    vtk_step[] += 1
    VTKGridFile("membrane_shear_dr-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t] = vtk
    end
    @printf("%-6d  %-12s  %-8.3f  %-8d  %-8d\n", vtk_step[], "shear(DR)", t, iters, nresets)
end

# Final Newton solve at t = 2: verify equilibrium and tighten to 1e-6.
# If DR left us close, this converges in 1-3 iterations; otherwise it flags a problem.
converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, 1e-6, 10)
if converged
    @printf("Final Newton polish: converged in %d iters\n", iters)
else
    @warn "Final Newton did not converge in 10 iters — DR solution may not be at equilibrium"
end
vtk_step[] += 1
VTKGridFile("membrane_shear_dr-$(vtk_step[])", dh) do vtk
    write_solution(vtk, dh, u); pvd[2.0] = vtk
end
close(pvd)
