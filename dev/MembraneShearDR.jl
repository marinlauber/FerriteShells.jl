using FerriteShells,LinearAlgebra,Printf,WriteVTK

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

# material model and mesh
mat  = LinearElastic(3500.0, 0.31, 0.025)
grid = make_grid(150, 50; primitive=Quadrilateral)

# interpolation space
ip  = Lagrange{RefQuadrilateral, 1}()
qr  = QuadratureRule{RefQuadrilateral}(3)
scv = ShellCellValues(qr, ip, ip; mitc=MITC4)

# degrees of freedom, displacement and rotations
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# initially stretch vertical and then shear
top_disp(x, t) = t ≤ 1.0 ? Vec{3}((0.0, 0.05*t, 0.0)) : Vec{3}((3.0*(t-1.0), 0.05, 0.0))

# apply the BCs
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), x -> zero(x),   [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "bottom"), x -> zeros(2),  [1,2]))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x,t) -> top_disp(x,t), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "top"), x -> zeros(2),     [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# storage
N_dof = ndofs(dh)
free  = ch.free_dofs
K_int = allocate_matrix(dh)
K_eff = allocate_matrix(dh)
r_int = zeros(N_dof)
Δu    = zeros(N_dof)
u     = zeros(N_dof); apply!(u, ch)

# prepare the lu factorisation
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_eff.nzval .= K_int.nzval
apply_zero!(K_eff, r_int, ch)
F_lu = lu(K_eff)

# Positions of diagonal entries in nzval — used for diagonal regularization.
diag_idx = let
    idx = Int[]
    for j in 1:N_dof
        for k in K_eff.colptr[j]:K_eff.colptr[j+1]-1
            K_eff.rowval[k] == j && (push!(idx, k); break)
        end
    end
    idx
end

# save the fields
pvd = paraview_collection("membrane_shear")
vtk_step = Ref(0)
VTKGridFile("membrane_shear-0", dh) do vtk
    write_solution(vtk, dh, u); pvd[0.0] = vtk
end

@printf("%-6s  %-12s  %-8s  %-8s  %-12s\n", "step", "phase", "t", "iters", "Δτ")

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

# Pseudo-Transient Continuation with SER Δτ update.
# Convergence check uses the unregularized residual ‖R[free]‖ so the tolerance is
# the same as for Newton. The regularization I/Δτ is added only for the solve.
function ptc!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, tol, max_iter, diag_idx; Δτ₀=1.0)
    Δτ = Δτ₀; res_old = Inf; converged = false; iters = 0; res = 0.0
    for iter in 1:max_iter
        iters = iter
        assemble_all!(K_int, r_int, dh, scv, u, mat)
        K_eff.nzval .= K_int.nzval
        rhs = .-r_int; apply_zero!(K_eff, rhs, ch)
        res = norm(@views rhs[free])
        res < tol && (converged = true; break)
        iter > 1 && (Δτ *= res_old / res)   # SER: grow Δτ when residual falls, shrink when it rises
        res_old = res
        @views K_eff.nzval[diag_idx] .+= 1/Δτ
        lu!(F_lu, K_eff); ldiv!(Δu, F_lu, rhs)
        u .+= Δu; apply!(u, ch)
    end
    converged, iters, Δτ, res
end

# Newton-Raphson prestress (t ∈ [0,1]) — smooth path, no bifurcation.
for t in range(0.0, 1.0, 11)[2:end]
    Ferrite.update!(ch, t); apply!(u, ch)
    converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, 1e-6, 20)
    !converged && error("Prestress Newton failed at t=$t")
    vtk_step[] += 1
    VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t] = vtk
    end
    @printf("%-6d  %-12s  %-8.3f  %-8d  %-12s\n", vtk_step[], "prestress(NR)", t, iters, "-")
end

# Compute Δτ₀ from K at t = 1: set 1/Δτ₀ = mean(diag K_free).
# This makes I/Δτ₀ ≈ K on average → the initial regularized system is always positive-definite.
assemble_all!(K_int, r_int, dh, scv, u, mat)
K_diag_free = @views K_int.nzval[diag_idx[free]]
Δτ₀_ptc = length(K_diag_free) / sum(K_diag_free)   # = 1 / mean(diag K_free)

# PTC shear (t ∈ [1,2]).
# SER drives Δτ → ∞ near equilibrium (recovering Newton) and back to Δτ₀ through snap-throughs.
tol_ptc      = 1e-6
max_iter_ptc = 200
for t in range(1.0, 2.0, 101)[2:end]
    Ferrite.update!(ch, t); apply!(u, ch)
    converged, iters, Δτ_f, res = ptc!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free,
                                        tol_ptc, max_iter_ptc, diag_idx; Δτ₀=Δτ₀_ptc)
    !converged && @warn "PTC not converged at t=$(round(t,digits=3)) after $iters iters (Δτ=$(round(Δτ_f,sigdigits=3)), res=$(round(res,sigdigits=3)))"
    vtk_step[] += 1
    VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
        write_solution(vtk, dh, u); pvd[t] = vtk
    end
    @printf("%-6d  %-12s  %-8.3f  %-8d  %-12.3e\n", vtk_step[], "shear(PTC)", t, iters, Δτ_f)
end

# Final Newton solve at t = 2: verify equilibrium and tighten to 1e-6.
# If PTC left us close, this converges in 1-3 iterations; otherwise it flags a problem.
converged, iters = newton!(u, K_int, K_eff, r_int, F_lu, Δu, dh, scv, mat, ch, free, 1e-6, 10)
if converged
    @printf("Final Newton polish: converged in %d iters\n", iters)
else
    @warn "Final Newton did not converge in 10 iters — PTC solution may not be at equilibrium"
end
vtk_step[] += 1
VTKGridFile("membrane_shear-$(vtk_step[])", dh) do vtk
    write_solution(vtk, dh, u); pvd[2.0] = vtk
end
close(pvd)
