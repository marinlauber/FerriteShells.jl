using FerriteShells
using LinearAlgebra
using Printf
using WriteVTK

# Square pillow clamped on all four edges, inflated by follower pressure.
# Quarter-domain symmetry model.
#
# Crisfield spherical arc-length (Riks) method — no external ODE library.
# State: (u, λ) ∈ ℝᴺ × ℝ.  Equilibrium: R_int(u) = λ·F(u).
#
# For this thin shell (t/L = 0.001) the equilibrium path has a limit point
# (snap-through) at very low pressure (~0.15 Pa).  A Newton bootstrap with
# Armijo line search first steps through the snap-through region to land on
# the stable inflated branch at λ = λ_start; the arc-length then continues
# from there to p_max = 500.
#
# Arc-length corrector decomposition:
#   K_eff·δu₁ = −R              (residual correction)
#   K_eff·δu₂ =  F              (load-direction correction)
#   δu = δu₁ + δλ·δu₂,  δλ from the spherical constraint ‖Δu‖² + Δλ² = Δs².

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

n   = 32
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
add!(ch, Dirichlet(:u, getnodeset(grid, "edge"),  x -> 0.0, [3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_x"), x -> 0.0, [1]))
add!(ch, Dirichlet(:u, getnodeset(grid, "sym_y"), x -> 0.0, [2]))
close!(ch)
Ferrite.update!(ch, 0.0)

u0 = zeros(ndofs(dh))
for cell in CellIterator(dh)
    x    = getcoordinates(cell)
    dofs = celldofs(cell)
    for i in eachindex(x)
        u0[dofs[3i]] = 1e-2 * sin(π * x[i][1] / (L/2)) * sin(π * x[i][2] / (L/2))
    end
end
apply!(u0, ch)

N     = ndofs(dh)
p_max = 500.0

K_mem  = allocate_matrix(dh)
K_pres = allocate_matrix(dh)
r_int  = zeros(N)
F_ext  = zeros(N)
F_dir  = zeros(N)

# Bootstrap: Newton load-stepping with Armijo line search from p=0 to λ_start.
# Steps in the snap-through region (p ≲ 140) roll back harmlessly; steps on
# the stable inflated branch (p ≳ 150) converge and build up the solution.
λ_start  = 200.0
u_start  = copy(u0)
rhs_boot = zeros(N)
r_tmp    = zeros(N)
F_tmp    = zeros(N)
println("Bootstrap Newton (p = 10 … $λ_start):")
let u = u_start
    for pb in 10.0:10.0:λ_start
        println("pb = $pb")
        u_prev = copy(u)
        converged = false
        for iter in 1:30
            print("    iter = $iter,")
            fill!(F_ext, 0.0)
            assemble_global!(K_mem, r_int, dh, scv, u, mat)
            assemble_pressure_global!(F_ext, dh, scv, u, pb)
            assemble_pressure_tangent_global!(K_pres, dh, scv, u, pb)
            K_eff = K_mem - K_pres
            @. rhs_boot = F_ext - r_int
            apply_zero!(K_eff, rhs_boot, ch)
            rhs_norm = norm(rhs_boot)
            print(" norm(rhs) = $rhs_norm")
            if rhs_norm < 1e-8
                converged = true
                break
            end
            du = K_eff \ rhs_boot
            α  = 1.0
            for _ in 1:8
                assemble_residual_only!(r_tmp, F_tmp, dh, scv, u .+ α .* du, mat, pb)
                @. r_tmp = F_tmp - r_tmp
                for dof in ch.prescribed_dofs; r_tmp[dof] = 0.0; end
                norm(r_tmp) < rhs_norm && break
                α /= 2
            end
            println(", α = $α")
            u .+= α .* du
        end
        if converged
            w = u[3]   # approximate center deflection (first z-dof)
            @printf("  p=%6.1f  converged  w≈%.4e\n", pb, w)
        else
            u .= u_prev   # roll back
        end
    end
end

# Arc-length parameters
Δs_init   = 5.0
Δs_min    = 1e-8
Δs_max    = 500.0
n_desired = 5
max_iter  = 20
tol       = 1e-8
max_steps = 2000

ph  = PointEvalHandler(grid, [Vec{3}((0.0, 0.0, 0.0))])
pvd = paraview_collection("pillow_arclength")

println("Arc-length (Crisfield spherical) method — starting from λ = $λ_start")
println("  pressure | deflection | iters |    Δs")

let u = u_start, λ = λ_start, tangent_n = nothing, n_step = 0, t_vtk = 0, Δs = Δs_init
while λ < p_max && n_step < max_steps
    n_step += 1

    # ============ PREDICTOR ============
    assemble_global!(K_mem, r_int, dh, scv, u, mat)
    assemble_pressure_tangent_global!(K_pres, dh, scv, u, λ)
    fill!(F_ext, 0.0)
    assemble_pressure_global!(F_ext, dh, scv, u, 1.0)

    K_eff = K_mem - K_pres
    apply_zero!(K_eff, F_ext, ch)

    v = K_eff \ F_ext  # sensitivity du/dλ at current state

    dλds_abs     = 1.0 / sqrt(dot(v, v) + 1.0)
    full_tangent = [v .* dλds_abs; dλds_abs]

    sign_f = tangent_n === nothing ? 1.0 : (dot(full_tangent, tangent_n) >= 0.0 ? 1.0 : -1.0)

    dλds    = sign_f * dλds_abs
    dλ_pred = dλds * Δs
    u_c     = u .+ v .* (dλds * Δs)
    λ_c     = λ + dλ_pred

    # ============ CORRECTOR ============
    converged = false
    n_iter    = 0
    for iter in 1:max_iter
        fill!(F_ext, 0.0)
        assemble_global!(K_mem, r_int, dh, scv, u_c, mat)
        assemble_pressure_global!(F_ext, dh, scv, u_c, 1.0)
        assemble_pressure_tangent_global!(K_pres, dh, scv, u_c, λ_c)

        K_eff_c = K_mem - K_pres
        r_c     = r_int .- λ_c .* F_ext
        apply_zero!(K_eff_c, r_c, ch)

        n_iter = iter
        if norm(r_c) < tol
            converged = true
            n_iter    = iter - 1
            break
        end

        K_fact = factorize(K_eff_c)
        δu_1   = K_fact \ (-r_c)

        fill!(F_dir, 0.0)
        assemble_pressure_global!(F_dir, dh, scv, u_c, 1.0)
        for dof in ch.prescribed_dofs; F_dir[dof] = 0.0; end
        δu_2 = K_fact \ F_dir

        Δu = u_c .- u
        Δλ = λ_c - λ
        w  = Δu .+ δu_1

        a₁   = dot(δu_2, δu_2) + 1.0
        a₂   = 2.0 * (dot(w, δu_2) + Δλ)
        a₃   = dot(w, w) + Δλ^2 - Δs^2
        disc = a₂^2 - 4*a₁*a₃

        if disc < 0.0
            δλ = -a₂ / (2*a₁)
        else
            δλ_1 = (-a₂ + sqrt(disc)) / (2*a₁)
            δλ_2 = (-a₂ - sqrt(disc)) / (2*a₁)
            δλ = abs(Δλ + δλ_1 - dλ_pred) ≤ abs(Δλ + δλ_2 - dλ_pred) ? δλ_1 : δλ_2
        end

        u_c .+= δu_1 .+ δλ .* δu_2
        λ_c  += δλ
    end

    if !converged
        Δs = max(Δs / 2, Δs_min)
        Δs > Δs_min || (@warn "minimum Δs reached, stopping"; break)
        n_step -= 1
        continue
    end

    # Accept step
    u         = copy(u_c)
    λ         = λ_c
    tangent_n = sign_f .* full_tangent
    Δs        = clamp(Δs * sqrt(n_desired / max(n_iter, 1)), Δs_min, Δs_max)

    t_vtk += 1
    VTKGridFile("pillow_arclength-$t_vtk", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t_vtk] = vtk
    end
    w = evaluate_at_points(ph, dh, u, :u)[1][3]
    @printf("  %8.2f | %10.4e | %d | %.4e\n", λ, w, n_iter, Δs)
end
end  # let
vtk_save(pvd)
