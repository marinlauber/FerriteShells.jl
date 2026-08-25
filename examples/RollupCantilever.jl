using FerriteShells,LinearAlgebra,Printf

# make the grid
function make_rollup_grid(;L=10.0, W=1.0, n_x=20, n_y=2)
    grid    = shell_grid(generate_grid(QuadraticQuadrilateral, (n_x, n_y), 
                                       Vec{2}((0.0, 0.0)), Vec{2}((L, W))))
    addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
    addfacetset!(grid, "free_end", x -> x[1] ≈ L)
    addnodeset!(grid,  "tip",      x -> x[1] ≈ L && x[2] ≈ W/2)
    return grid
end

# Apply dead-load moment M (about y-axis, bending in xz-plane) to RM shell.
# Virtual work: δW = ∫ m·δφ₁ dΓ,  f_{φ₁,I} = −m · ∫_edge N_I dΓ
# where m = M/W.  Sign convention: φ₁>0 tilts the director toward T₁=x̂, which
# corresponds to downward bending (u_z<0). A moment producing u_z>0 needs f_{φ₁}<0.
function apply_end_moment_RM!(f, dh, facetset, ip, fqr, m)
    n_base = getnbasefunctions(ip)
    n      = n_base
    fe     = zeros(ndofs_per_cell(dh))
    for fc in FacetIterator(dh, facetset)
        fill!(fe, 0.0)
        x        = getcoordinates(fc)
        facet_nr = fc.current_facet_id
        qr_f     = fqr.facet_rules[facet_nr]
        tdir     = facet_nr ∈ (1, 3) ? 1 : 2
        for (ξ, w_q) in zip(qr_f.points, qr_f.weights)
            Jt = zero(Vec{3,Float64})
            for I in 1:n_base
                dN, _ = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                Jt   += dN[tdir] * x[I]
            end
            dΓ = norm(Jt) * w_q
            for I in 1:n_base
                _, NI = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                fe[3n + 2I - 1] -= m * NI * dΓ   # θ₁ = φ₁ DOF (negative, upward bend)
            end
        end
        f[celldofs(fc)] .+= fe
    end
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

# Total strain energy (membrane + bending + shear) summed over all elements.
function strain_energy(dh, scv, u, mat)
    E = 0.0
    for cell in CellIterator(dh)
        reinit!(scv, cell)
        u_e = u[shelldofs(cell)]
        E += FerriteShells.energy_RM(u_e, scv, mat)
    end
    return E
end

# Total potential Π = E_int − F·u.  Newton direction is a descent direction for Π
# when K is positive definite, making this the correct Armijo merit function.
potential(dh, scv, u, mat, F) = strain_energy(dh, scv, u, mat) - dot(F, u)

# Analytical tip displacement for dead-load moment M = λ·M_ref (α = λ·2π rad).
function analytical_tip(λ)
    α = λ * 2π
    iszero(α) && return (0.0, 0.0)
    L * (sin(α)/α - 1),  L * (1 - cos(α))/α
end

# parameters
t = 0.1
W = 1.0
L = 10

# Material model and reference moment
mat   = LinearElastic(1.2e6, 0.0, t)
grid  = make_rollup_grid(; L, W)
EI    = mat.E * W * t^3 / 12
M_ref = 2π * EI / L

# interpolation space and shell
ip    = Lagrange{RefQuadrilateral, 2}()
qr    = QuadratureRule{RefQuadrilateral}(3)
fqr   = FacetQuadratureRule{RefQuadrilateral}(3)
scv   = ShellCellValues(qr, ip, ip; mitc=MITC9)

# degress of freedom
dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

# boundary conditions
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

# allocate matrices and vectors
N_dofs = ndofs(dh)
K   = allocate_matrix(dh)
r   = zeros(N_dofs)
rhs = zeros(N_dofs)
F_ext = zeros(N_dofs)

# apply moment once
apply_end_moment_RM!(F_ext, dh, getfacetset(grid, "free_end"), ip, fqr, M_ref)
tip_node = only(getnodeset(grid, "tip"))

# solver settings for Newton-Rahpson
n_steps = 200
tol      = 1e-6
max_iter = 20
armijo_c = 1e-4   # sufficient-decrease constant for energy Armijo

println("Roll-up cantilever M_ref=$(round(M_ref;digits=4))")
println("  step |  λ     |  u_x_tip  |  u_z_tip  | ux_an   | uz_an   | iters")

u = zeros(N_dofs); tip = []
# Load stepping with cutback: past λ ≈ 0.7 (tip tilt beyond ~250°) the Newton
# radius shrinks and a fixed Δλ = 1/n_steps silently truncates the roll-up.
# Halving the increment on non-convergence and re-growing it on success reaches
# λ = 1 from the same base step count instead of stopping at ~0.7.
let λ = 0.0, Δλ = 1 / n_steps, Δλ_min = 1 / (64 * n_steps), step = 0
    while λ < 1.0 - 1e-12
        λ_trial = min(λ + Δλ, 1.0)
        F = λ_trial .* F_ext
        u_prev = copy(u)
        converged = false; n_iter = 0
        for iter in 1:max_iter
            assemble_global!(K, r, dh, scv, u, mat)
            @. rhs = F - r; apply_zero!(K, rhs, ch)
            rhs_norm = norm(rhs)
            rhs_norm < tol && (converged = true; n_iter = iter - 1; break)
            n_iter = iter
            du     = K \ rhs
            slope  = dot(rhs, du)    # = du'Kdu > 0  (descent slope for Π)
            Π0     = potential(dh, scv, u, mat, F)
            α_ls   = 1.0
            for ks in 1:15
                u_trial = u .+ α_ls .* du
                Π_trial = potential(dh, scv, u_trial, mat, F)
                Π_trial ≤ Π0 - armijo_c * α_ls * slope && break
                α_ls /= 2
            end
            u .+= α_ls .* du
        end
        if !converged
            u .= u_prev
            Δλ /= 2
            Δλ < Δλ_min && error("roll-up: no convergence even at Δλ = $Δλ (λ = $λ)")
            continue
        end
        λ = λ_trial; step += 1
        Δλ = min(2Δλ, 1 / n_steps)
        # extract solution
        tip_ux, tip_uz = 0.0, 0.0
        for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
            if gid == tip_node
                cd = celldofs(cell); tip_ux = u[cd[3I-2]]; tip_uz = u[cd[3I]]; break
            end
        end
        push!(tip, [λ, tip_ux, tip_uz])
        ux_an, uz_an = analytical_tip(λ)
        (step % 10 == 0 || λ ≥ 1.0 - 1e-12) && @printf("  %4d | %.4f | %9.4f | %9.4f | %7.4f | %7.4f | %d\n",
                               step, λ, tip_ux, tip_uz, ux_an, uz_an, n_iter)

        # write to vtk
        VTKGridFile("cantilever_rollup", dh) do vtk
            write_solution(vtk, dh, u)
        end
    end
end

using Plots
sol = analytical_tip.(0:0.05:1)
x, y = getindex.(sol,1), getindex.(sol, 2)
scatter([-x./L, y./L], 0:0.05:1, marker=:o, label=["uₐ-analytic" "wₐ-analytic"])
tipx, tipy, λ = getindex.(tip, 2), getindex.(tip, 3), getindex.(tip, 1)
plot!([-tipx./L, tipy./L], λ, label=["uₐ" "wₐ"], xlabel="Tip delfection (/L)",
        ylabel="Load factor (λ)")
