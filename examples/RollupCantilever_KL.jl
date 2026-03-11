using FerriteShells
using LinearAlgebra
using Printf

# Cantilever roll-up under dead-load end moment (Sze, Liu & Lo 2004, Problem 1).
# L=10, W=1, t=0.1, E=1.2e6, ν=0.
# At full load M_ref = 2π·EI/L the tip forms a complete circle:
#   u_x = L(sin(α)/α - 1),  u_z = L(1−cos(α))/α,  α = ML/(EI).
# At α=2π: u_x = -L = -10, u_z = 0.
#
# Moment application (KL natural BC):
#   δW_ext = m · ∫_edge δ(∂w/∂x) dΓ  →  f_{z,I} = m · ∫_edge ∂N_I/∂x dΓ
# where m = M/W is the moment per unit width.
#
# Clamped BC: fix all DOFs at x=0, plus w=0 over the first element column
# (x ≤ Δx) to enforce zero slope at the clamped end (KL workaround).

const L = 10.0
const W = 1.0
const t = 0.1
const n_x = 20    # elements along length
const n_y = 2     # elements across width

function make_rollup_grid()
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L, 0.0)), Vec{2}((L, W)), Vec{2}((0.0, W))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (n_x, n_y), corners))
    Δx   = L / n_x
    addfacetset!(grid, "clamped",     x -> x[1] ≈ 0.0)
    addfacetset!(grid, "free_end",    x -> x[1] ≈ L)
    addnodeset!(grid, "clamped_col",  x -> x[1] ≤ Δx + 1e-10)
    addnodeset!(grid, "tip",          x -> x[1] ≈ L && x[2] ≈ W/2)
    return grid
end

# Dead-load end moment (KL): f_{z,I} = m · ∫_edge ∂N_I/∂x dΓ
# m = M/W = moment per unit width, about the y-axis (bending in xz-plane).
function apply_end_moment_KL!(f, dh, facetset, ip, fqr, m)
    n_base = getnbasefunctions(ip)
    fe     = zeros(ndofs_per_cell(dh))
    for fc in FacetIterator(dh, facetset)
        fill!(fe, 0.0)
        x        = getcoordinates(fc)
        facet_nr = fc.current_facet_id
        qr_f     = fqr.facet_rules[facet_nr]
        tdir     = facet_nr ∈ (1, 3) ? 1 : 2   # tangential reference direction
        ndir     = facet_nr ∈ (1, 3) ? 2 : 1   # normal reference direction
        for (ξ, w_q) in zip(qr_f.points, qr_f.weights)
            Jt = zero(Vec{3,Float64})
            Jn = zero(Vec{3,Float64})
            for I in 1:n_base
                dN, _ = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                Jt   += dN[tdir] * x[I]
                Jn   += dN[ndir] * x[I]
            end
            dΓ     = norm(Jt) * w_q
            Jn_mag = norm(Jn)
            for I in 1:n_base
                dN, _   = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                dNdn    = dN[ndir] / Jn_mag   # ∂N_I/∂x (outward normal direction)
                fe[3I] += m * dNdn * dΓ
            end
        end
        f[celldofs(fc)] .+= fe
    end
end

function assemble_global!(K, r, dh, scv, u, mat)
    n_e = ndofs_per_cell(dh)
    ke  = zeros(n_e, n_e)
    re  = zeros(n_e)
    asm = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        u_e = u[celldofs(cell)]
        membrane_tangent_KL!(ke, scv, u_e, mat)
        membrane_residuals_KL!(re, scv, u_e, mat)
        bending_tangent_KL!(ke, scv, u_e, mat)
        bending_residuals_KL!(re, scv, u_e, mat)
        assemble!(asm, celldofs(cell), ke, re)
    end
end

function assemble_residual!(r, dh, scv, u, mat)
    fill!(r, 0.0)
    re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell)
        u_e = u[celldofs(cell)]
        membrane_residuals_KL!(re, scv, u_e, mat)
        bending_residuals_KL!(re, scv, u_e, mat)
        r[celldofs(cell)] .+= re
    end
end

# Analytical solution for dead-load end moment M = λ·M_ref, M_ref = 2π·EI/L.
function analytical_tip(λ; L=L)
    α = λ * 2π
    iszero(α) && return (0.0, 0.0)
    return L * (sin(α)/α - 1),  L * (1 - cos(α))/α
end

mat  = LinearElastic(1.2e6, 0.0, t)
EI   = mat.E * W * t^3 / 12           # bending stiffness
M_ref = 2π * EI / L                   # moment for full roll-up
m_ref = M_ref / W                     # moment per unit width

grid = make_rollup_grid()
ip   = Lagrange{RefQuadrilateral, 2}()
qr   = QuadratureRule{RefQuadrilateral}(3)
fqr  = FacetQuadratureRule{RefQuadrilateral}(3)
scv  = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"),    x -> zero(x), [1,2,3]))
add!(ch, Dirichlet(:u, getnodeset(grid, "clamped_col"), x -> 0.0,     [3]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dofs = ndofs(dh)
K = allocate_matrix(dh)
r = zeros(N_dofs)
r_trial = zeros(N_dofs)
rhs = zeros(N_dofs)

# Dead-load moment force vector (assembled once, scaled by λ each step)
F_ext = zeros(N_dofs)
apply_end_moment_KL!(F_ext, dh, getfacetset(grid, "free_end"), ip, fqr, m_ref)

tip_node = only(getnodeset(grid, "tip"))

n_steps  = 20
tol      = 1e-8
max_iter = 30

println("Roll-up cantilever (KL, Q9), M_ref = $(round(M_ref; digits=4))")
println("  step |  λ   |  u_x_tip   |  u_z_tip   | u_x_an  | u_z_an  | iters")

u = zeros(N_dofs)
for step in 1:n_steps
    λ      = step / n_steps
    F      = λ .* F_ext
    u_prev = copy(u)

    converged = false
    n_iter    = 0
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)
        @. rhs = F - r
        apply_zero!(K, rhs, ch)
        rhs_norm = norm(rhs)
        if rhs_norm < tol
            converged = true; n_iter = iter - 1; break
        end
        n_iter = iter
        du = K \ rhs
        α_ls = 1.0
        for _ in 1:8
            assemble_residual!(r_trial, dh, scv, u .+ α_ls .* du, mat)
            @. rhs = F - r_trial
            for dof in ch.prescribed_dofs; rhs[dof] = 0.0; end
            norm(rhs) < rhs_norm && break
            α_ls /= 2
        end
        u .+= α_ls .* du
    end

    if !converged
        u .= u_prev
        @warn "step $step (λ=$λ) did not converge; rolled back"
        break
    end

    tip_ux, tip_uz = 0.0, 0.0
    for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
        if gid == tip_node
            cd = celldofs(cell)
            tip_ux = u[cd[3I-2]]; tip_uz = u[cd[3I]]
            break
        end
    end

    ux_an, uz_an = analytical_tip(λ)
    @printf("  %4d | %.2f | %10.4f | %10.4f | %7.4f | %7.4f | %d\n",
            step, λ, tip_ux, tip_uz, ux_an, uz_an, n_iter)
end
