using FerriteShells
using LinearAlgebra
using Printf

# Cantilever roll-up under dead-load end moment (Sze, Liu & Lo 2004, Problem 1).
# L=10, W=1, t=0.1, E=1.2e6, ν=0.
# Analytical: u_x = L(sin(α)/α - 1),  u_z = L(1−cos(α))/α,  α = ML/(EI).
# At full load M_ref = 2π·EI/L the tip forms a complete circle (α=2π).
#
# Uses Reissner–Mindlin (5 DOF/node).  The dead-load end moment is applied as
# a distributed load on the φ₁ rotation DOFs at x=L (consistent with the RM
# virtual work δW = ∫ m·δφ₁ dΓ where m = M/W is moment per unit width).
#
# Limitation: the director parametrisation d = G₃ + φ₁T₁ + φ₂T₂ develops a
# singularity at α = 90° (φ₁ → −∞).  The solver is therefore run only up to
# α_max = 0.8·π/2 ≈ 72° (M = M_ref/5) where convergence is reliable.  A full
# geometrically-exact formulation (Rodrigues/quaternion directors) is required
# for the complete 360° roll-up.

const L = 10.0
const W = 1.0
const t = 0.1
const n_x = 20    # elements along length
const n_y = 2     # elements across width

function make_rollup_grid()
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((L, 0.0)), Vec{2}((L, W)), Vec{2}((0.0, W))]
    grid    = shell_grid(generate_grid(QuadraticQuadrilateral, (n_x, n_y), corners))
    addfacetset!(grid, "clamped",  x -> x[1] ≈ 0.0)
    addfacetset!(grid, "free_end", x -> x[1] ≈ L)
    addnodeset!(grid,  "tip",      x -> x[1] ≈ L && x[2] ≈ W/2)
    return grid
end

# Apply dead-load moment M (about y-axis, bending in xz-plane) to RM shell.
# Virtual work: δW = ∫ m·δφ₁ dΓ  →  f_{φ₁,I} = m · ∫_edge N_I dΓ
# where m = M/W and φ₁ is the first rotation DOF (tips the director toward T₁=x̂).
# In the two-field DofHandler [u(ip^3), θ(ip^2)], the θ₁ DOF of node I in a cell
# sits at celldofs[3n + 2I − 1].
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
                _, NI = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                dN, _ = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                Jt   += dN[tdir] * x[I]
            end
            dΓ = norm(Jt) * w_q
            for I in 1:n_base
                _, NI = Ferrite.reference_shape_gradient_and_value(ip, ξ, I)
                fe[3n + 2I - 1] += m * NI * dΓ   # θ₁ = φ₁ DOF
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
        membrane_tangent_RM!(ke, scv, u_e, mat); membrane_residuals_RM!(re, scv, u_e, mat)
        bending_tangent_RM!(ke, scv, u_e, mat);  bending_residuals_RM!(re, scv, u_e, mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
end

function assemble_residual!(r, dh, scv, u, mat)
    fill!(r, 0.0); re = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        fill!(re, 0.0); reinit!(scv, cell); u_e = u[shelldofs(cell)]
        membrane_residuals_RM!(re, scv, u_e, mat)
        bending_residuals_RM!(re, scv, u_e, mat)
        r[shelldofs(cell)] .+= re
    end
end

# Analytical tip displacement for dead-load moment M = λ·M_ref (α = λ·2π).
function analytical_tip(λ)
    α = λ * 2π
    iszero(α) && return (0.0, 0.0)
    L * (sin(α)/α - 1),  L * (1 - cos(α))/α
end

mat   = LinearElastic(1.2e6, 0.0, t)
EI    = mat.E * W * t^3 / 12
M_ref = 2π * EI / L
m_ref = M_ref / W

grid  = make_rollup_grid()
ip    = Lagrange{RefQuadrilateral, 2}()
qr    = QuadratureRule{RefQuadrilateral}(3)
fqr   = FacetQuadratureRule{RefQuadrilateral}(3)
scv   = ShellCellValues(qr, ip, ip)

dh = DofHandler(grid)
add!(dh, :u, ip^3)
add!(dh, :θ, ip^2)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zero(x), [1,2,3]))
add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
close!(ch); Ferrite.update!(ch, 0.0)

N_dofs = ndofs(dh)
K = allocate_matrix(dh)
r = zeros(N_dofs); r_trial = zeros(N_dofs); rhs = zeros(N_dofs)

F_ext = zeros(N_dofs)
apply_end_moment_RM!(F_ext, dh, getfacetset(grid, "free_end"), ip, fqr, m_ref)

tip_node = only(getnodeset(grid, "tip"))

# Load up to α_max = 0.8·(π/2) ≈ 72°  (stay clear of the 90° singularity)
α_max   = 0.8 * (π / 2)
λ_max   = α_max / (2π)
n_steps = 20
tol     = 1e-8
max_iter = 30

println("Roll-up cantilever (RM, Q9)  M_ref=$(round(M_ref;digits=4))  α_max=$(round(rad2deg(α_max);digits=1))°")
println("  step |  λ    |  α(°) |  u_x_tip  |  u_z_tip  | ux_an   | uz_an   | iters")

u = zeros(N_dofs)
for step in 1:n_steps
    λ      = step / n_steps * λ_max
    F      = λ .* F_ext
    u_prev = copy(u)

    converged = false; n_iter = 0
    for iter in 1:max_iter
        assemble_global!(K, r, dh, scv, u, mat)
        @. rhs = F - r; apply_zero!(K, rhs, ch)
        rhs_norm = norm(rhs)
        if rhs_norm < tol
            converged = true; n_iter = iter - 1; break
        end
        n_iter = iter
        du = K \ rhs
        α_ls = 1.0
        for _ in 1:10
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
            cd = celldofs(cell); tip_ux = u[cd[3I-2]]; tip_uz = u[cd[3I]]; break
        end
    end
    ux_an, uz_an = analytical_tip(λ)
    α_deg = λ * 360
    @printf("  %4d | %.4f | %5.1f | %9.4f | %9.4f | %7.4f | %7.4f | %d\n",
            step, λ, α_deg, tip_ux, tip_uz, ux_an, uz_an, n_iter)
end
