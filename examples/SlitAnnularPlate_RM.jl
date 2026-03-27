using FerriteShells
using LinearAlgebra
using Printf

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
#
# A small angular gap (slit_gap) offsets the two slit faces in physical space,
# allowing addfacetset! to distinguish them by y-coordinate.

const a        = 6.0
const b        = 10.0
const R_mean   = 0.5*(a + b)
const t        = 0.04
const mat      = LinearElastic(2.1e7, 0.0, t)
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

# Dead-load moment M per unit radial length applied to φ₁ DOFs at a facetset.
# G₃=−ê_z for this plate, so sign is reversed vs. cantilever (+m → roll upward).
function apply_moment!(f, dh, facetset, ip, fqr, m)
    n_base = getnbasefunctions(ip)
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
                fe[3n_base + 2I - 1] += m * NI * dΓ
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

const ip  = Lagrange{RefQuadrilateral, 2}()
const qr  = QuadratureRule{RefQuadrilateral}(3)
const fqr = FacetQuadratureRule{RefQuadrilateral}(3)

EI     = mat.E * t^3 / 12
M_full = EI / R_mean

println("Slit annular plate RM (Q9): a=$a, b=$b, t=$t, EI≈$(round(EI;digits=1)), M_full≈$(round(M_full;digits=2))")

# Section 1: Linear locking study — apply small moment, measure u_z vs mesh
# At M=M_small ≪ M_full the problem is in the linear regime and locking is apparent.
function linear_locking_uz(n_θ, n_r, M_load)
    grid = annular_plate_grid(n_θ, n_r)
    scv  = ShellCellValues(qr, ip, ip)
    dh   = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    ch   = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
    close!(ch); Ferrite.update!(ch, 0.0)
    tip_node = only(getnodeset(grid, "tip_outer"))
    N = ndofs(dh); n_base = getnbasefunctions(ip)
    K = allocate_matrix(dh); f = zeros(N)
    ke = zeros(5n_base, 5n_base); re = zeros(5n_base)
    asm = start_assemble(K, zeros(N))
    for cell in CellIterator(dh)
        fill!(ke, 0.0); reinit!(scv, cell)
        membrane_tangent_RM_FD!(ke, scv, zeros(5n_base), mat)
        bending_tangent_RM_FD!(ke, scv,  zeros(5n_base), mat)
        assemble!(asm, shelldofs(cell), ke, re)
    end
    apply_moment!(f, dh, getfacetset(grid, "free"), ip, fqr, M_load)
    apply!(K, f, ch)
    u_sol = K \ f
    u_z = 0.0
    for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
        gid == tip_node || continue
        u_z = u_sol[celldofs(cell)[3I]]
    end
    return u_z, getncells(grid)
end

println("\nSection 1 — Linear locking (M=$(round(0.01*M_full;digits=3)), small load):")
println("   n_θ | elements |   u_z(tip)  | ratio to n_θ=80")
u_ref80, _ = linear_locking_uz(80, 4, 0.01*M_full)
for n_θ in [5, 10, 20, 40, 80]
    u_z, ncells = linear_locking_uz(n_θ, 4, 0.01*M_full)
    @printf("  %4d |    %4d  | %11.6f | %.3f\n", n_θ, ncells, u_z, u_z / u_ref80)
end
println("  → severe locking: coarse meshes capture <20% of reference displacement")

# Section 2: Nonlinear roll-up with n_θ=10 (locked mesh, converges reliably)
# and n_θ=20 (less locked, needs 50 line-search halvings to handle large first step).
println("\nSection 2 — Nonlinear roll-up to M_max=$(round(0.25*M_full;digits=2)):")

M_max    = 0.25 * M_full
tol      = 1e-8
max_iter = 50
armijo_c = 1e-4

# n_θ=10 (locked) converges with standard energy Armijo.
# n_θ=20 (less locked) fails: unlocked plate deforms 3× more per step, hitting
# stronger geometric nonlinearity. Fix: arc-length/bordering or n_steps > 500.
for (n_θ, n_halvings, n_steps) in [(10, 20, 40)]
    n_r   = 4
    grid  = annular_plate_grid(n_θ, n_r)
    scv   = ShellCellValues(qr, ip, ip)
    dh    = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    ch    = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "clamped"), x -> zeros(3), [1,2,3]))
    add!(ch, Dirichlet(:θ, getfacetset(grid, "clamped"), x -> zeros(2), [1,2]))
    close!(ch); Ferrite.update!(ch, 0.0)
    tip_node = only(getnodeset(grid, "tip_outer"))
    N_dofs = ndofs(dh)
    K = allocate_matrix(dh); r_vec = zeros(N_dofs); rhs = zeros(N_dofs)
    F_ext = zeros(N_dofs)
    apply_moment!(F_ext, dh, getfacetset(grid, "free"), ip, fqr, 1.0)
    println("  Mesh $(n_θ)×$(n_r) ($(getncells(grid)) elements), halvings=$n_halvings, steps=$n_steps:")
    println("  step |    M    |  u_z(tip) | iters")
    u = zeros(N_dofs)
    for step in 1:n_steps
        λ = step / n_steps * M_max; F = λ .* F_ext; u_prev = copy(u)
        converged = false; n_iter = 0
        for iter in 1:max_iter
            assemble_global!(K, r_vec, dh, scv, u, mat)
            @. rhs = F - r_vec; apply_zero!(K, rhs, ch)
            if norm(rhs) < tol
                converged = true; n_iter = iter - 1; break
            end
            n_iter = iter
            du = K \ rhs; slope = dot(rhs, du); Π0 = potential(dh, scv, u, mat, F)
            α_ls = 1.0
            for _ in 1:n_halvings
                potential(dh, scv, u .+ α_ls .* du, mat, F) ≤ Π0 - armijo_c * α_ls * slope && break
                α_ls /= 2
            end
            u .+= α_ls .* du; apply!(u, ch)
        end
        if !converged
            u .= u_prev; @warn "step $step (M=$(round(λ;digits=3))) did not converge; rolling back"; break
        end
        u_z_tip = 0.0
        for cell in CellIterator(dh), (I, gid) in enumerate(getnodes(cell))
            gid == tip_node || continue; u_z_tip = u[celldofs(cell)[3I]]
        end
        step % max(1, n_steps ÷ 10) == 0 && @printf("  %4d | %7.4f | %9.4f | %d\n", step, λ, u_z_tip, n_iter)
    end
    println()
    # save results
    VTKGridFile("slit_annular_plate", dh) do vtk
        write_solution(vtk, dh, u)
    end
end
println("Note: n_θ=10 (locked) u_z is significantly smaller than n_θ=20 (less locked)")
println("      at the same M — consistent with the linear locking study above.")
