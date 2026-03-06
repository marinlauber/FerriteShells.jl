using FerriteShells
using LinearAlgebra
using Test
using Random

function make_scv(; qr_order=1)
    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(qr_order)
    return ShellCellValues(qr, ip, ip)
end

# Unit square in the XY plane
const X_UNIT_SQUARE = [
    Vec{3}((0.0, 0.0, 0.0)),
    Vec{3}((1.0, 0.0, 0.0)),
    Vec{3}((1.0, 1.0, 0.0)),
    Vec{3}((0.0, 1.0, 0.0)),
]

# Helper: compute residual into a fresh zeroed vector
function residual(scv, x, u_vec, mat)
    re = zeros(length(u_vec))
    membrane_residuals!(re, scv, x, reinterpret(Vec{3, Float64}, u_vec), mat)
    return re
end

# Helper: compute tangent into a fresh zeroed matrix
function tangent(scv, x, u_vec, mat)
    ke = zeros(length(u_vec), length(u_vec))
    membrane_tangent!(ke, scv, x, reinterpret(Vec{3, Float64}, u_vec), mat)
    return ke
end

# Central-difference numerical tangent: O(ε²) accuracy
function numerical_tangent(scv, x, u_vec, mat; ε=1e-5)
    n = length(u_vec)
    Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u_vec); up[j] += ε
        um = copy(u_vec); um[j] -= ε
        Kfd[:, j] = (residual(scv, x, up, mat) .- residual(scv, x, um, mat)) ./ (2ε)
    end
    return Kfd
end

@testset "assembly.jl" begin
    @testset "membrane residuals and tangent" begin
        # make a material and a cell
        mat = LinearElastic(1.0, 0.3, 0.1)
        scv = make_scv(qr_order=1)
        reinit!(scv, X_UNIT_SQUARE)
        n_dof = 12  # 4 nodes × 3 DOFs
        # test zero displacement: should give zero residual
        re = residual(scv, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(re) ≤ 1e-14
        # Uniform shift in x-direction — pure translation, zero strain
        u_vec = repeat([0.5, 0.0, 0.0], 4)
        re = residual(scv, X_UNIT_SQUARE, u_vec, mat)
        @test norm(re) ≤ 1e-13
        # rigid body rotation should give zero membrane residual
        R = Tensor{2,3}([cos(-π/2) -sin(-π/2) 0; sin(-π/2) cos(-π/2) 0; 0 0 1])
        u = vcat([(R⋅xᵢ)-xᵢ for xᵢ in X_UNIT_SQUARE]...) # passed
        membrane_residuals!(re, scv, X_UNIT_SQUARE, reinterpret(Vec{3,Float64}, u), mat)
        @test norm(re) ≤ 10eps(Float64) && sum(re) ≤ 10eps(Float64)
        # Tangent symmetry (zero displacement) should be exactly symmetric since geometric stiffness is zero.
        ke = tangent(scv, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(ke .- ke') ≤ 1e-14 * norm(ke)
        # tangent symmetry (nonzero displacement)
        Random.seed!(1)
        u_vec = 0.05 * randn(n_dof)
        ke = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        @test norm(ke .- ke') / norm(ke) ≤ 1e-12
        # FD consistency (small displacement)
        # Small displacement: geometric stiffness is negligible; material part dominates.
        Random.seed!(2)
        u_vec = 0.01 * randn(n_dof)
        ke    = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        ke_fd = numerical_tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7
        # FD consistency (moderate displacement)
        # Moderate displacement: geometric stiffness is non-negligible.
        Random.seed!(3)
        u_vec = 0.3 * randn(n_dof)
        ke = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        ke_fd = numerical_tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7
        # autodiff consistency
        # using ForwardDiff
        # ke_ad = ForwardDiff.jacobian(u -> residual(scv, X_UNIT_SQUARE, u, mat), u_vec)
        # rel_err = norm(ke .- ke_ad) / (norm(ke_ad) + 1e-14)
        # @test rel_err < 1e-7
        # tangent positive semi-definiteness: should have no significantly negative eigenvalues.
        # For a free element, rigid-body modes yield zero eigenvalues.
        Random.seed!(4)
        u_vec = 0.01 * randn(n_dof)
        ke = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        λ  = eigvals(Symmetric(ke))
        @test minimum(λ) ≥ -1e-10 * maximum(abs, λ)
        # check that higher-order rules work correctly.
        scv2 = make_scv(qr_order=2)
        reinit!(scv2, X_UNIT_SQUARE)
        re2 = residual(scv2, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(re2) ≤ 1e-14
    end
    @testset "bending residuals and tangent" begin
        # TODO implement bending residuals and tangent, then add tests here
    end
    @testset "combined membrane and bending" begin
        # TODO implement combined residuals and tangent, then add tests here
    end
    @testset "shear residuals and tangent" begin
        # TODO implement shear residuals and tangent, then add tests here
    end
end



# ── Patch mesh ────────────────────────────────────────────────────────────────
#
# Classic 5-element quadrilateral patch: 4 outer ring elements + 1 central
# element. The 4 inner nodes (5–8) are intentionally placed off-grid to make
# every element non-orthogonal.
#
#   4(0,10) ─────────────── 3(10,10)
#     │     8(4,7)──7(8,7)    │
#     │  E4  │   E5  │  E3   │
#     │     5(2,2)──6(8,3)    │
#     │         E1      E2    │
#   1(0,0)  ─────────────── 2(10,0)
#
#   E1:(1,2,6,5)  E2:(2,3,7,6)  E3:(3,4,8,7)
#   E4:(4,1,5,8)  E5:(5,6,7,8)  ← central element
#
# Boundary nodes : 1,2,3,4  (outer corners, on the patch perimeter)
# Interior nodes : 5,6,7,8  (inner quad corners, shared only between elements)
#
# Patch test condition: for a linear displacement field u(x,y) the internal
# forces at the 4 interior (free) nodes must be zero.  This holds iff the
# element correctly reproduces constant strain states on arbitrary quads.

function patch_grid_2(; primitive=Quadrilateral)
    nodes = [
        Vec{3}(( 0.0,  0.0, 0.0)),
        Vec{3}((10.0,  0.0, 0.0)),
        Vec{3}((10.0, 10.0, 0.0)),
        Vec{3}(( 0.0, 10.0, 0.0)),
        Vec{3}(( 2.0,  2.0, 0.0)),
        Vec{3}(( 8.0,  3.0, 0.0)),
        Vec{3}(( 8.0,  7.0, 0.0)),
        Vec{3}(( 4.0,  7.0, 0.0)),
    ]
    if primitive == Triangle
        cells = [(1,2,5), (2,6,5), (2,3,6), (3,7,6), (3,8,7),
                 (3,4,8), (4,5,8), (4,1,5), (5,6,8), (6,7,8)]
    else
        cells = [(1,2,6,5), (2,3,7,6), (3,4,8,7), (4,1,5,8),
                 (5,6,7,8)]
    end
    return Grid([primitive(c) for c in cells], Node.(nodes))
end

# ── Assembly helpers ──────────────────────────────────────────────────────────

function assemble_residual!(r, dh, scv, u, mat)
    n  = ndofs_per_cell(dh)
    re = zeros(n)
    for cell in CellIterator(dh)
        fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = reinterpret(Vec{3, Float64}, u[celldofs(cell)])
        membrane_residuals!(re, scv, x, u_e, mat)
        r[celldofs(cell)] .+= re
    end
end

function assemble_tangent_and_residual!(K, r, dh, scv, u, mat)
    n         = ndofs_per_cell(dh)
    ke        = zeros(n, n)
    re        = zeros(n)
    assembler = start_assemble(K, r)
    for cell in CellIterator(dh)
        fill!(ke, 0.0); fill!(re, 0.0)
        reinit!(scv, cell)
        x   = getcoordinates(cell)
        u_e = reinterpret(Vec{3, Float64}, u[celldofs(cell)])
        membrane_tangent!(ke, scv, x, u_e, mat)
        membrane_residuals!(re, scv, x, u_e, mat)
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

# Build a global DOF vector by evaluating f at every node.
# Relies on the standard Ferrite layout for ip^3 with a single :u field:
#   celldofs = [u1x, u1y, u1z, u2x, u2y, u2z, ...]  (blocked by node)
function make_u_exact(dh, f::Function)
    u = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        coords = getcoordinates(cell)
        dofs   = celldofs(cell)
        for i in eachindex(coords)
            val          = f(coords[i])
            u[dofs[3i-2]] = val[1]
            u[dofs[3i-1]] = val[2]
            u[dofs[3i  ]] = val[3]
        end
    end
    return u
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Membrane patch test (5-element non-orthogonal quad mesh)" begin

    # ── Setup ─────────────────────────────────────────────────────────────────
    grid = patch_grid_2(primitive = Quadrilateral)

    # Outer corners lie on the patch perimeter; everything else is interior.
    addnodeset!(grid, "boundary",
        x -> isapprox(x[1],  0.0, atol=1e-10) || isapprox(x[1], 10.0, atol=1e-10) ||
                isapprox(x[2],  0.0, atol=1e-10) || isapprox(x[2], 10.0, atol=1e-10))
    addnodeset!(grid, "interior",
        x -> !( isapprox(x[1],  0.0, atol=1e-10) || isapprox(x[1], 10.0, atol=1e-10) ||
                isapprox(x[2],  0.0, atol=1e-10) || isapprox(x[2], 10.0, atol=1e-10)))

    ip  = Lagrange{RefQuadrilateral, 1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    scv = ShellCellValues(qr, ip, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    close!(dh)

    mat = LinearElastic(1.0e6, 0.3, 0.01)

    # Linear displacement field representing a uniform in-plane strain state.
    # Amplitudes are small so the nonlinear (GL) correction is negligible.
    ε_xx, ε_yy, γ_xy = 1.0e-3, 2.0e-3, 5.0e-4
    u_lin(x::Vec{3}) = Vec{3}((ε_xx * x[1] + 0.5γ_xy * x[2],
                                0.5γ_xy * x[1] + ε_yy * x[2],
                                0.0))

    # ── Test 1: interior residual is zero under the exact linear field ─────────
    # Apply u_lin to every node (interior included) and check that the
    # assembled residual at the free (interior) DOFs is numerically zero.
    @testset "Interior residual vanishes under exact linear displacement" begin
        u_exact = make_u_exact(dh, u_lin)

        r = zeros(ndofs(dh))
        assemble_residual!(r, dh, scv, u_exact, mat)

        # Use a temporary ConstraintHandler to identify free DOFs.
        ch_tmp = ConstraintHandler(dh)
        add!(ch_tmp, Dirichlet(:u, getnodeset(grid, "boundary"),
                x -> zero(x), [1, 2, 3]))
        close!(ch_tmp)
        free_dofs = setdiff(1:ndofs(dh), ch_tmp.prescribed_dofs)

        # The boundary reactions can be large; only interior entries must be zero.
        @test norm(r[free_dofs]) ≤ 1e-8 * norm(r)
        @test sum(r) < 1e-14
    end

    # ── Test 2: solve with linear BCs recovers the exact interior field ────────
    # Apply u_lin as Dirichlet BCs on the 4 outer corners only, assemble the
    # linearised stiffness (tangent at u = 0), and solve for the 4 inner nodes.
    # The solution must coincide with u_lin at all interior nodes.
    @testset "Linear solve with boundary data recovers interior displacements" begin
        K = allocate_matrix(dh)
        r = zeros(ndofs(dh))
        # Tangent at u=0: zero strain → zero stress → Kgeo = 0, only Kmat.
        assemble_tangent_and_residual!(K, r, dh, scv, zeros(ndofs(dh)), mat)
        # r ≡ 0 at this point (undeformed config → zero residual).

        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getnodeset(grid, "boundary"),
                x -> u_lin(x), [1, 2, 3]))
        # The membrane has no out-of-plane (z) stiffness.  Interior z-DOFs are
        # zero-energy modes that make K singular.  Pin them to zero to regularise.
        add!(ch, Dirichlet(:u, getnodeset(grid, "interior"), x -> 0.0, [3]))
        close!(ch)
        Ferrite.update!(ch, 0.0)

        apply!(K, r, ch)
        u_solved = K \ r

        u_exact   = make_u_exact(dh, u_lin)
        free_dofs = setdiff(1:ndofs(dh), ch.prescribed_dofs)

        err = norm(u_solved[free_dofs] .- u_exact[free_dofs])
        ref = norm(u_exact[free_dofs])
        @test err ≤ 1e-8 * ref
    end
end
