using FerriteShells
using LinearAlgebra
using Test

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

        # The boundary reactions can be large; only interior entries must be zero.
        @test norm(r[ch_tmp.free_dofs]) ≤ 1e-8 * norm(r)
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

        err = norm(u_solved[ch.free_dofs] .- u_exact[ch.free_dofs])
        ref = norm(u_exact[ch.free_dofs])
        @test err ≤ 1e-8 * ref
    end
end
