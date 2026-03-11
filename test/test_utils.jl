using FerriteShells
using LinearAlgebra
using Test

@testset "shelldofs DOF reordering" begin
    # Build a small 2×2 Q4 mesh and a two-field DofHandler (:u ip^3, :θ ip^2).
    # shelldofs must reorder celldofs from [u_block | θ_block] to per-node interleaved
    # [u1,u2,u3,θ1,θ2] layout required by the RM assembly functions.
    grid = shell_grid(generate_grid(Quadrilateral, (2, 2),
                                   Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))
    ip_u = Lagrange{RefQuadrilateral, 1}()
    dh   = DofHandler(grid)
    add!(dh, :u, ip_u^3)
    add!(dh, :θ, ip_u^2)
    close!(dh)

    for cell in CellIterator(dh)
        cd = celldofs(cell)   # [u_block (3n) | θ_block (2n)]
        sd = shelldofs(cell)  # interleaved [u1,u2,u3,θ1,θ2, ...]
        n  = length(cd) ÷ 5  # nodes per cell (4 for Q4)

        @test length(sd) == length(cd) == 5n
        @test sort(sd) == sort(cd)   # same set of DOFs, just reordered

        for I in 1:n
            @test sd[5I-4:5I-2] == cd[3I-2:3I]     # u₁,u₂,u₃ for node I
            @test sd[5I-1]      == cd[3n + 2I-1]    # θ₁ for node I
            @test sd[5I  ]      == cd[3n + 2I  ]    # θ₂ for node I
        end
    end

    # Also test with Q9 elements to cover the higher-order case.
    ip_q9  = Lagrange{RefQuadrilateral, 2}()
    grid9  = shell_grid(generate_grid(QuadraticQuadrilateral, (2, 2),
                                      Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))
    dh9    = DofHandler(grid9)
    add!(dh9, :u, ip_q9^3)
    add!(dh9, :θ, ip_q9^2)
    close!(dh9)

    for cell in CellIterator(dh9)
        cd = celldofs(cell)
        sd = shelldofs(cell)
        n  = length(cd) ÷ 5   # 9 for Q9

        @test length(sd) == length(cd) == 5n
        @test sort(sd) == sort(cd)
        for I in 1:n
            @test sd[5I-4:5I-2] == cd[3I-2:3I]
            @test sd[5I-1]      == cd[3n + 2I-1]
            @test sd[5I  ]      == cd[3n + 2I  ]
        end
    end
end

@testset "assemble_traction! regression: two-field DofHandler" begin
    # Regression for the bug where assemble_traction! used the interleaved 5-DOF
    # block (5I-4:5I-2) for a two-field DofHandler, scattering force into θ-DOFs.
    # Fix: detect two-field layout and use the 3-DOF block (3I-2:3I) for u only.
    grid = shell_grid(generate_grid(Quadrilateral, (1, 1),
                                   Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))
    addfacetset!(grid, "right", x -> isapprox(x[1], 1.0, atol=1e-10))

    ip  = Lagrange{RefQuadrilateral, 1}()
    fqr = FacetQuadratureRule{RefQuadrilateral}(2)
    t_z = Vec{3}((0.0, 0.0, 1.0))   # unit z-traction; right edge length = 1

    # Single-field reference (3 DOFs/node).
    dh1 = DofHandler(grid); add!(dh1, :u, ip^3); close!(dh1)
    f1  = zeros(ndofs(dh1))
    assemble_traction!(f1, dh1, getfacetset(grid, "right"), ip, fqr, t_z)

    # Two-field (5 DOFs/node).
    dh2 = DofHandler(grid); add!(dh2, :u, ip^3); add!(dh2, :θ, ip^2); close!(dh2)
    f2  = zeros(ndofs(dh2))
    assemble_traction!(f2, dh2, getfacetset(grid, "right"), ip, fqr, t_z)

    # Total z-force must equal traction × edge_length = 1 × 1 = 1 in both cases.
    @test sum(f1) ≈ 1.0 atol=1e-10
    @test sum(f2) ≈ 1.0 atol=1e-10

    # θ-DOFs (last 2*n_nodes entries in the global vector) must receive no force.
    n_nodes = getnnodes(grid)
    @test iszero(f2[3n_nodes+1:end])

    # Functional regression: solve a clamped RM beam, check positive z-tip deflection.
    # If traction landed on θ-DOFs instead of u₃-DOFs, the tip would not deflect.
    grid2 = shell_grid(generate_grid(Quadrilateral, (4, 1),
                                    Vec{2}((0.0, 0.0)), Vec{2}((4.0, 1.0))))
    addfacetset!(grid2, "right", x -> isapprox(x[1], 4.0, atol=1e-10))
    addnodeset!(grid2, "left",  x -> isapprox(x[1], 0.0, atol=1e-10))

    ip2  = Lagrange{RefQuadrilateral, 1}()
    fqr2 = FacetQuadratureRule{RefQuadrilateral}(2)
    dh3  = DofHandler(grid2); add!(dh3, :u, ip2^3); add!(dh3, :θ, ip2^2); close!(dh3)
    f3   = zeros(ndofs(dh3))
    assemble_traction!(f3, dh3, getfacetset(grid2, "right"), ip2, fqr2, t_z)

    mat3  = LinearElastic(1e3, 0.3, 0.1)
    scv3  = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip2, ip2)
    n_el  = ndofs_per_cell(dh3)
    K3    = allocate_matrix(dh3)
    asmb3 = start_assemble(K3, zeros(ndofs(dh3)))
    ke3   = zeros(n_el, n_el); re3 = zeros(n_el)
    for cell in CellIterator(dh3)
        fill!(ke3, 0.0); fill!(re3, 0.0)
        reinit!(scv3, cell)
        x = getcoordinates(cell); u_e = zeros(n_el)
        membrane_tangent_RM!(ke3, scv3, x, u_e, mat3)
        bending_tangent_RM!(ke3, scv3, x, u_e, mat3)
        assemble!(asmb3, shelldofs(cell), ke3, re3)
    end
    dbc3 = ConstraintHandler(dh3)
    add!(dbc3, Dirichlet(:u, getnodeset(grid2, "left"), x -> zeros(3), [1,2,3]))
    add!(dbc3, Dirichlet(:θ, getnodeset(grid2, "left"), x -> zeros(2), [1,2]))
    close!(dbc3); Ferrite.update!(dbc3, 0.0)
    apply!(K3, f3, dbc3)
    u3 = K3 \ f3

    ph3    = PointEvalHandler(grid2, [Vec{3}((4.0, 0.5, 0.0))])
    u_tip3 = first(evaluate_at_points(ph3, dh3, u3, :u))
    @test u_tip3[3] > 0.0   # positive z-deflection under +z traction
end
