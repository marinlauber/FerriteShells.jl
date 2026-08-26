using FerriteShells
using LinearAlgebra
using Random
using Test

@testset "function_value on ShellCellValues" begin
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(3)
    scv = ShellCellValues(qr, ip, ip)
    x   = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.0)), Vec{3}((1.0,1.0,0.0)),
           Vec{3}((0.0,1.0,0.0)), Vec{3}((0.5,0.0,0.0)), Vec{3}((1.0,0.5,0.0)),
           Vec{3}((0.5,1.0,0.0)), Vec{3}((0.0,0.5,0.0)), Vec{3}((0.5,0.5,0.0))]
    reinit!(scv, x)
    n = getnbasefunctions(scv.ip_shape)

    # zero displacement → zero function value (all DOF layouts)
    for stride in (3, 5)
        u_zero = zeros(stride * n)
        for qp in 1:getnquadpoints(scv)
            @test norm(Ferrite.function_value(scv, qp, u_zero)) == 0.0
        end
    end

    # KL layout: result matches manual sum N_I * u_I
    Random.seed!(42)
    u_kl = randn(3n)
    for qp in 1:getnquadpoints(scv)
        v_manual = sum(scv.N[I,qp] * Vec{3}((u_kl[3I-2], u_kl[3I-1], u_kl[3I])) for I in 1:n)
        @test norm(Ferrite.function_value(scv, qp, u_kl) - v_manual) < 1e-14
    end

    # RM layout: rotation DOFs (4th and 5th) are ignored, displacements match
    u_rm = randn(5n)
    for qp in 1:getnquadpoints(scv)
        v_manual = sum(scv.N[I,qp] * Vec{3}((u_rm[5I-4], u_rm[5I-3], u_rm[5I-2])) for I in 1:n)
        @test norm(Ferrite.function_value(scv, qp, u_rm) - v_manual) < 1e-14
    end

    # RM: perturbing rotation DOFs alone must not change function_value
    u_rm2 = copy(u_rm)
    for I in 1:n; u_rm2[5I-1] += 1.0; u_rm2[5I] -= 1.0; end
    for qp in 1:getnquadpoints(scv)
        @test norm(Ferrite.function_value(scv, qp, u_rm) - Ferrite.function_value(scv, qp, u_rm2)) < 1e-14
    end

    # Partition of unity: interpolating node coords recovers position (KL layout)
    u_pos = vcat([collect(Tuple(xi)) for xi in x]...)
    for qp in 1:getnquadpoints(scv)
        x_interp = Ferrite.function_value(scv, qp, u_pos)
        x_manual = sum(scv.N[I,qp] * x[I] for I in 1:n)
        @test norm(x_interp - x_manual) < 1e-14
    end
end

@testset "function_gradient on ShellCellValues" begin
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(3)
    scv = ShellCellValues(qr, ip, ip)
    x9  = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.0)), Vec{3}((1.0,1.0,0.0)),
           Vec{3}((0.0,1.0,0.0)), Vec{3}((0.5,0.0,0.0)), Vec{3}((1.0,0.5,0.0)),
           Vec{3}((0.5,1.0,0.0)), Vec{3}((0.0,0.5,0.0)), Vec{3}((0.5,0.5,0.0))]
    reinit!(scv, x9)
    n = getnbasefunctions(scv.ip_shape)

    # zero displacement → zero gradient (KL and RM)
    for stride in (3, 5)
        u_zero = zeros(stride * n)
        for qp in 1:getnquadpoints(scv)
            @test norm(Ferrite.function_gradient(scv, qp, u_zero)) == 0.0
        end
    end

    # pure translation (u = const) → zero gradient (partition of unity: Σ ∂N_I/∂ξ = 0)
    for stride in (3, 5)
        u_trans = zeros(stride * n)
        for I in 1:n; u_trans[stride*(I-1)+1:stride*(I-1)+3] .= [1.3, -0.7, 2.1]; end
        for qp in 1:getnquadpoints(scv)
            @test norm(Ferrite.function_gradient(scv, qp, u_trans)) < 1e-13
        end
    end

    # matches manual sum Σ u_I ⊗ [∂N_I/∂ξ₁, ∂N_I/∂ξ₂, 0]
    # Note: this is the parametric gradient (∂u/∂ξ), not the physical gradient (∂u/∂X).
    # For computing the surface deformation gradient F = I + ∂u/∂ξ, this is the intended quantity.
    Random.seed!(7)
    u_kl = randn(3n)
    for qp in 1:getnquadpoints(scv)
        ∇u_manual = sum(Vec{3}((u_kl[3I-2], u_kl[3I-1], u_kl[3I])) ⊗
                        Vec{3}((scv.dNdξ[I,qp][1], scv.dNdξ[I,qp][2], 0.0)) for I in 1:n)
        @test norm(Ferrite.function_gradient(scv, qp, u_kl) - ∇u_manual) < 1e-14
    end

    # RM: rotation DOFs have no effect on the displacement gradient
    u_rm  = randn(5n); u_rm2 = copy(u_rm)
    for I in 1:n; u_rm2[5I-1] += 1.0; u_rm2[5I] -= 1.0; end
    for qp in 1:getnquadpoints(scv)
        @test norm(Ferrite.function_gradient(scv, qp, u_rm) -
                   Ferrite.function_gradient(scv, qp, u_rm2)) < 1e-14
    end

    # F = I + ∇u is identity at zero displacement: det(F) = 1, F = I₃
    u_zero5 = zeros(5n)
    for qp in 1:getnquadpoints(scv)
        ∇u = Ferrite.function_gradient(scv, qp, u_zero5)
        F  = one(∇u) + ∇u
        @test det(F) ≈ 1.0
        @test norm(F - one(Tensor{2,3})) < 1e-14
    end
end

@testset "compute_volume" begin
    corners = [Vec{2}((0.0,0.0)), Vec{2}((1.0,0.0)), Vec{2}((1.0,1.0)), Vec{2}((0.0,1.0))]
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (1,1), corners))
    ip   = Lagrange{RefQuadrilateral, 2}()
    qr   = QuadratureRule{RefQuadrilateral}(3)
    scv  = ShellCellValues(qr, ip, ip)
    dh   = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)

    # Helper: set u_z = f(x) for all nodes via cell iteration
    function set_uz!(u, dh, f)
        for cell in CellIterator(dh)
            cd = celldofs(cell); coords = getcoordinates(cell); n_c = length(coords)
            for I in 1:n_c; u[cd[3I]] = f(coords[I]); end
        end
    end

    # compute_volume returns −V_physical (volume_residual sign convention: return -val).
    # Physical enclosed volume = −compute_volume.

    # Zero displacement, explicit b_z=-0.1: enclosed height = 0.1 over unit area → physical = 0.1.
    @test compute_volume(dh, scv, zeros(ndofs(dh)); b=Vec((0.0,0.0,-0.1))) ≈ -0.1 atol=1e-10

    # Default b=(0,0,0): zero displacement at z=0 → zero volume.
    @test compute_volume(dh, scv, zeros(ndofs(dh))) ≈ 0.0 atol=1e-10

    # Uniform z-translation by Δz: F=I, physical volume = Δz → compute_volume = -Δz.
    Δz = 0.3
    u_inf = zeros(ndofs(dh))
    set_uz!(u_inf, dh, _ -> Δz)
    @test compute_volume(dh, scv, u_inf) ≈ -Δz atol=1e-10

    # Linearly varying u_z = α*x: ∫₀¹∫₀¹ α*x dx dy = α/2 → compute_volume = -α/2.
    α = 0.2
    u_shear = zeros(ndofs(dh))
    set_uz!(u_shear, dh, x -> α * x[1])
    @test compute_volume(dh, scv, u_shear) ≈ -α/2 atol=1e-8
end

@testset "compute_volume: closed surface (divergence theorem)" begin
    # Unit cube [0,1]³ as 6 Q4 shell faces with outward normals.
    # By the divergence theorem: ∮(z+0.1)n̂_z dA = ∫∫∫ 1 dV = V.
    # The `volume -= ...` convention with outward normals returns -V.
    #
    # Winding order chosen so G₃ = A₁×A₂ points outward for each face:
    #   top  (z=1): (0,0,1)→(1,0,1)→(1,1,1)→(0,1,1)  A₁=ê_x, A₂=ê_y → G₃=+ê_z ✓
    #   bot  (z=0): (0,0,0)→(0,1,0)→(1,1,0)→(1,0,0)  A₁=ê_y, A₂=ê_x → G₃=-ê_z ✓
    #   right(x=1): (1,0,0)→(1,1,0)→(1,1,1)→(1,0,1)  A₁=ê_y, A₂=ê_z → G₃=+ê_x ✓
    #   left (x=0): (0,0,0)→(0,0,1)→(0,1,1)→(0,1,0)  A₁=ê_z, A₂=ê_y → G₃=-ê_x ✓
    #   back (y=1): (0,1,0)→(0,1,1)→(1,1,1)→(1,1,0)  A₁=ê_z, A₂=ê_x → G₃=+ê_y ✓
    #   front(y=0): (0,0,0)→(1,0,0)→(1,0,1)→(0,0,1)  A₁=ê_x, A₂=ê_z → G₃=-ê_y ✓
    function make_cube_grid(L=1.0, W=1.0, H=1.0)
        nodes = Node.([
            Vec{3}((0.,0.,H)), Vec{3}((L,0.,H)), Vec{3}((L,W,H)), Vec{3}((0.,W,H)),
            Vec{3}((0.,0.,0.)), Vec{3}((0.,W,0.)), Vec{3}((L,W,0.)), Vec{3}((L,0.,0.)),
            Vec{3}((L,0.,0.)), Vec{3}((L,W,0.)), Vec{3}((L,W,H)), Vec{3}((L,0.,H)),
            Vec{3}((0.,0.,0.)), Vec{3}((0.,0.,H)), Vec{3}((0.,W,H)), Vec{3}((0.,W,0.)),
            Vec{3}((0.,W,0.)), Vec{3}((0.,W,H)), Vec{3}((L,W,H)), Vec{3}((L,W,0.)),
            Vec{3}((0.,0.,0.)), Vec{3}((L,0.,0.)), Vec{3}((L,0.,H)), Vec{3}((0.,0.,H)),
        ])
        cells = Quadrilateral.([(1,2,3,4),(5,6,7,8),(9,10,11,12),
                                 (13,14,15,16),(17,18,19,20),(21,22,23,24)])
        Grid(cells, nodes)
    end

    ip  = Lagrange{RefQuadrilateral, 1}()
    qr  = QuadratureRule{RefQuadrilateral}(2)
    scv = ShellCellValues(qr, ip, ip)

    for (L, W, H) in ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0))
        grid = make_cube_grid(L, W, H)
        dh   = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        @test compute_volume(dh, scv, zeros(ndofs(dh))) ≈ -L * W * H atol=1e-10
    end
end

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

@testset "director_field" begin
    # Flat Q9 plate in the x-y plane: G₃ = ê_z at all nodes.
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (2, 2),
                                    Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))
    ip  = Lagrange{RefQuadrilateral, 2}()
    qr  = QuadratureRule{RefQuadrilateral}(3)
    scv = ShellCellValues(qr, ip, ip)
    dh  = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    n_nodes = getnnodes(grid)

    # Zero displacement: director must equal G₃ everywhere, both unit vectors.
    d, G3 = director_field(dh, scv, zeros(ndofs(dh)))
    @test size(d)  == (3, n_nodes)
    @test size(G3) == (3, n_nodes)
    @test all(norm(G3[:, i]) ≈ 1.0 for i in 1:n_nodes)
    @test all(d[:, i] ≈ G3[:, i] for i in 1:n_nodes)   # no rotation → d = G₃
    @test all(G3[3, i] ≈ 1.0 for i in 1:n_nodes)        # flat plate → G₃ = ê_z

    # Known rotation: set φ₁ = angle for all nodes; director must rotate by angle about T₁.
    # For a flat x-y plate: T₁ ≈ ê_x, T₂ ≈ ê_y, G₃ = ê_z.
    # Rodrigues: d = cos(angle)·ê_z + sin(angle)·ê_x
    angle = π / 6
    u_rot = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        cd = celldofs(cell); n_loc = length(cell.nodes)
        for I in 1:n_loc; u_rot[cd[3n_loc + 2I-1]] = angle; end  # φ₁ = angle
    end
    d2, _ = director_field(dh, scv, u_rot)
    for i in 1:n_nodes
        @test norm(d2[:, i]) ≈ 1.0 atol=1e-12          # unit length (Rodrigues exact)
        @test d2[1, i] ≈ sin(angle) atol=1e-12          # T₁ component
        @test d2[3, i] ≈ cos(angle) atol=1e-12          # G₃ component
    end
end

@testset "assemble_traction! regression: two-field DofHandler" begin
    # Regression for the bug where assemble_traction! used the interleaved 5-DOF
    # block (5I-4:5I-2) for a two-field DofHandler, scattering force into θ-DOFs.
    # Fix: detect two-field layout and use the 3-DOF block (3I-2:3I) for u only.
    grid = shell_grid(generate_grid(Quadrilateral, (1, 1),
                                   Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0))))

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
        tangent_RM_FD!(ke3, scv3, u_e, mat3)
        assemble!(asmb3, shelldofs(cell), ke3, re3)
    end
    dbc3 = ConstraintHandler(dh3)
    add!(dbc3, Dirichlet(:u, getfacetset(grid2, "left"), x -> zeros(3), [1,2,3]))
    add!(dbc3, Dirichlet(:θ, getfacetset(grid2, "left"), x -> zeros(2), [1,2]))
    close!(dbc3); Ferrite.update!(dbc3, 0.0)
    apply!(K3, f3, dbc3)
    u3 = K3 \ f3

    ph3    = PointEvalHandler(grid2, [Vec{3}((4.0, 0.5, 0.0))])
    u_tip3 = first(evaluate_at_points(ph3, dh3, u3, :u))
    @test u_tip3[3] > 0.0   # positive z-deflection under +z traction
end

@testset "assemble_traction! regression: RefTriangle facets" begin
    # Regression for a bug where `facet_dxi` for RefTriangle hardcoded the wrong vertex
    # order and the wrong facet parametrization scale (t∈[-1,1], like RefQuadrilateral,
    # instead of Ferrite's actual t∈[0,1] for triangle facets). This silently under-
    # integrated boundary tractions on any Triangle/QuadraticTriangle mesh (Cook's
    # membrane tip deflection converged to ~76% of the correct value). Check that the
    # total applied force always equals traction × boundary length, regardless of how
    # the boundary is triangulated or which local facet number lands on the boundary.
    corners = [Vec{2}((0.0, 0.0)), Vec{2}((4.0, 1.0)), Vec{2}((4.0, 3.0)), Vec{2}((0.0, 2.0))]
    t_y = Vec{3}((0.0, 1.0, 0.0))   # right edge (from (4,1) to (4,3)) has length 2

    for (primitive, order, refshape) in ((Triangle, 1, RefTriangle), (QuadraticTriangle, 2, RefTriangle))
        for n in (1, 2, 3)
            grid = shell_grid(generate_grid(primitive, (2n, n), corners))
            addfacetset!(grid, "traction", x -> isapprox(x[1], 4.0, atol=1e-10))

            ip  = Lagrange{refshape, order}()
            fqr = FacetQuadratureRule{refshape}(order + 1)
            dh  = DofHandler(grid); add!(dh, :u, ip^3); close!(dh)
            f   = zeros(ndofs(dh))
            assemble_traction!(f, dh, getfacetset(grid, "traction"), ip, fqr, t_y)

            @test sum(f) ≈ 2.0 atol=1e-10
        end
    end
end

@testset "utils.jl" begin
    # test embeding
    Cαβ = SymmetricTensor{2,2}(rand(3))
    C = FerriteShells.embed23(Cαβ)
    @test all(C[1:2,1:2] .≈ Cαβ) # correct embedding for output
    # NodeFrames
    for (P,E,O) in zip([Quadrilateral, QuadraticQuadrilateral, Triangle, QuadraticTriangle],
                        [RefQuadrilateral, RefQuadrilateral, RefTriangle, RefTriangle],
                        [1,2,1,2])
        corners = [Vec{2}((0.0,0.0)), Vec{2}((1.0,0.0)), Vec{2}((1.0,1.0)), Vec{2}((0.0,1.0))]
        grid = shell_grid(generate_grid(P, (1,1), corners))
        ip   = Lagrange{E, O}()
        nf  = NodeFrames(grid, ip)
        @test all([all(Gᵢ .≈ nf.G₃[1]) for Gᵢ in nf.G₃])
        @test all([all(Gᵢ .≈ Vec{3}((0.0, 0.0, 1.0))) for Gᵢ in nf.G₃])
        @test all([all(T₁ .≈ nf.T₁[1]) for T₁ in nf.T₁])
        @test all([all(T₂ .≈ nf.T₂[1]) for T₂ in nf.T₂])
        # try reinit
        qr  = QuadratureRule{E}(O+1)
        scv = ShellCellValues(qr, ip, ip)
        dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
        cell = first(CellIterator(dh))
        reinit!(scv, cell, nf)
        node_ids = getnodes(cell)
        @test all([all(scv.G₃_elem[I] .≈ nf.G₃[node_ids[I]]) for I in 1:getnbasefunctions(scv.ip_geo)])
        @test all([all(scv.T₁_elem[I] .≈ nf.T₁[node_ids[I]]) for I in 1:getnbasefunctions(scv.ip_geo)])
        @test all([all(scv.T₂_elem[I] .≈ nf.T₂[node_ids[I]]) for I in 1:getnbasefunctions(scv.ip_geo)])
        # test the shell_strain output, should all be zero
        uₑ = zero(shelldofs(cell))
        @test all(shell_strains(scv, 1, uₑ) .≈ ([0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0], [0.0; 0.0]))
    end
end


@testset "dof lookup is layout independent" begin
    # Positional dof arithmetic (`:u` at `3I-2:3I`, `:θ` after it) holds only for the
    # canonical two-field order. These four layouts all describe the same shell; the
    # loaders must agree on every one of them. The oracle for "the `:u` dofs of these
    # nodes" is a ConstraintHandler, which resolves the field by name independently of
    # anything under test.
    ip   = Lagrange{RefQuadrilateral, 1}()
    fqr  = FacetQuadratureRule{RefQuadrilateral}(2)
    grid = shell_grid(generate_grid(Quadrilateral, (2, 2)))
    # every node, so the loaders are exercised at local indices beyond 1 — at I = 1 the
    # positional and by-name offsets coincide for every layout, hiding the defect.
    addnodeset!(grid, "all", x -> true)

    layouts = (
        ("two-field",   dh -> (add!(dh, :u, ip^3); add!(dh, :θ, ip^2))),
        ("θ first",     dh -> (add!(dh, :θ, ip^2); add!(dh, :u, ip^3))),
        ("extra field", dh -> (add!(dh, :p, ip); add!(dh, :u, ip^3); add!(dh, :θ, ip^2))),
        ("interleaved", dh -> add!(dh, :u, ip^5)),
    )

    u_dofs_of(dh, set) = begin
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, set, (x, t) -> zeros(3), 1:3))
        close!(ch)
        Set(ch.prescribed_dofs)
    end

    for (name, build!) in layouts
        dh = DofHandler(grid); build!(dh); close!(dh)

        f = zeros(ndofs(dh))
        apply_pointload!(f, dh, "all", Vec{3}((1.0, 2.0, 3.0)))
        @test Set(findall(!iszero, f)) == u_dofs_of(dh, getnodeset(grid, "all"))
        @test sort(f[findall(!iszero, f)]) == repeat([1.0, 2.0, 3.0], inner = getnnodes(grid))
        @test sum(f) ≈ 6.0 * getnnodes(grid)   # each node loaded exactly once

        f = zeros(ndofs(dh))
        fs = getfacetset(grid, "left")
        assemble_traction!(f, dh, fs, ip, fqr, Vec{3}((0.0, 0.0, 5.0)))
        @test issubset(Set(findall(!iszero, f)), u_dofs_of(dh, fs))
        @test sum(f) ≈ 5.0 * 2.0   # edge length 2, uniform pressure 5
    end

    # shelldofs: the positional form is right only for the canonical layout, the
    # SubDofHandler form for all of them.
    dh_ok = DofHandler(grid); add!(dh_ok, :u, ip^3); add!(dh_ok, :θ, ip^2); close!(dh_ok)
    dh_sw = DofHandler(grid); add!(dh_sw, :θ, ip^2); add!(dh_sw, :u, ip^3); close!(dh_sw)
    for (dh, canonical) in ((dh_ok, true), (dh_sw, false))
        sdh  = only(dh.subdofhandlers)
        cell = first(CellIterator(dh))
        sd   = shelldofs(sdh, cell)
        cd   = celldofs(cell)
        ru, rθ = Ferrite.dof_range(sdh, :u), Ferrite.dof_range(sdh, :θ)
        @test length(sd) == 20
        for I in 1:4
            @test sd[5I-4:5I-2] == cd[ru[3I-2:3I]]
            @test sd[5I-1:5I]   == cd[rθ[2I-1:2I]]
        end
        @test (shelldofs(cell) == sd) == canonical
        # in-place form: same answer, and allocation-free once warm
        buf = Int[]; shelldofs!(buf, sdh, cell)
        @test buf == sd
        @test (@allocated shelldofs!(buf, sdh, cell)) == 0
    end

    # a layout that is not 5 dofs/node is rejected instead of silently mis-permuted
    dh_bad = DofHandler(grid); add!(dh_bad, :u, ip^3); add!(dh_bad, :θ, ip^3); close!(dh_bad)
    @test_throws ArgumentError shelldofs(first(CellIterator(dh_bad)))
    @test_throws ArgumentError shelldofs!(Int[], only(dh_bad.subdofhandlers), first(CellIterator(dh_bad)))
end

@testset "reference director curvature B₀" begin
    # The bending measure is κ = ½(a_α·d,β + a_β·d,α) − B₀, with B₀ built from the
    # *interpolated initial director* d₀ = Σ N_I G₃_elem[I] — the field the kernels
    # rotate — not from the geometric patch curvature B = A_{α,β}·G₃. The two coincide
    # in the continuum but not discretely, so subtracting B leaves a reference bending
    # strain κ(0) = B₀ − B ≠ 0 on curved or warped elements: a pre-moment in the
    # undeformed configuration.
    ref9 = [Vec{2}((x, y)) for (x, y) in
        ((0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),(0.5,0.0),(1.0,0.5),(0.5,1.0),(0.0,0.5),(0.5,0.5))]
    curved9    = [Vec{3}((p[1], p[2], 0.3 * (p[1]^2 - p[2]^2 / 2))) for p in ref9]
    nonplanar4 = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.1)), Vec{3}((1.0,1.0,0.0)), Vec{3}((0.0,1.0,0.1))]
    mat = LinearElastic(1.0e6, 0.3, 0.01)

    cases = ((Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3), MITC9,   curved9),
             (Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3), nothing, curved9),
             (Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2), MITC4,   nonplanar4),
             (Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2), nothing, nonplanar4))
    for (ip, qr, mitc, x) in cases
        scv = ShellCellValues(qr, ip, ip; mitc)
        reinit!(scv, x)
        n_dof = 5 * getnbasefunctions(ip)
        u0 = zeros(n_dof)
        for qp in 1:getnquadpoints(scv)
            E, κ, _ = shell_strains(scv, qp, u0)
            @test norm(E) ≤ 1.0e-14
            @test norm(κ) ≤ 1.0e-13      # was ‖B₀ − B‖ ≠ 0 when B was subtracted
        end
        # ... and the user-visible consequence: no internal bending force at u = 0
        re = zeros(n_dof); bending_residuals_RM!(re, scv, u0, mat)
        @test norm(re) ≤ 1.0e-11      # was O(0.1) on these geometries
        ke = zeros(n_dof, n_dof); bending_tangent_RM!(ke, scv, u0, mat)
        @test norm(ke - ke') ≤ 1.0e-10 * norm(ke)
    end

    # The same holds through the NodeFrames entry point, on a mesh where the per-node
    # frames genuinely differ from each element's centroid frame.
    grid = shell_grid(generate_grid(Quadrilateral, (3, 3)); map = n -> (n.x[1], n.x[2], 0.25 * n.x[1]^2 - 0.15 * n.x[2]^2))
    ip   = Lagrange{RefQuadrilateral, 1}()
    nf   = NodeFrames(grid, ip)
    dh   = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    u0   = zeros(20)
    for mitc in (MITC4, nothing)
        scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip; mitc)
        for cell in CellIterator(dh)
            reinit!(scv, cell, nf)
            for qp in 1:getnquadpoints(scv)
                _, κ, _ = shell_strains(scv, qp, u0)
                @test norm(κ) ≤ 1.0e-13
            end
            re = zeros(20); bending_residuals_RM!(re, scv, u0, mat)
            @test norm(re) ≤ 1.0e-11
        end
    end

    # Flat elements with centroid frames are untouched: B₀ = B = 0 exactly.
    scv_flat = ShellCellValues(QuadratureRule{RefQuadrilateral}(3),
                               Lagrange{RefQuadrilateral,2}(), Lagrange{RefQuadrilateral,2}())
    reinit!(scv_flat, [Vec{3}((p[1], p[2], 0.0)) for p in ref9])
    @test all(iszero, scv_flat.B₀)
    @test all(iszero, scv_flat.B)
end

@testset "shell_strains shear reference" begin
    # The MITC tying strains already subtract their own per-tying-point reference, so
    # the interpolated γ is measured from the reference state; subtracting dot(A_α, d₀)
    # on top of that double-counts. Zero on flat elements (A_α ⟂ d₀), a spurious O(0.1)
    # reference shear on curved ones. The assembly kernels dispatch through
    # `reference_shear_offset`; shell_strains must do the same or it reports strains
    # that disagree with the ones the residual is built from.
    ip   = Lagrange{RefQuadrilateral, 2}()
    qr   = QuadratureRule{RefQuadrilateral}(3)
    grid = shell_grid(generate_grid(QuadraticQuadrilateral, (2, 2));
                      map = n -> (n.x[1], n.x[2], 0.25 * n.x[1]^2 - 0.15 * n.x[2]^2))
    dh = DofHandler(grid); add!(dh, :u, ip^3); add!(dh, :θ, ip^2); close!(dh)
    nf = NodeFrames(grid, ip)
    u0 = zeros(5 * getnbasefunctions(ip))
    for mitc in (MITC9, nothing), use_nf in (false, true)
        scv = ShellCellValues(qr, ip, ip; mitc)
        for cell in CellIterator(dh)
            use_nf ? reinit!(scv, cell, nf) : reinit!(scv, cell)
            for qp in 1:getnquadpoints(scv)
                _, _, γ = shell_strains(scv, qp, u0)
                @test norm(γ) ≤ 1.0e-13   # was ≈ 0.09 for MITC9 on this geometry
            end
        end
    end
end
