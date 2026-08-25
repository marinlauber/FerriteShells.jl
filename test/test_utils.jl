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


@testset "shell_strains: zero strain at u = 0 on curved MITC elements" begin
    # The MITC tying strains carry their own reference subtraction; subtracting
    # the QP-direct Aα·d₀ again produced a spurious reference shear (γ ≈ 0.11)
    # on curved elements. All three strain measures must vanish identically in
    # the undeformed configuration, for every element/MITC combination.
    ref9 = [Vec{2}((x, y)) for (x, y) in
        ((0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),(0.5,0.0),(1.0,0.5),(0.5,1.0),(0.0,0.5),(0.5,0.5))]
    curved9 = [Vec{3}((p[1], p[2], 0.3 * (p[1]^2 - p[2]^2 / 2))) for p in ref9]
    nonplanar4 = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.1)), Vec{3}((1.0,1.0,0.0)), Vec{3}((0.0,1.0,0.1))]
    cases = (
        (Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3), MITC9, curved9),
        (Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3), nothing, curved9),
        (Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2), MITC4, nonplanar4),
        (Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2), nothing, nonplanar4),
    )
    for (ip, qr, mitc, x) in cases
        scv = ShellCellValues(qr, ip, ip; mitc)
        reinit!(scv, x)
        u0 = zeros(5 * getnbasefunctions(ip))
        for qp in 1:getnquadpoints(scv)
            E, κ, γ = shell_strains(scv, qp, u0)
            @test norm(E) ≤ 1.0e-14
            @test norm(γ) ≤ 1.0e-14
            # κ is measured against the reference director gradient B₀
            # (the flip condition of the formulation question this testset
            # used to pin): the reference configuration of ANY element —
            # curved, warped or flat, whatever the frame choice — is free of
            # bending strain. Subtracting the patch curvature B instead left
            # κ(0) = −B, a reference pre-moment whose twist part persists
            # under refinement on bilinear panels of doubly-curved surfaces.
            @test norm(κ) ≤ 1.0e-13
        end
    end
end

@testset "director_field uses the kernel frame" begin
    # The element kernels rotate about G₃_elem/T₁_elem/T₂_elem; the exported
    # director_field must reproduce exactly that rotation. On a skewed element
    # the quadrature-point geometric frame (A₁/‖A₁‖) is a different frame and
    # was measured to tilt the reported director by ~7°.
    nodes3 = [Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.4,0.0)), Vec{3}((1.3,1.2,0.0)), Vec{3}((0.2,1.0,0.0))]
    grid = Grid([Quadrilateral((1,2,3,4))], Node.(nodes3))
    ip = Lagrange{RefQuadrilateral,1}()
    scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    add!(dh, :θ, ip^2)
    close!(dh)
    φ = (0.3, 0.1)
    u = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        sd = shelldofs(cell)
        for I in 1:4
            u[sd[5I-1]] = φ[1]
            u[sd[5I]] = φ[2]
        end
    end
    d, G3 = director_field(dh, scv, u)
    # Expected: the kernel's own Rodrigues rotation about the element frame.
    reinit!(scv, nodes3)
    θ = sqrt(φ[1]^2 + φ[2]^2)
    for I in 1:4
        d_exp = cos(θ) * scv.G₃_elem[I] + sin(θ)/θ * (φ[1] * scv.T₁_elem[I] + φ[2] * scv.T₂_elem[I])
        @test maximum(abs, Vec{3}(ntuple(r -> d[r, I], 3)) - d_exp) ≤ 1.0e-14
        @test Vec{3}(ntuple(r -> G3[r, I], 3)) ≈ scv.G₃_elem[I]
    end
    # With NodeFrames the same statement holds per node frame.
    grid2 = shell_grid(generate_grid(Quadrilateral, (2, 1)); map = n -> (n.x[1], n.x[2], 0.2 * n.x[1]^2))
    nf = NodeFrames(grid2, ip)
    scv2 = ShellCellValues(QuadratureRule{RefQuadrilateral}(2), ip, ip)
    dh2 = DofHandler(grid2)
    add!(dh2, :u, ip^3)
    add!(dh2, :θ, ip^2)
    close!(dh2)
    u2 = zeros(ndofs(dh2))
    for cell in CellIterator(dh2)
        sd = shelldofs(cell)
        for I in 1:4
            u2[sd[5I-1]] = φ[1]
            u2[sd[5I]] = φ[2]
        end
    end
    d2, _ = director_field(dh2, scv2, u2; frames = nf)
    for nid in 1:Ferrite.getnnodes(grid2)
        d_exp = cos(θ) * nf.G₃[nid] + sin(θ)/θ * (φ[1] * nf.T₁[nid] + φ[2] * nf.T₂[nid])
        @test maximum(abs, Vec{3}(ntuple(r -> d2[r, nid], 3)) - d_exp) ≤ 1.0e-13
    end
end

@testset "shelldofs hardening and shelldofs!" begin
    grid = shell_grid(generate_grid(Quadrilateral, (2, 2)))
    ip = Lagrange{RefQuadrilateral,1}()
    # Two-field layout: shelldofs! reproduces shelldofs exactly.
    dh = DofHandler(grid)
    add!(dh, :u, ip^3)
    add!(dh, :θ, ip^2)
    close!(dh)
    sd = Int[]
    for cell in CellIterator(dh)
        @test shelldofs!(sd, only(dh.subdofhandlers), cell) == shelldofs(cell)
    end
    # A third field: shelldofs throws instead of returning uninitialized
    # memory; shelldofs! keeps working, and its :u/:θ dofs match dof_range.
    dh3 = DofHandler(grid)
    add!(dh3, :u, ip^3)
    add!(dh3, :θ, ip^2)
    add!(dh3, :p, ip)
    close!(dh3)
    sdh3 = only(dh3.subdofhandlers)
    ru, rθ = Ferrite.dof_range(sdh3, :u), Ferrite.dof_range(sdh3, :θ)
    for cell in CellIterator(dh3)
        @test_throws ArgumentError shelldofs(cell)
        shelldofs!(sd, sdh3, cell)
        cd = celldofs(cell)
        for I in 1:4
            @test sd[5I-4:5I-2] == cd[ru[3I-2:3I]]
            @test sd[5I-1:5I] == cd[rθ[2I-1:2I]]
        end
    end
    # Field order reversed: the by-name lookup stays correct where positional
    # assumptions would scatter into :θ.
    dhr = DofHandler(grid)
    add!(dhr, :θ, ip^2)
    add!(dhr, :u, ip^3)
    close!(dhr)
    f = zeros(ndofs(dhr))
    addnodeset!(grid, "corner_pl", x -> norm(x) < 1e-10)
    apply_pointload!(f, dhr, "corner_pl", Vec{3}((1.0, 2.0, 3.0)))
    sdhr = only(dhr.subdofhandlers)
    rθr = Ferrite.dof_range(sdhr, :θ)
    for cell in CellIterator(dhr)
        cd = celldofs(cell)
        @test all(iszero, f[cd[rθr]])   # nothing leaked into the rotations
    end
    @test sum(abs, f) ≈ 6.0             # exactly one node loaded with (1,2,3)
end

@testset "ShellCellValues carries frames" begin
    # Folded two-quad strip: the shared-edge node frames (area-weighted averages)
    # differ from either element's centroid frame, so frame routing is observable.
    nodes = [Node(Vec((0.0, 0.0, 0.0))), Node(Vec((1.0, 0.0, 0.0))),
             Node(Vec((1.0, 1.0, 0.0))), Node(Vec((0.0, 1.0, 0.0))),
             Node(Vec((2.0, 0.0, 0.5))), Node(Vec((2.0, 1.0, 0.5)))]
    cells = [Quadrilateral((1, 2, 3, 4)), Quadrilateral((2, 5, 6, 3))]
    grid  = Grid(cells, nodes)
    ip    = Lagrange{RefQuadrilateral, 1}()
    nf    = NodeFrames(grid, ip)
    qr    = QuadratureRule{RefQuadrilateral}(2)
    scv_plain  = ShellCellValues(qr, ip, ip)
    scv_frames = ShellCellValues(qr, ip, ip; frames = nf)
    @test scv_frames.frames === nf

    dh = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2)
    close!(dh)
    for cell in CellIterator(dh)
        reinit!(scv_frames, cell)          # stored frames applied automatically
        G3_auto = copy(scv_frames.G₃_elem)
        reinit!(scv_plain, cell, nf)       # explicit-frames path
        @test G3_auto == scv_plain.G₃_elem
        reinit!(scv_plain, cell)           # frameless: centroid frame
        @test G3_auto != scv_plain.G₃_elem # differs at the shared fold edge
    end
end

@testset "copy(::ShellCellValues) independence" begin
    scv = ShellCellValues(QuadratureRule{RefQuadrilateral}(3),
                          Lagrange{RefQuadrilateral, 2}(), Lagrange{RefQuadrilateral, 2}();
                          mitc = MITC9)
    reinit!(scv, X_Q9_UNIT)
    scv2 = copy(scv)
    @test scv2.detJdV == scv.detJdV && scv2.G₃ == scv.G₃
    @test scv2.mitc.A₁_tie_1 == scv.mitc.A₁_tie_1
    @test scv2.qr === scv.qr && scv2.N !== scv.N   # shares immutables, owns buffers

    # reinit! on the copy must not touch the original (incl. the MITC tie data)
    orig_detJ = copy(scv.detJdV)
    orig_tie  = copy(scv.mitc.A₁_tie_1)
    reinit!(scv2, [2v for v in X_Q9_UNIT])
    @test scv.detJdV == orig_detJ && scv.mitc.A₁_tie_1 == orig_tie
    @test scv2.detJdV ≈ 4 .* orig_detJ

    @test copy(NoMITC()) isa NoMITC
end

@testset "max_director_tilt" begin
    grid = generate_grid(Quadrilateral, (2, 1))
    ip   = Lagrange{RefQuadrilateral, 1}()
    dh   = DofHandler(grid)
    add!(dh, :u, ip^3); add!(dh, :θ, ip^2)
    close!(dh)
    u = zeros(ndofs(dh))
    @test max_director_tilt(dh, u) == 0.0
    sdh  = only(dh.subdofhandlers)
    rθ   = Ferrite.dof_range(sdh, :θ)
    dofs = celldofs(dh, 1)
    u[dofs[rθ[1]]] = 0.3
    u[dofs[rθ[2]]] = 0.4
    @test max_director_tilt(dh, u) ≈ 0.5
end
