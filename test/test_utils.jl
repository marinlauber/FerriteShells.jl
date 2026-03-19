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

    # Zero displacement: shell at z=0, base at b_z=-0.1 → enclosed height = 0.1 over unit area.
    # The formula uses `volume -= ...` with outward normal +ê_z, so the result is -0.1.
    @test compute_volume(dh, scv, zeros(ndofs(dh))) ≈ -0.1 atol=1e-10

    # Uniform z-translation by Δz: F = I (∇u = 0 for constant u), volume = -(0.1 + Δz).
    Δz = 0.3
    u_inf = zeros(ndofs(dh))
    set_uz!(u_inf, dh, _ -> Δz)
    @test compute_volume(dh, scv, u_inf) ≈ -(0.1 + Δz) atol=1e-10

    # Linearly varying u_z = α*x: F has a non-trivial deformation gradient.
    # Via the divergence theorem: ∫₀¹∫₀¹ (α*x + 0.1) dx dy = α/2 + 0.1.
    # F⁻ᵀ·n Nanson factor: det(F) * F⁻ᵀ*ê_z correctly reduces to just (α*x+0.1) * ê_z.
    α = 0.2
    u_shear = zeros(ndofs(dh))
    set_uz!(u_shear, dh, x -> α * x[1])
    @test compute_volume(dh, scv, u_shear) ≈ -(α/2 + 0.1) atol=1e-8
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
        @test compute_volume(dh, scv, zeros(ndofs(dh))) ≈ -(L * W * H) atol=1e-10
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
        membrane_tangent_RM!(ke3, scv3, u_e, mat3)
        bending_tangent_RM!(ke3, scv3, u_e, mat3)
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
