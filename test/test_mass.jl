function make_me(ρ, mat; coords=X_Q9_UNIT)
    scv = make_q9_scv()
    reinit!(scv, coords)
    me = zeros(45, 45)
    mass_matrix!(me, scv, ρ, mat)
    me
end

@testset "mass_matrix!" begin
    ρ = 800.0
    mat = LinearElastic(0.35e6, 0.3, 0.002)
    me = make_me(ρ, mat)

    # symmetry
    @test me ≈ me'

    # rotational DOFs (indices 4,5 per node) must be zero — no rotational inertia
    rot_dofs = vcat([5I-1:5I for I in 1:9]...)
    @test all(me[rot_dofs, :] .== 0.0)
    @test all(me[:, rot_dofs] .== 0.0)

    # translational diagonal entries must all be positive
    trans_dofs = vcat([5I-4:5I-2 for I in 1:9]...)
    @test all(diag(me)[trans_dofs] .> 0.0)

    # total mass: ∑_{IJ} M_{IJ,aa} = ρ*t*A (partition of unity, A=1 for unit square)
    expected_mass = ρ * mat.thickness * 1.0
    x_dofs = 1:5:45
    @test sum(me[x_dofs, x_dofs]) ≈ expected_mass  rtol=1e-10
    @test sum(me[x_dofs.+1, x_dofs.+1]) ≈ expected_mass  rtol=1e-10  # y-direction
    @test sum(me[x_dofs.+2, x_dofs.+2]) ≈ expected_mass  rtol=1e-10  # z-direction

    # rigid body test: M * v_rigid = momentum vector, total momentum = ρ*t*A
    v_rigid = zeros(45); v_rigid[1:5:45] .= 1.0    # unit x-velocity at all nodes
    @test sum(me * v_rigid) ≈ expected_mass  rtol=1e-10

    # isotropy: all three translational blocks are identical
    @test me[x_dofs, x_dofs] ≈ me[x_dofs.+1, x_dofs.+1]
    @test me[x_dofs, x_dofs] ≈ me[x_dofs.+2, x_dofs.+2]

    # positive semidefinite (consistent mass matrix is PD on translational DOFs)
    @test minimum(eigvals(Symmetric(me[trans_dofs, trans_dofs]))) ≥ -1e-14

    # linear scaling in ρ
    me2 = make_me(2ρ, mat)
    @test me2 ≈ 2 .* me

    # linear scaling in thickness (via a new material)
    mat2 = LinearElastic(0.35e6, 0.3, 2 * mat.thickness)
    me3 = make_me(ρ, mat2)
    @test me3 ≈ 2 .* me

    # 2×2 scaled element (A=4): total mass should be 4× the unit square
    X_2x2 = [2v for v in X_Q9_UNIT]
    me_scaled = make_me(ρ, mat; coords=X_2x2)
    @test sum(me_scaled[x_dofs, x_dofs]) ≈ 4 * expected_mass  rtol=1e-10
end

@testset "lumped_mass!" begin
    ρ   = 800.0
    mat = LinearElastic(0.35e6, 0.3, 0.002)
    scv = make_q9_scv()
    reinit!(scv, X_Q9_UNIT)
    trans = vcat([5I-4:5I-2 for I in 1:9]...)
    rot   = vcat([5I-1:5I   for I in 1:9]...)

    m = zeros(45)
    lumped_mass!(m, scv, ρ, mat)   # rotary = :floor

    # mass conserved per translational component (unit square, A = 1)
    expected_mass = ρ * mat.thickness * 1.0
    x_dofs = 1:5:45
    @test sum(m[x_dofs])      ≈ expected_mass  rtol=1e-12
    @test sum(m[x_dofs .+ 1]) ≈ expected_mass  rtol=1e-12
    @test sum(m[x_dofs .+ 2]) ≈ expected_mass  rtol=1e-12

    # HRZ positivity — everywhere, including the T6 corners where row-sum lumping vanishes
    @test all(m[trans] .> 0)
    scv_t6 = make_t6_scv()
    reinit!(scv_t6, X_T6_UNIT)
    m_t6 = zeros(30)
    lumped_mass!(m_t6, scv_t6, ρ, mat)
    @test all(m_t6[vcat([5I-4:5I-2 for I in 1:6]...)] .> 0)
    @test sum(m_t6[1:5:30]) ≈ ρ * mat.thickness * 0.5  rtol=1e-12
    # ... and row summing the consistent T6 matrix really does vanish at corners (why HRZ)
    me_t6 = zeros(30, 30)
    mass_matrix!(me_t6, scv_t6, ρ, mat)
    @test minimum(sum(me_t6, dims=2)[1:5:30]) < 1e-12 * expected_mass

    # rotary options: exact per-node relations (h = √A = 1 ≫ t on the unit square)
    @test all(m[rot] .> 0)
    m_cons = zeros(45); lumped_mass!(m_cons, scv, ρ, mat; rotary=:consistent)
    m_zero = zeros(45); lumped_mass!(m_zero, scv, ρ, mat; rotary=:zero)
    for I in 1:9
        @test m[5I-1] == m[5I] ≈ m[5I-4] * 1.0 / 12               # floor: max(t, 1)²/12
        @test m_cons[5I-1] ≈ m_cons[5I-4] * mat.thickness^2 / 12  # thin-plate share
        @test m_zero[5I-1] == m_zero[5I] == 0.0
    end
    @test m_zero[trans] == m[trans] == m_cons[trans]

    # scaling: linear in ρ; thickness → translational ×2, consistent rotary ×8
    m2 = zeros(45); lumped_mass!(m2, scv, 2ρ, mat)
    @test m2 ≈ 2m
    mat2 = LinearElastic(0.35e6, 0.3, 2 * mat.thickness)
    m3 = zeros(45); lumped_mass!(m3, scv, ρ, mat2; rotary=:consistent)
    @test m3[trans] ≈ 2 .* m_cons[trans]
    @test m3[rot]   ≈ 8 .* m_cons[rot]

    # accumulates (+= semantics) and validates its inputs
    m4 = copy(m); lumped_mass!(m4, scv, ρ, mat)
    @test m4 ≈ 2m
    @test_throws ArgumentError lumped_mass!(zeros(44), scv, ρ, mat)
    @test_throws ArgumentError lumped_mass!(zeros(45), scv, ρ, mat; rotary=:hrz)

    # thickness accessor is the material contract the mass kernels use
    @test thickness(mat) == mat.thickness
end
