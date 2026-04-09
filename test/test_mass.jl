using FerriteShells
using LinearAlgebra
using Test

# Q9 unit square in XY plane — same setup as test_rm.jl
const X_Q9_MASS = [
    Vec{3}((0.0, 0.0, 0.0)), Vec{3}((1.0, 0.0, 0.0)), Vec{3}((1.0, 1.0, 0.0)),
    Vec{3}((0.0, 1.0, 0.0)), Vec{3}((0.5, 0.0, 0.0)), Vec{3}((1.0, 0.5, 0.0)),
    Vec{3}((0.5, 1.0, 0.0)), Vec{3}((0.0, 0.5, 0.0)), Vec{3}((0.5, 0.5, 0.0)),
]

function make_mass_scv()
    ip = Lagrange{RefQuadrilateral, 2}()
    qr = QuadratureRule{RefQuadrilateral}(3)
    ShellCellValues(qr, ip, ip)
end

function make_me(ρ, mat; coords=X_Q9_MASS)
    scv = make_mass_scv()
    reinit!(scv, coords)
    me = zeros(45, 45)
    mass_matrix!(me, scv, ρ, mat)
    return me
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
    X_2x2 = [2v for v in X_Q9_MASS]
    me_scaled = make_me(ρ, mat; coords=X_2x2)
    @test sum(me_scaled[x_dofs, x_dofs]) ≈ 4 * expected_mass  rtol=1e-10
end
