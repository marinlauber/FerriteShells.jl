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
