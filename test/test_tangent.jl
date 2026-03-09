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
    membrane_residuals!(re, scv, x, u_vec, mat)
    return re
end

# Helper: compute tangent into a fresh zeroed matrix
function tangent(scv, x, u_vec, mat)
    ke = zeros(length(u_vec), length(u_vec))
    membrane_tangent!(ke, scv, x, u_vec, mat)
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

@testset "Membrane tangent" begin

    mat = LinearElastic(1.0, 0.3, 0.1)
    scv = make_scv(qr_order=1)
    reinit!(scv, X_UNIT_SQUARE)
    n_dof = 12  # 4 nodes × 3 DOFs

    @testset "Zero displacement: residual vanishes" begin
        re = residual(scv, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(re) ≤ 1e-14
    end

    @testset "Rigid-body translation: residual vanishes" begin
        # Uniform shift in x-direction — pure translation, zero strain
        u_vec = repeat([0.5, 0.0, 0.0], 4)
        re = residual(scv, X_UNIT_SQUARE, u_vec, mat)
        @test norm(re) ≤ 1e-13
    end

    @testset "Tangent symmetry (zero displacement)" begin
        ke = tangent(scv, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(ke .- ke') ≤ 1e-14 * norm(ke)
    end

    @testset "Tangent symmetry (nonzero displacement)" begin
        Random.seed!(1)
        u_vec = 0.05 * randn(n_dof)
        ke = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        @test norm(ke .- ke') / norm(ke) ≤ 1e-12
    end

    @testset "FD consistency (small displacement)" begin
        # Small displacement: geometric stiffness is negligible; material part dominates.
        Random.seed!(2)
        u_vec = 0.01 * randn(n_dof)
        ke    = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        ke_fd = numerical_tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7
    end

    @testset "FD consistency (moderate displacement)" begin
        # Moderate displacement: geometric stiffness is non-negligible.
        Random.seed!(3)
        u_vec = 0.3 * randn(n_dof)
        ke    = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        ke_fd = numerical_tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        rel_err = norm(ke .- ke_fd) / (norm(ke_fd) + 1e-14)
        @test rel_err < 1e-7
    end

    @testset "Positive semi-definiteness at small displacement" begin
        # K should have no significantly negative eigenvalues.
        # For a free element, rigid-body modes yield zero eigenvalues.
        Random.seed!(4)
        u_vec = 0.01 * randn(n_dof)
        ke = tangent(scv, X_UNIT_SQUARE, u_vec, mat)
        λ  = eigvals(Symmetric(ke))
        @test minimum(λ) ≥ -1e-10 * maximum(abs, λ)
    end

    @testset "Multi-point quadrature gives same zero residual" begin
        # Previously, loops used length(scv.N) = n_qp*n_nodes instead of n_nodes.
        # Now that getnbasefunctions is used, higher-order rules work correctly.
        scv2 = make_scv(qr_order=2)
        reinit!(scv2, X_UNIT_SQUARE)
        re2 = residual(scv2, X_UNIT_SQUARE, zeros(n_dof), mat)
        @test norm(re2) ≤ 1e-14
    end

    @testset "Zero-energy mode spectrum" begin
        # Full 2×2 Gauss integration — required to suppress hourglass modes.
        # With 1-point (reduced) integration, Q4 gains 3 spurious hourglass modes,
        # increasing the zero-eigenvalue count from 7 to 10.
        scv2 = make_scv(qr_order=2)
        reinit!(scv2, X_UNIT_SQUARE)
        ke = tangent(scv2, X_UNIT_SQUARE, zeros(n_dof), mat)

        λs  = eigvals(Symmetric(ke))
        tol = 1e-10 * maximum(abs, λs)

        n_zero     = count(λ ->  abs(λ) ≤ tol, λs)
        n_positive = count(λ ->      λ  > tol, λs)

        # 4-node Q4 membrane, 4 nodes × 3 DOFs = 12 DOFs:
        #   4 zero eigenvalues — z-DOFs carry no membrane stiffness
        #   3 zero eigenvalues — in-plane rigid body modes: Tx, Ty, Rz
        #   5 positive eigenvalues — independent in-plane deformation modes
        #
        # n_zero + n_positive == n_dof implies no negative eigenvalues.
        @test n_zero     == 7
        @test n_positive == 5
        @test n_zero + n_positive == n_dof
    end
end
