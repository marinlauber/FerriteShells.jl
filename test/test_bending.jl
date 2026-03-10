using FerriteShells
using LinearAlgebra
using Test
using Random

# Q9 (serendipity-style 9-node quad) gives full bending curvature via second derivatives.
function make_bending_scv(; qr_order=3)
    ip = Lagrange{RefQuadrilateral, 2}()
    qr = QuadratureRule{RefQuadrilateral}(qr_order)
    return ShellCellValues(qr, ip, ip)
end

# 9-node unit square in XY plane (Q9 node ordering matches Ferrite QuadraticQuadrilateral).
# Corner nodes first, then edge midpoints, then center.
const X_Q9_UNIT = [
    Vec{3}((0.0, 0.0, 0.0)),
    Vec{3}((1.0, 0.0, 0.0)),
    Vec{3}((1.0, 1.0, 0.0)),
    Vec{3}((0.0, 1.0, 0.0)),
    Vec{3}((0.5, 0.0, 0.0)),
    Vec{3}((1.0, 0.5, 0.0)),
    Vec{3}((0.5, 1.0, 0.0)),
    Vec{3}((0.0, 0.5, 0.0)),
    Vec{3}((0.5, 0.5, 0.0)),
]

# Bending residual helper
function bending_residual(scv, x, u_vec, mat)
    re = zeros(length(u_vec))
    bending_residuals_KL!(re, scv, x, u_vec, mat)
    return re
end

# Bending tangent helper
function bending_tangent_mat(scv, x, u_vec, mat)
    ke = zeros(length(u_vec), length(u_vec))
    bending_tangent_KL!(ke, scv, x, u_vec, mat)
    return ke
end

# Central-difference numerical tangent
function numerical_bending_tangent(scv, x, u_vec, mat; ε=1e-5)
    n = length(u_vec)
    Kfd = zeros(n, n)
    for j in 1:n
        up = copy(u_vec); up[j] += ε
        um = copy(u_vec); um[j] -= ε
        Kfd[:, j] = (bending_residual(scv, x, up, mat) .- bending_residual(scv, x, um, mat)) ./ (2ε)
    end
    return Kfd
end

@testset "KL bending" begin
    mat = LinearElastic(1.0e6, 0.3, 0.01)
    scv = make_bending_scv()
    reinit!(scv, X_Q9_UNIT)
    n_dof = 27  # 9 nodes × 3 DOFs

    @test bending_residual(scv, X_Q9_UNIT, zeros(n_dof), mat) ≈ zeros(n_dof) atol=1e-14

    # Rigid-body translation: zero curvature → zero residual
    u_trans = repeat([0.1, 0.2, 0.3], 9)
    @test norm(bending_residual(scv, X_Q9_UNIT, u_trans, mat)) ≤ 1e-10

    # Uniform in-plane stretch: flat surface stays flat → zero bending
    u_stretch = zeros(n_dof)
    for I in 1:9
        u_stretch[3I-2] = 0.01 * X_Q9_UNIT[I][1]
        u_stretch[3I-1] = 0.01 * X_Q9_UNIT[I][2]
    end
    @test norm(bending_residual(scv, X_Q9_UNIT, u_stretch, mat)) ≤ 1e-10

    # Tangent symmetry at zero displacement
    ke0 = bending_tangent_mat(scv, X_Q9_UNIT, zeros(n_dof), mat)
    @test norm(ke0 .- ke0') ≤ 1e-12 * norm(ke0)

    # Tangent symmetry at nonzero displacement
    Random.seed!(42)
    u_rnd = 0.02 * randn(n_dof)
    ke_rnd = bending_tangent_mat(scv, X_Q9_UNIT, u_rnd, mat)
    @test norm(ke_rnd .- ke_rnd') / norm(ke_rnd) ≤ 1e-10

    # Finite-difference tangent consistency (small displacement)
    Random.seed!(7)
    u_small = 0.001 * randn(n_dof)
    ke_an = bending_tangent_mat(scv, X_Q9_UNIT, u_small, mat)
    ke_fd = numerical_bending_tangent(scv, X_Q9_UNIT, u_small, mat)
    @test norm(ke_an .- ke_fd) / (norm(ke_fd) + 1e-14) < 1e-6

    # Finite-difference tangent consistency (moderate out-of-plane displacement)
    Random.seed!(8)
    u_mod = zeros(n_dof)
    for I in 1:9
        u_mod[3I] = 0.05 * randn()  # z-displacements induce curvature
    end
    ke_an2 = bending_tangent_mat(scv, X_Q9_UNIT, u_mod, mat)
    ke_fd2 = numerical_bending_tangent(scv, X_Q9_UNIT, u_mod, mat)
    @test norm(ke_an2 .- ke_fd2) / (norm(ke_fd2) + 1e-14) < 1e-6

    # Positive semi-definiteness at zero displacement (geometric stiffness vanishes → only material term)
    λs  = eigvals(Symmetric(ke0))
    tol = 1e-8 * maximum(abs, λs)
    @test minimum(λs) ≥ -tol

    # Pure twist: Q4 gives non-zero κ₁₂, Q9 also captures κ₁₁/κ₂₂ from ∂²N terms.
    # A parabolic z-field z = x*(1-x) satisfies zero BCs at x=0,1.
    # The curvature κ₁₁ = ∂²z/∂x² ≠ 0, so bending energy must be positive.
    u_curve = zeros(n_dof)
    for I in 1:9
        xI = X_Q9_UNIT[I][1]
        u_curve[3I] = xI * (1.0 - xI)
    end
    W = FerriteShells.bending_energy_KL(u_curve, scv, X_Q9_UNIT, mat)
    @test W > 0.0

    # Zero-energy mode spectrum at u=0.
    # In-plane DOFs (9 nodes × 2 = 18) + z rigid-body modes (Tz, Rx, Ry = 3) → 21 zero modes.
    # Remaining quadratic z-DOF modes → 6 positive modes.
    λs_all = eigvals(Symmetric(ke0))
    tol_modes = 1e-8 * maximum(abs, λs_all)
    n_zero     = count(λ -> abs(λ) ≤ tol_modes, λs_all)
    n_positive = count(λ ->     λ  > tol_modes, λs_all)
    @test n_zero     == 21
    @test n_positive == 6
    @test n_zero + n_positive == n_dof

    # Rigid-body rotation invariance: bending energy is frame-indifferent.
    # Test with in-plane rotation (about z) and out-of-plane tilt (about x).
    rotZ(θ) = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
    rotX(θ) = [1.0 0.0 0.0; 0.0 cos(θ) -sin(θ); 0.0 sin(θ) cos(θ)]
    for R in (rotZ(π/5), rotX(π/4))
        x_rot = [Vec{3}(Tuple(R * collect(xi))) for xi in X_Q9_UNIT]
        scv_rot = make_bending_scv()
        reinit!(scv_rot, x_rot)
        u_rot = zeros(n_dof)
        for I in 1:9
            u_rot[3I-2:3I] = R * u_curve[3I-2:3I]
        end
        W_rot = FerriteShells.bending_energy_KL(u_rot, scv_rot, x_rot, mat)
        @test W_rot ≈ W rtol=1e-8
    end

    # Analytical energy check: for ε·u_curve with ε → 0, linearized KL gives
    # W ≈ 0.5 · D¹¹¹¹ · κ₁₁² · Area where κ₁₁ = ∂²(ε·x(1-x))/∂x² = -2ε.
    # D¹¹¹¹ = E·t³/(12(1-ν²)) for a flat plate with identity metric.
    # Relative error is O(ε²) since the nonlinear correction to the normal is O(ε²).
    let ε = 1e-3
        D11 = mat.E * mat.thickness^3 / (12 * (1 - mat.ν^2))
        W_lin_an = 0.5 * D11 * 4.0 * ε^2   # κ₁₁ = -2ε, Area = 1
        W_lin_fe = FerriteShells.bending_energy_KL(ε * u_curve, scv, X_Q9_UNIT, mat)
        @test W_lin_fe ≈ W_lin_an rtol=1e-4
    end

    # Non-flat reference geometry: cylindrical arc wrapped from the unit square.
    # R=5 gives mild curvature so Q9 integration is accurate.
    let R = 5.0
        ref_pos = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0),
                   (0.5,0.0),(1.0,0.5),(0.5,1.0),(0.0,0.5),(0.5,0.5)]
        X_cyl = [Vec{3}((R*sin(s/R), t, R*(1.0 - cos(s/R)))) for (s,t) in ref_pos]
        scv_cyl = make_bending_scv()
        reinit!(scv_cyl, X_cyl)
        # Reference state: κ = b - B = 0 → zero residual
        @test norm(bending_residual(scv_cyl, X_cyl, zeros(n_dof), mat)) ≤ 1e-12
        # FD tangent consistency on curved geometry
        Random.seed!(99)
        u_cyl = zeros(n_dof); u_cyl[3:3:end] .= 0.01 * randn(9)
        ke_cyl    = bending_tangent_mat(scv_cyl, X_cyl, u_cyl, mat)
        ke_cyl_fd = numerical_bending_tangent(scv_cyl, X_cyl, u_cyl, mat)
        @test norm(ke_cyl .- ke_cyl_fd) / (norm(ke_cyl_fd) + 1e-14) < 1e-6
    end

    # Combined membrane + bending tangent: symmetry and FD consistency.
    combined_re(u) = (re = zeros(length(u)); membrane_residuals_KL!(re, scv, X_Q9_UNIT, u, mat); bending_residuals_KL!(re, scv, X_Q9_UNIT, u, mat); re)
    @test norm(combined_re(zeros(n_dof))) ≤ 1e-14
    ke_comb = zeros(n_dof, n_dof)
    membrane_tangent_KL!(ke_comb, scv, X_Q9_UNIT, u_curve, mat)
    bending_tangent_KL!(ke_comb, scv, X_Q9_UNIT, u_curve, mat)
    @test norm(ke_comb .- ke_comb') / norm(ke_comb) ≤ 1e-10
    ke_comb_fd = zeros(n_dof, n_dof)
    let ε = 1e-5
        for j in 1:n_dof
            up = copy(u_curve); up[j] += ε
            um = copy(u_curve); um[j] -= ε
            ke_comb_fd[:, j] = (combined_re(up) .- combined_re(um)) ./ (2ε)
        end
    end
    @test norm(ke_comb .- ke_comb_fd) / (norm(ke_comb_fd) + 1e-14) < 1e-6
end

@testset "Bending energy h-convergence" begin
    # Manufactured solution: w(x,y) = ε·sin(πx)·sin(πy) on [0,1]².
    # Curvatures: κ₁₁ = κ₂₂ = -π²·sin(πx)·sin(πy), κ₁₂ = -π²·cos(πx)·cos(πy).
    # Each of ∫κ₁₁², ∫κ₂₂², ∫κ₁₁κ₂₂, ∫κ₁₂² over [0,1]² equals π⁴/4.
    # Summing: W_an = ε² · (D¹¹¹¹ + 2D¹¹²² + D²²²² + 4D¹²¹²) · π⁴/8 = ε² · D₁₁ · π⁴/2
    # where D₁₁ = E·t³/(12(1-ν²)). Nonlinear correction is O(ε⁴), negligible for ε = 0.01.
    mat = LinearElastic(1e6, 0.3, 0.01)
    ε   = 0.01
    D11 = mat.E * mat.thickness^3 / (12 * (1 - mat.ν^2))
    W_an = ε^2 * D11 * π^4 / 2

    # Q9 node layout for a rectangular element [x0,x1]×[y0,y1]:
    # corners (CCW), edge midpoints (CCW from bottom), centre — matching Ferrite ordering.
    function q9_coords(x0, x1, y0, y1)
        xm, ym = (x0+x1)/2, (y0+y1)/2
        xs = (x0, x1, x1, x0, xm, x1, xm, x0, xm)
        ys = (y0, y0, y1, y1, y0, ym, y1, ym, ym)
        return [Vec{3}((xs[k], ys[k], 0.0)) for k in 1:9], xs, ys
    end

    scv_h = make_bending_scv()
    errors = Float64[]
    for N in [2, 4, 8, 16]
        W_FE = 0.0
        for j in 0:N-1, i in 0:N-1
            x_el, xs, ys = q9_coords(i/N, (i+1)/N, j/N, (j+1)/N)
            reinit!(scv_h, x_el)
            u_el = zeros(27)
            for k in 1:9
                u_el[3k] = ε * sin(π * xs[k]) * sin(π * ys[k])
            end
            W_FE += FerriteShells.bending_energy_KL(u_el, scv_h, x_el, mat)
        end
        push!(errors, abs(W_FE - W_an))
    end

    rates = [log2(errors[i] / errors[i+1]) for i in 1:length(errors)-1]
    @test all(r -> r ≥ 1.5, rates)
end
