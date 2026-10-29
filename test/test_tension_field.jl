using FerriteShells, Test, LinearAlgebra

@testset "tension field" begin
    E, ν, t = 1.0e6, 0.3, 1.0e-3
    mat0 = LinearElastic(E, ν, t)
    matw = LinearElastic(E, ν, t; tension_field=true, ε_tf=0.0)

    # flat orthonormal reference frame ⇒ physical transform P = I, so eigvals(N)
    # are directly the principal membrane stresses.
    A₁ = Vec{3}((1.0, 0.0, 0.0)); A₂ = Vec{3}((0.0, 1.0, 0.0)); G₃ = Vec{3}((0.0, 0.0, 1.0))
    A_metric = SymmetricTensor{2,2}((1.0, 0.0, 1.0))
    cms(Em) = A_metric + 2Em

    # backward compatibility: default (tension_field=false) unchanged vs a 3-arg build
    Em_taut = SymmetricTensor{2,2}((0.01, 0.0, 0.01))
    N0, C0 = membrane_stress_and_tangent(mat0, cms(Em_taut), A_metric, A₁, A₂, G₃)
    Nw, Cw = membrane_stress_and_tangent(matw, cms(Em_taut), A_metric, A₁, A₂, G₃)
    @test Nw ≈ N0            # taut biaxial tension → unchanged
    @test Cw ≈ C0

    # wrinkled: ε₁₁>0, ε₂₂<0 ⇒ σ₁>0>σ₂ → minor principal relaxes to 0, major kept
    Em_wr = SymmetricTensor{2,2}((0.02, 0.0, -0.02))
    N0w, _ = membrane_stress_and_tangent(mat0, cms(Em_wr), A_metric, A₁, A₂, G₃)
    Nwr, _ = membrane_stress_and_tangent(matw, cms(Em_wr), A_metric, A₁, A₂, G₃)
    σ0 = eigvals(N0w); σw = eigvals(Nwr)
    @test minimum(σ0) < 0
    @test isapprox(minimum(σw), 0.0; atol=1e-6 * maximum(σ0))
    @test isapprox(maximum(σw), maximum(σ0); rtol=1e-8)

    # slack: biaxial compression → N ≈ 0
    Em_sl = SymmetricTensor{2,2}((-0.01, 0.0, -0.01))
    Nsl, _ = membrane_stress_and_tangent(matw, cms(Em_sl), A_metric, A₁, A₂, G₃)
    @test isapprox(norm(Nsl), 0.0; atol=1e-8)

    # relaxation is frame-invariant: rotate the frame 30°, principal stresses match
    θ = π/6; c, s = cos(θ), sin(θ)
    B₁ = Vec{3}((c, s, 0.0)); B₂ = Vec{3}((-s, c, 0.0))
    Nrot, _ = membrane_stress_and_tangent(matw, cms(Em_wr), A_metric, B₁, B₂, G₃)
    @test isapprox(sort(eigvals(Nrot)), sort(σw); rtol=1e-6)

    # consistent tangent: FD of the relaxed stress in the wrinkled regime (ε floor)
    matf = LinearElastic(E, ν, t; tension_field=true, ε_tf=1e-6)
    N_of(c) = membrane_stress_and_tangent(matf, c, A_metric, A₁, A₂, G₃)[1]
    _, Cf = membrane_stress_and_tangent(matf, cms(Em_wr), A_metric, A₁, A₂, G₃)
    δ = SymmetricTensor{2,2}((0.7, -0.4, 0.2))
    h = 1e-8
    dN_fd = (N_of(cms(Em_wr) + h*δ) - N_of(cms(Em_wr) - h*δ)) / (2h)
    @test isapprox(dN_fd, (Cf ⊡ δ) / 2; rtol=1e-4)   # ∂N/∂c = C/2
end
