include(joinpath(@__DIR__, "rm_d3.jl")); using Random; Random.seed!(1)
for (name, ip, qr) in (("Q9", Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3)),
                       ("Q4", Lagrange{RefQuadrilateral,1}(), QuadratureRule{RefQuadrilateral}(2)))
    s = RMShell(ip, ip, qr; E = 1e3, ν = 0.3, t = 0.05)
    x = [Vec{3}((ξ[1], 0.8ξ[2] + 0.1ξ[1], 0.2ξ[1]^2 - 0.1ξ[2]^2 + 0.05ξ[1]ξ[2])) for ξ in Ferrite.reference_coordinates(ip)]
    reinit!(s.cv, x)
    s.D .= [normalize(rand(Vec{3}) + Vec{3}((-0.5,-0.5,1.5))) for _ in x]
    n = 6length(x); ue = 0.3 .* (rand(n) .- 0.5)
    ke, re = element!(zeros(n, n), zeros(n), s, ue)
    kfd = zeros(n, n); h = 1e-6
    for j in 1:n
        up = copy(ue); up[j] += h; um = copy(ue); um[j] -= h
        kfd[:, j] = (element!(zeros(n,n), zeros(n), s, up)[2] - element!(zeros(n,n), zeros(n), s, um)[2]) / 2h
    end
    println(name, ": ‖K-K_fd‖/‖K‖ = ", norm(ke - kfd)/norm(ke), "   ‖K-Kᵀ‖/‖K‖ = ", norm(ke - ke')/norm(ke))
    # rigid body motion: translation + finite rotation R(θ) of the whole element → zero residual
    θ = Vec{3}((0.4, -0.3, 0.7)); R = Tensors.gradient(v -> rotate(θ, v), zero(Vec{3}))  # R = ∂(R v)/∂v
    urb = vcat([Vec{3}((0.1,0.2,0.3)) + R ⋅ xi - xi for xi in x]...)
    # nodal rotation vector composing R with zero initial rotation is θ itself: d_a = R·D_a
    ωrb = vcat([θ for _ in x]...)
    _, r_rb = element!(zeros(n,n), zeros(n), s, vcat(collect(urb), collect(ωrb)))
    println(name, ": rigid-body residual ‖r‖ = ", norm(r_rb))
end
