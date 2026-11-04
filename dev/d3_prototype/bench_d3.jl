include(joinpath(@__DIR__, "rm_d3.jl")); using Random; Random.seed!(2)
ip, qr = Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3)
s = RMShell(ip, ip, qr; E = 1e3, ν = 0.3, t = 0.05)
x = [Vec{3}((ξ[1], ξ[2], 0.2ξ[1]^2)) for ξ in Ferrite.reference_coordinates(ip)]
reinit!(s.cv, x); s.D .= [normalize(Vec{3}((-0.4ξ[1], 0.0, 1.0))) for ξ in Ferrite.reference_coordinates(ip)]
n = 54; ue = 0.1 .* randn(n); for a in 1:9  # make ω ⟂ D
    ω = Vec{3}(k -> ue[27 + 3(a-1) + k]); ω -= (ω ⋅ s.D[a]) * s.D[a]; ue[27 .+ (3(a-1)+1:3a)] .= ω
end
ke, re = zeros(n, n), zeros(n)
f() = (fill!(ke, 0); fill!(re, 0); element!(ke, re, s, ue))
f(); a = @allocated f()
t = minimum(@elapsed(for _ in 1:200; f(); end) for _ in 1:5) / 200
t_r = minimum(@elapsed(for _ in 1:200; reinit!(s.cv, x); end) for _ in 1:5) / 200
println("D3 prototype   element!: $(round(t*1e6, digits=1)) μs, $(a) bytes;  reinit!(cv): $(round(t_r*1e6, digits=2)) μs")
