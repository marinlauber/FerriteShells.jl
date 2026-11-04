# Reference for ../bench_d3.jl: one Q9 element, residual + tangent, FerriteShells RM kernels.
# Run from the repository root with the package environment: julia --project=. dev/d3_prototype/reference/bench_ferriteshells.jl
using FerriteShells, Random; Random.seed!(2)
ip, qr = Lagrange{RefQuadrilateral,2}(), QuadratureRule{RefQuadrilateral}(3)
mat = LinearElastic(1e3, 0.3, 0.05)
x = [Vec{3}((ξ[1], ξ[2], 0.2ξ[1]^2)) for ξ in Ferrite.reference_coordinates(ip)]
for (name, m) in (("no MITC", nothing), ("MITC9", MITC9))
    scv = ShellCellValues(qr, ip, ip; mitc = m); reinit!(scv, x)
    ue = 0.1 .* randn(45); ke, re = zeros(45, 45), zeros(45)
    f() = (fill!(ke, 0); fill!(re, 0); membrane_tangent_RM!(ke, scv, ue, mat); membrane_residuals_RM!(re, scv, ue, mat);
           bending_tangent_RM!(ke, scv, ue, mat); bending_residuals_RM!(re, scv, ue, mat))
    f(); a = @allocated f()
    t = minimum(@elapsed(for _ in 1:200; f(); end) for _ in 1:5) / 200
    t_r = minimum(@elapsed(for _ in 1:200; reinit!(scv, x); end) for _ in 1:5) / 200
    println("FerriteShells $name: $(round(t*1e6, digits=1)) μs, $(a) bytes;  reinit!(scv): $(round(t_r*1e6, digits=2)) μs")
end
