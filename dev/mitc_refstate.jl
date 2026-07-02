using FerriteShells, LinearAlgebra, Printf

# Is the reference configuration (u=0) stress-free for the MITC element on a CURVED
# element? If the residual = ∂energy_RM/∂u at u=0 is nonzero, the reference is spuriously
# pre-stressed → geometric stiffness can go indefinite. Also probe the tying-point shear
# strains directly at u=0: they must vanish for a consistent reference.

mat = LinearElastic(1.0e6, 0.3, 0.01)
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)

X_FLAT = [
    Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.0)), Vec{3}((1.0,1.0,0.0)), Vec{3}((0.0,1.0,0.0)),
    Vec{3}((0.5,0.0,0.0)), Vec{3}((1.0,0.5,0.0)), Vec{3}((0.5,1.0,0.0)), Vec{3}((0.0,0.5,0.0)),
    Vec{3}((0.5,0.5,0.0)),
]
warp(p) = Vec{3}((p[1], p[2], 0.15*(p[1]^2 + p[2]^2)))
X_CURV = warp.(X_FLAT)

for (name, X) in (("flat", X_FLAT), ("curved", X_CURV))
    println("-- $name --")
    for (lbl, mitc) in (("NoMITC", nothing), ("MITC9", MITC9))
        scv = mitc === nothing ? ShellCellValues(qr, ip, ip) : ShellCellValues(qr, ip, ip; mitc=mitc)
        reinit!(scv, X)
        re = zeros(45); residuals_RM_FD!(re, scv, zeros(45), mat)
        if mitc !== nothing
            γ₁_k, γ₂_k = FerriteShells.tying_shear_strains(scv.mitc, zeros(45))
            @printf("  %-7s |R(u=0)|=%.3e   max|γ_tie(u=0)|=%.3e\n",
                    lbl, norm(re), max(maximum(abs,γ₁_k), maximum(abs,γ₂_k)))
        else
            @printf("  %-7s |R(u=0)|=%.3e\n", lbl, norm(re))
        end
    end
end
