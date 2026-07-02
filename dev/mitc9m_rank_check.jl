using FerriteShells, LinearAlgebra, Printf

# First gate for the MITC9M (membrane-tied) element: a single Q9 element stiffness,
# assembled as the FD Hessian of energy_RM, must have EXACTLY 6 zero-energy modes
# (3 translations + 3 rotations). Any extra near-zero eigenvalue is a spurious
# (hourglass) mode introduced by the in-plane assumed-strain field.

mat = LinearElastic(1.0e6, 0.3, 0.01)
ip  = Lagrange{RefQuadrilateral, 2}()
qr  = QuadratureRule{RefQuadrilateral}(3)

X_FLAT = [
    Vec{3}((0.0,0.0,0.0)), Vec{3}((1.0,0.0,0.0)), Vec{3}((1.0,1.0,0.0)), Vec{3}((0.0,1.0,0.0)),
    Vec{3}((0.5,0.0,0.0)), Vec{3}((1.0,0.5,0.0)), Vec{3}((0.5,1.0,0.0)), Vec{3}((0.0,0.5,0.0)),
    Vec{3}((0.5,0.5,0.0)),
]
# doubly-curved single element: lift to a paraboloid z = 0.15(x²+y²)
warp(p) = Vec{3}((p[1], p[2], 0.15*(p[1]^2 + p[2]^2)))
X_CURV = warp.(X_FLAT)
# distorted-in-plane (skewed) element to stress the in-plane field
skew(p) = Vec{3}((p[1] + 0.2*p[2], p[2] + 0.15*p[1], 0.0))
X_SKEW = skew.(X_FLAT)

function single_element_K(mitc, coords)
    scv = mitc === nothing ? ShellCellValues(qr, ip, ip) : ShellCellValues(qr, ip, ip; mitc=mitc)
    reinit!(scv, coords)
    ke = zeros(45, 45)
    tangent_RM_FD!(ke, scv, zeros(45), mat)
    Symmetric(ke)
end

function report(label, mitc, coords)
    K = single_element_K(mitc, coords)
    λ = sort(eigvals(Matrix(K)))
    λmax = maximum(abs, λ)
    tol = 1e-8 * λmax
    nzero = count(<(tol), abs.(λ))
    sym = norm(K - K') / norm(K)
    @printf("%-22s zero-modes=%d (expect 6)  λ[1:8]=%s  sym=%.1e\n",
            label, nzero, string(round.(λ[1:8] ./ λmax, sigdigits=2)), sym)
    nzero
end

println("=== single-element rank / spurious-mode check ===")
for (name, X) in (("flat", X_FLAT), ("curved", X_CURV), ("skew", X_SKEW))
    println("-- $name element --")
    report("  NoMITC (displ. RM) ", nothing, X)
    report("  MITC9  (shear only)", MITC9,  X)
    report("  MITC9M (membrane)  ", MITC9M, X)
end
