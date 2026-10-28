"""
    MITC{9,12,T}

Mixed Interpolation of Tensorial Components data for the 9-node quadrilateral shell element
(Bucalem & Bathe 1993).

Tying points (reference domain ``[-1,1]^2``), on the ``\\pm1/\\sqrt3`` Gauss lines:
  ``\\gamma_1 = a_1 \\cdot d`` at ``(\\pm1/\\sqrt3,\\ -1), (\\pm1/\\sqrt3,\\ 0), (\\pm1/\\sqrt3,\\ +1)``
  ``\\gamma_2 = a_2 \\cdot d`` at ``(-1,\\ \\pm1/\\sqrt3), (0,\\ \\pm1/\\sqrt3), (+1,\\ \\pm1/\\sqrt3)``
Each condition ties a single covariant component, so ``M = 12`` and the off-component
columns of `h_tie_1`/`h_tie_2` vanish.

Assumed field: ``\\tilde\\gamma_1 \\in`` span``\\{1,\\xi_1\\}\\otimes\\{1,\\xi_2,\\xi_2^2\\}`` — linear in ``\\xi_1``,
quadratic in ``\\xi_2`` — and ``\\tilde\\gamma_2`` its transpose. The ``\\pm1/\\sqrt3`` stations are the
superconvergent points of the degree that is dropped.
"""
MITC9(ip_shape::Interpolation, qr::QuadratureRule) = MITC{9}(ip_shape, qr, MITC9)

# Gauss station of the reduced direction: γ₁ is sampled at ξ₁ = ∓S_Q9, γ₂ at ξ₂ = ∓S_Q9.
const S_Q9 = 1 / sqrt(3)

tying_conditions(::typeof(MITC9)) = (
    ((Vec{2}((-S_Q9, -1.0)), Ê₁), (Vec{2}((S_Q9, -1.0)), Ê₁),
     (Vec{2}((-S_Q9,  0.0)), Ê₁), (Vec{2}((S_Q9,  0.0)), Ê₁),
     (Vec{2}((-S_Q9,  1.0)), Ê₁), (Vec{2}((S_Q9,  1.0)), Ê₁),
     (Vec{2}((-1.0, -S_Q9)), Ê₂), (Vec{2}((0.0, -S_Q9)), Ê₂), (Vec{2}((1.0, -S_Q9)), Ê₂),
     (Vec{2}((-1.0,  S_Q9)), Ê₂), (Vec{2}((0.0,  S_Q9)), Ê₂), (Vec{2}((1.0,  S_Q9)), Ê₂)),
    (ξ -> Vec{2}((1.0, 0.0)),           ξ -> Vec{2}((ξ[1], 0.0)),
     ξ -> Vec{2}((ξ[2], 0.0)),          ξ -> Vec{2}((ξ[1]*ξ[2], 0.0)),
     ξ -> Vec{2}((ξ[2]^2, 0.0)),        ξ -> Vec{2}((ξ[1]*ξ[2]^2, 0.0)),
     ξ -> Vec{2}((0.0, 1.0)),           ξ -> Vec{2}((0.0, ξ[1])),
     ξ -> Vec{2}((0.0, ξ[2])),          ξ -> Vec{2}((0.0, ξ[1]*ξ[2])),
     ξ -> Vec{2}((0.0, ξ[1]^2)),        ξ -> Vec{2}((0.0, ξ[1]^2*ξ[2]))),
)
