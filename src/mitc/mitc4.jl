"""
    MITC{4,4,T}

Mixed Interpolation of Tensorial Components data for the 4-node quadrilateral shell element
(Dvorkin & Bathe 1984).

Tying points (reference domain ``[-1,1]^2``), at the edge midpoints:
  ``\\gamma_1 = a_1 \\cdot d`` at ``(0, \\pm1)``
  ``\\gamma_2 = a_2 \\cdot d`` at ``(\\pm1, 0)``
Each condition ties a single covariant component, so ``M = 4`` and the off-component
columns of `h_tie_1`/`h_tie_2` vanish.

Assumed field: ``\\tilde\\gamma_1 = a_1 + b_1\\xi_2`` (constant in ``\\xi_1``, linear in ``\\xi_2``) and
``\\tilde\\gamma_2 = a_2 + b_2\\xi_1``, the spaces obtained by differentiating the bilinear
displacement field ``w = c_0 + c_1\\xi_1 + c_2\\xi_2 + c_3\\xi_1\\xi_2``.
"""
MITC4(ip_shape::Interpolation, qr::QuadratureRule) = MITC{4}(ip_shape, qr, MITC4)

# γ₁ tied on the two ξ₁-edges, γ₂ on the two ξ₂-edges, against span{1, ξ₂} × span{1, ξ₁}.
tying_conditions(::typeof(MITC4)) = (
    ((Vec{2}(( 0.0, -1.0)), Ê₁), (Vec{2}(( 0.0, 1.0)), Ê₁),
     (Vec{2}((-1.0,  0.0)), Ê₂), (Vec{2}(( 1.0, 0.0)), Ê₂)),
    (ξ -> Vec{2}((1.0, 0.0)), ξ -> Vec{2}((ξ[2], 0.0)),
     ξ -> Vec{2}((0.0, 1.0)), ξ -> Vec{2}((0.0, ξ[1]))),
)
