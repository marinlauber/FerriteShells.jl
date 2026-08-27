"""
    MITC{3,4,T}

Mixed Interpolation of Tensorial Components data for the 3-node triangular shell element
(Lee & Bathe 2004). The transverse shear is assumed *constant along each edge*, the triangular
counterpart of MITC4.

Tying points (reference domain ``r,s \\ge 0``, ``r+s \\le 1``), at the edge midpoints:
  ``\\gamma_1 = a_1 \\cdot d`` at ``(1/2, 0)``
  ``\\gamma_2 = a_2 \\cdot d`` at ``(0, 1/2)``
  ``\\gamma_q = (\\gamma_2-\\gamma_1)/\\sqrt{2}`` at ``(1/2, 1/2)`` — the hypotenuse condition, which
  needs *both* covariant components there, giving ``M = 4`` tying entries.

Assumed field (rotated Raviart–Thomas ``RT_0``), Lee & Bathe Eq. (25):
``\\tilde\\gamma_1 = a_1 + c\\,s``, ``\\tilde\\gamma_2 = a_2 - c\\,r``, i.e. the span of
``(1,0)``, ``(0,1)`` and ``(s,-r)``.
"""
MITC3(ip_shape::Interpolation, qr::QuadratureRule) = MITC{3}(ip_shape, qr, MITC3)

# constant shear along each of the three edges, tied against the RT₀ space
tying_conditions(::typeof(MITC3)) = (
    ((Vec{2}((0.5, 0.0)), Ê₁),
     (Vec{2}((0.0, 0.5)), Ê₂),
     (Vec{2}((0.5, 0.5)), Ê_q)),
    (ξ -> Vec{2}((1.0, 0.0)),
     ξ -> Vec{2}((0.0, 1.0)),
     ξ -> Vec{2}((ξ[2], -ξ[1]))),
)
