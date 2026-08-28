"""
    MITC{6,10,T}

Mixed Interpolation of Tensorial Components data for the 6-node triangular shell element,
variant **MITC6-a** (Lee & Bathe 2004, Eq. 35–39) — linear transverse shear along the edges plus
one interior tying point carrying the quadratic part.

Tying points (reference domain ``r,s \\ge 0``, ``r+s \\le 1``): the two Gauss stations
``1/2 \\mp 1/(2\\sqrt3)`` of each edge — ``\\gamma_1`` on ``s=0``, ``\\gamma_2`` on ``r=0``, and
``\\gamma_q = (\\gamma_2-\\gamma_1)/\\sqrt{2}`` on the hypotenuse — plus the centroid ``(1/3,1/3)``.
The hypotenuse and centroid conditions need both covariant components, giving ``M = 10`` entries.

Assumed field (rotated Raviart–Thomas ``RT_1``, the MITC7 plate space), Lee & Bathe Eq. (39):
``\\tilde\\gamma_1 = a_1 + b_1 r + c_1 s + s(d r + e s)``,
``\\tilde\\gamma_2 = a_2 + b_2 r + c_2 s - r(d r + e s)``.

Dropping the two interior conditions and the last two basis fields gives the linear variant
MITC6-b (Eq. 40–41), which is stiffer in bending-dominated problems.
"""
MITC6a(ip_shape::Interpolation, qr::QuadratureRule) = MITC{6}(ip_shape, qr, MITC6a)

const MITC6 = MITC6a

# Edge tying abscissae (Lee & Bathe Eq. 35): the two-point Gauss stations of each edge, which
# make the assumed shear linear along the edges.
const R₁_T6 = 0.5 - 1 / (2 * sqrt(3))
const R₂_T6 = 0.5 + 1 / (2 * sqrt(3))

tying_conditions(::typeof(MITC6a)) = (
    ((Vec{2}((R₁_T6, 0.0)),   Ê₁),  (Vec{2}((R₂_T6, 0.0)),   Ê₁),
     (Vec{2}((0.0, R₁_T6)),   Ê₂),  (Vec{2}((0.0, R₂_T6)),   Ê₂),
     (Vec{2}((R₂_T6, R₁_T6)), Ê_q), (Vec{2}((R₁_T6, R₂_T6)), Ê_q),
     (Vec{2}((1/3, 1/3)),     Ê₁),  (Vec{2}((1/3, 1/3)),     Ê₂)),
    (ξ -> Vec{2}((1.0, 0.0)), ξ -> Vec{2}((ξ[1], 0.0)), ξ -> Vec{2}((ξ[2], 0.0)),
     ξ -> Vec{2}((0.0, 1.0)), ξ -> Vec{2}((0.0, ξ[1])), ξ -> Vec{2}((0.0, ξ[2])),
     ξ -> Vec{2}((ξ[2]*ξ[1], -ξ[1]^2)), ξ -> Vec{2}((ξ[2]^2, -ξ[1]*ξ[2]))),
)

# Lee & Bathe pair MITC6 with a tied *in-plane* strain field (Eq. 30–34) to also cure membrane
# locking. Not covered here: that is the triangular analogue of the MITC9M membrane tying.
