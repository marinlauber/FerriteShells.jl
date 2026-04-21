"""
    MITC{9,6,T}

Mixed Interpolation of Tensorial Components data for the 9-node quadrilateral shell element.

Tying points (reference domain ``[-1,1]^2``):
  ``\\gamma_1 = a_1\\cdot d``: (``\\pm1/\\sqrt{3}``, ``-1``), (``\\pm1/\\sqrt{3}``, ``0``), (``\\pm1/\\sqrt{3}``, ``+1``) 6 points, linear in ``\\xi_1`` between ``\\pm1/\\sqrt{3}``, quadratic in ``\\xi_2``
  ``\\gamma_2 = a_2\\cdot d``: (``-1``, ``\\pm1/\\sqrt{3}``), (``0``, ``\\pm1/\\sqrt{3}``), (``+1``, ``\\pm1/\\sqrt{3}``) 6 points, quadratic in ``\\xi_1``, linear in ``\\xi_2`` between ``\\pm1/\\sqrt{3}``

The points ``\\pm1/\\sqrt{3}`` are superconvergent points for linear functions.
"""
function MITC9(ip_shape::Interpolation, qr::QuadratureRule)
    T    = Float64
    n_qp = length(qr.weights)
    # tying point for Q9
    s = T(1/sqrt(3))
    ξ_tie_1 = [Vec{2}((-s,-1.)), Vec{2}((s,-1.)), Vec{2}((-s,0.)), Vec{2}((s,0.)), Vec{2}((-s,1.)), Vec{2}((s,1.))]
    ξ_tie_2 = [Vec{2}((-1.,-s)), Vec{2}((0.,-s)), Vec{2}((1.,-s)), Vec{2}((-1.,s)), Vec{2}((0.,s)), Vec{2}((1.,s))]
    # Interpolation weights h_tie[qp, k] such that γ̃(ξ_qp) = Σ_k h_tie[qp,k] · γ(ξ_tie_k)
    h_tie_1 = zeros(T, n_qp, 6);  h_tie_2 = zeros(T, n_qp, 6)
    for q in 1:n_qp
        ξ, η = qr.points[q][1], qr.points[q][2]
        h₁ = (1 - sqrt(3)*ξ)/2;  h₂ = (1 + sqrt(3)*ξ)/2
        L₁ = η*(η-1)/2;          L₂ = 1 - η^2;           L₃ = η*(η+1)/2
        h_tie_1[q, :] = [h₁*L₁, h₂*L₁, h₁*L₂, h₂*L₂, h₁*L₃, h₂*L₃]
        l₁ = ξ*(ξ-1)/2;           l₂ = 1 - ξ^2;           l₃ = ξ*(ξ+1)/2
        g₁ = (1 - sqrt(3)*η)/2;   g₂ = (1 + sqrt(3)*η)/2
        h_tie_2[q, :] = [l₁*g₁, l₂*g₁, l₃*g₁, l₁*g₂, l₂*g₂, l₃*g₂]
    end
    # return the structure
    MITC{9}(ip_shape, h_tie_1, h_tie_2, ξ_tie_1, ξ_tie_2)
end