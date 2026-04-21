"""
    MITC{4,2,T}

Mixed Interpolation of Tensorial Components data for the 4-node quadrilateral shell element.

Tying points (reference domain ``[-1,1]^2``):
  ``\\gamma_1 = a_1 \\cdot d``: (``0,`` ``\\pm1``)   2 points, constant in ``\\xi_1``, linear in ``\\xi_2``
  ``\\gamma_2 = a_2 \\cdot d``: (``\\pm1``,`` 0``)   2 points, linear in ``\\xi_1``, constant in ``\\xi_2``
"""
function MITC4(ip_shape::Interpolation, qr::QuadratureRule)
    T    = Float64
    n_qp = length(qr.weights)
    # tying points for Q4
    ξ_tie_1 = [Vec{2,T}((0.,-1.)), Vec{2,T}((0.,1.))]
    ξ_tie_2 = [Vec{2,T}((-1.,0.)), Vec{2,T}((1.,0.))]
    # Interpolation weights h_tie[qp, k] such that γ̃(ξ_qp) = Σ_k h_tie[qp,k] · γ(ξ_tie_k)
    h_tie_1 = zeros(T, n_qp, 2);  h_tie_2 = zeros(T, n_qp, 2)
    for q in 1:n_qp
        ξ, η = qr.points[q][1], qr.points[q][2]
        h_tie_1[q, :] = [(1 - η)/2, (1 + η)/2]
        h_tie_2[q, :] = [(1 - ξ)/2, (1 + ξ)/2]
    end
    # return the structure
    MITC{4}(ip_shape, h_tie_1, h_tie_2, ξ_tie_1, ξ_tie_2)
end