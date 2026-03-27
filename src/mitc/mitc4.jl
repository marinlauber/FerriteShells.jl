"""
    MITC{4,T}

Mixed Interpolation of Tensorial Components data for the 4-node quadrilateral shell element
(Bucalem & Bathe 1993).

Tying points (reference domain [-1,1]²):
  γ₁ = a₁·d: (±1, 0)   [2 points, vary in ξ₂]
  γ₂ = a₂·d: (0, ±1)   [2 points, vary in ξ₁]
"""
function MITC4(ip_shape::Interpolation, qr::QuadratureRule)
    T       = Float64
    n_shape = getnbasefunctions(ip_shape)
    n_qp    = length(qr.weights)
    # tying points for Q4
    ξ_tie_1 = [Vec{2}((-1.,0.)), Vec{2}((1.,0.))]
    ξ_tie_2 = [Vec{2}((0.,-1.)), Vec{2}((0.,1.))]
    # shape values there
    N_tie_1 = zeros(T, n_shape, 2);  dNdξ_tie_1 = Matrix{Vec{2,T}}(undef, n_shape, 2)
    N_tie_2 = zeros(T, n_shape, 2);  dNdξ_tie_2 = Matrix{Vec{2,T}}(undef, n_shape, 2)
    for (k, ξ_k) in enumerate(ξ_tie_1)
        for I in 1:n_shape
            dN, Nval = Ferrite.reference_shape_gradient_and_value(ip_shape, ξ_k, I)
            N_tie_1[I, k] = Nval;  dNdξ_tie_1[I, k] = dN
        end
    end
    for (k, ξ_k) in enumerate(ξ_tie_2)
        for I in 1:n_shape
            dN, Nval = Ferrite.reference_shape_gradient_and_value(ip_shape, ξ_k, I)
            N_tie_2[I, k] = Nval;  dNdξ_tie_2[I, k] = dN
        end
    end

    # Interpolation weights h_tie[qp, k] such that γ̃(ξ_qp) = Σ_k h_tie[qp,k] · γ(ξ_tie_k)
    # γ₁: linear in ξ₁ (Lagrange over {-1,+1})
    # γ₂: linear in ξ₂ (Lagrange over {-1,+1})
    h_tie_1 = zeros(T, n_qp, 2);  h_tie_2 = zeros(T, n_qp, 2)
    for q in 1:n_qp
        ξ, η = qr.points[q][1], qr.points[q][2]
        h_tie_1[q, :] = [ , ]
        h_tie_2[q, :] = [ , ]
    end

    MITC{4,2,T}(
        N_tie_1, dNdξ_tie_1, N_tie_2, dNdξ_tie_2, h_tie_1, h_tie_2,
        fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2),
        fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2),
        fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2),
        fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2), fill(zero(Vec{3,T}), 2),
        ξ_tie_1, ξ_tie_2,
    )
end