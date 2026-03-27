"""
    MITC{9,T}

Mixed Interpolation of Tensorial Components data for the 9-node quadrilateral shell element
(Bucalem & Bathe 1993).

Tying points (reference domain [-1,1]²):
  γ₁ = a₁·d: (±1/√3, -1), (±1/√3,  0), (±1/√3, +1)   [6 points, vary in ξ₂]
  γ₂ = a₂·d: (-1, ±1/√3), ( 0, ±1/√3), (+1, ±1/√3)   [6 points, vary in ξ₁]
"""
function MITC9(ip_shape::Interpolation, qr::QuadratureRule)
    T       = Float64
    n_shape = getnbasefunctions(ip_shape)
    n_qp    = length(qr.weights)
    # tying point for Q9
    s = 1/sqrt(3)
    ξ_tie_1 = [Vec{2}((-s,-1.)), Vec{2}((s,-1.)), Vec{2}((-s,0.)), Vec{2}((s,0.)), Vec{2}((-s,1.)), Vec{2}((s,1.))]
    ξ_tie_2 = [Vec{2}((-1.,-s)), Vec{2}((0.,-s)), Vec{2}((1.,-s)), Vec{2}((-1.,s)), Vec{2}((0.,s)), Vec{2}((1.,s))]
    # shape values there
    N_tie_1 = zeros(T, n_shape, 6);  dNdξ_tie_1 = Matrix{Vec{2,T}}(undef, n_shape, 6)
    N_tie_2 = zeros(T, n_shape, 6);  dNdξ_tie_2 = Matrix{Vec{2,T}}(undef, n_shape, 6)
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
    # γ₁: linear in ξ₁ between ±1/√3, quadratic in ξ₂ (Lagrange over {-1,0,+1})
    # γ₂: quadratic in ξ₁ (Lagrange over {-1,0,+1}), linear in ξ₂ between ±1/√3
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

    MITC{9,6,T}(
        N_tie_1, dNdξ_tie_1, N_tie_2, dNdξ_tie_2, h_tie_1, h_tie_2,
        fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6),
        fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6),
        fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6),
        fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6), fill(zero(Vec{3,T}), 6),
        ξ_tie_1, ξ_tie_2,
    )
end