# Membrane kernels, shared by the Kirchhoff–Love and Reissner–Mindlin formulations

"""
    membrane_residuals!(re, scv, u_e, mat)

Membrane residual ``r_I = \\int N^{\\alpha\\beta} \\partial_\\alpha N_I a_\\beta \\, d\\Omega`` of a
Kirchhoff–Love or Reissner–Mindlin shell. `u_e = u[celldofs(cell)]`; its displacement dofs
are read at `scv.ru`. The stress resultant rows ``P_\\alpha = N^{\\alpha\\beta} a_\\beta`` are
computed once per quadrature point.
"""
membrane_residuals!(re, scv::ShellCellValues, u_e, mat) = rm_element!(nothing, re, scv, u_e, mat, Val(:membrane))

"""
    membrane_tangent!(ke, scv, u_e, mat)

Membrane tangent of a Kirchhoff–Love or Reissner–Mindlin shell.
Material part: ``K^\\text{mat}_{IJ} = \\partial_\\alpha N_I \\partial_\\delta N_J M_{\\alpha\\delta}`` where
``M_{\\alpha\\delta} = C^{\\alpha\\beta\\gamma\\delta} a_\\beta\\otimes a_\\gamma``.
Geometric part: ``K^\\text{geo}_{IJ} = (\\partial_\\alpha N_I N^{\\alpha\\beta} \\partial_\\beta N_J) \\mathbb{1}_3``.
Both ``M_{\\alpha\\delta}`` and ``N`` are computed once per quadrature point, outside the node loops.
"""
membrane_tangent!(ke, scv::ShellCellValues, u_e, mat) = rm_element!(ke, nothing, scv, u_e, mat, Val(:membrane))

# Membrane residual at a single quadrature point.
@inline function membrane_residual_qp!(re, scv, qp, a₁, a₂, N)
    P₁ = N[1,1]*a₁ + N[1,2]*a₂
    P₂ = N[2,1]*a₁ + N[2,2]*a₂
    dΩ = scv.detJdV[qp]
    ru = scv.ru
    for I in 1:getnbasefunctions(scv.ip_shape)
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        @views re[ru[3I-2:3I]] .+= (∂NI1*P₁ + ∂NI2*P₂) * dΩ
    end
end

# Per-qp stiffness blocks M_αδ = C^{αβγδ} a_β⊗a_γ
@inline function stiffness_block(C::SymmetricTensor{4,2,T}, a₁::Vec{3,T}, a₂::Vec{3,T}) where T
    M₁₁ = C[1,1,1,1]*(a₁⊗a₁) + C[1,1,1,2]*(a₁⊗a₂ + a₂⊗a₁) + C[1,2,1,2]*(a₂⊗a₂)
    M₁₂ = C[1,1,1,2]*(a₁⊗a₁) + C[1,1,2,2]*(a₁⊗a₂) + C[1,2,1,2]*(a₂⊗a₁) + C[1,2,2,2]*(a₂⊗a₂)
    M₂₂ = C[1,2,1,2]*(a₁⊗a₁) + C[1,2,2,2]*(a₁⊗a₂ + a₂⊗a₁) + C[2,2,2,2]*(a₂⊗a₂)
    M₁₁, M₁₂, M₂₂
end

# Membrane tangent at a single quadrature point.
function membrane_tangent_qp!(ke, scv, qp, a₁, a₂, N, C)
    M₁₁, M₁₂, M₂₂ = stiffness_block(C, a₁, a₂)
    dΩ = scv.detJdV[qp]
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru = scv.ru
    for I in 1:n_nodes
        ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
        for J in 1:n_nodes
            ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
            K_mat = ∂NI1*∂NJ1*M₁₁ + ∂NI1*∂NJ2*M₁₂ + ∂NI2*∂NJ1*transpose(M₁₂) + ∂NI2*∂NJ2*M₂₂
            K_geo = (∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) + ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)) * one(SymmetricTensor{2,3})
            @views ke[ru[3I-2:3I], ru[3J-2:3J]] .+= (K_mat + K_geo) * dΩ
        end
    end
end
