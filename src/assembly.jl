
"""
    membrane_residuals!(re, scv, x, u_e, mat)

Compute the element residual vector for the membrane contribution.
- `re`: the residual vector to be filled (preallocated)
- `scv`: the shell cell values containing quadrature and shape function info
- `x`: the current nodal coordinates of the element
- `u_e`: the current nodal displacements of the element
- `mat`: the material model providing stress computation
"""
# TODO this assumes u_e only contains in-plane displacements
function membrane_residuals!(re, scv, x, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        a₁,a₂,A_metric,a_metric = kinematics(scv, qp, x, u_e)
        E = 0.5 * (a_metric - A_metric)
        N = mat.H ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            re[3I-2:3I] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent!(Ke, scv, x, u_e, mat)

Compute the element tangent matrix for the membrane contribution.
- `Ke`: the tangent matrix to be filled (preallocated)
- `scv`: the shell cell values containing quadrature and shape function info
- `x`: the current nodal coordinates of the element
- `u_e`: the current nodal displacements of the element
- `mat`: the material model providing tangent computation
"""
# TODO this assumes u_e only contains in-plane displacements
function membrane_tangent!(ke, scv, x, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        a₁,a₂,A_metric,a_metric = kinematics(scv, qp, x, u_e)
        E = 0.5 * (a_metric - A_metric)
        N = mat.H ⊡ E
        C = mat.C
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, I)
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = Ferrite.reference_shape_gradient(scv.ip_shape, ξ, J)
                # geometric stiffness: N^{αβ} (∂NI/∂ξ_α)(∂NJ/∂ξ_β) I₃
                geo_scalar = ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
                             ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)
                Kgeo = geo_scalar * one(SymmetricTensor{2,3})
                # material stiffness: C^{αβγδ} (∂NI/∂ξ_α)(∂NJ/∂ξ_γ) a_β ⊗ a_δ
                H1 = SymmetricTensor{2,2}((∂NJ1, 0.5∂NJ2, 0.0))
                H2 = SymmetricTensor{2,2}((0.0, 0.5∂NJ1, ∂NJ2))
                D1 = C ⊡ H1
                D2 = C ⊡ H2
                Kmat = zero(Tensor{2,3})
                for (α,∂NIα) in enumerate((∂NI1, ∂NI2)), (β,aβ) in enumerate((a₁, a₂))
                    v = D1[α,β]*a₁ + D2[α,β]*a₂
                    Kmat += ∂NIα * (aβ ⊗ v)
                end
                ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * dΩ
            end
        end
    end
end
