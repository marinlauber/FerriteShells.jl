
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
        N = contravariant_elasticity(mat, A_metric) ⊡ E
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
        C = contravariant_elasticity(mat, A_metric)
        N = C ⊡ E
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


using ForwardDiff

"""
Kirchhoff-Love bending strain energy at an element.
Curvature change κ_{αβ} = b_{αβ} - B_{αβ} where b/B are the current/reference
second fundamental forms: b_{αβ} = a_{α,β}·n, B_{αβ} = A_{α,β}·N.
Second derivatives a_{α,β} = Σ_I (x_I+u_I) ∂²N_I/∂ξ_α∂ξ_β require Q2+ elements;
for Q4 bilinear, ∂²N/∂ξ₁² = ∂²N/∂ξ₂² = 0 so only the twist κ₁₂ is captured.
"""
function bending_energy(u_flat, scv, x, mat)
    T = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    u_e = [Vec{3,T}((u_flat[3i-2], u_flat[3i-1], u_flat[3i])) for i in 1:n_nodes]
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        ξ = scv.qr.points[qp]
        A₁  = zero(Vec{3,Float64}); A₂  = zero(Vec{3,Float64})
        a₁  = zero(Vec{3,T});       a₂  = zero(Vec{3,T})
        A₁₁ = zero(Vec{3,Float64}); A₁₂ = zero(Vec{3,Float64}); A₂₂ = zero(Vec{3,Float64})
        a₁₁ = zero(Vec{3,T});       a₁₂ = zero(Vec{3,T});       a₂₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            d2N, dN, _ = Ferrite.reference_shape_hessian_gradient_and_value(scv.ip_shape, ξ, I)
            XI = x[I]; UI = u_e[I]
            A₁  += XI * dN[1];        A₂  += XI * dN[2]
            a₁  += (XI + UI) * dN[1]; a₂  += (XI + UI) * dN[2]
            A₁₁ += XI * d2N[1,1];    A₁₂ += XI * d2N[1,2];    A₂₂ += XI * d2N[2,2]
            a₁₁ += (XI + UI) * d2N[1,1]; a₁₂ += (XI + UI) * d2N[1,2]; a₂₂ += (XI + UI) * d2N[2,2]
        end
        A_metric = SymmetricTensor{2,2,Float64}((dot(A₁,A₁), dot(A₁,A₂), dot(A₂,A₂)))
        N_n = (A₁ × A₂) / norm(A₁ × A₂)  # reference unit normal
        n_n = (a₁ × a₂) / norm(a₁ × a₂)  # current unit normal
        B = SymmetricTensor{2,2,Float64}((dot(A₁₁,N_n), dot(A₁₂,N_n), dot(A₂₂,N_n)))
        b = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ = b - B
        D = contravariant_bending_stiffness(mat, A_metric)
        W += 0.5 * (κ ⊡ D ⊡ κ) * scv.detJdV[qp]
    end
    return W
end

function bending_residuals!(re, scv, x, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy(u, scv, x, mat), u_e)
end

function bending_tangent!(ke, scv, x, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy(u, scv, x, mat), u_e)
end
