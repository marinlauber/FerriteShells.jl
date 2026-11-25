# Kirchhoff-Love assembly routines

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e = u[celldofs(cell)]`; its displacement dofs are read at `scv.ru`.
"""
function membrane_residuals_KL!(re, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru      = scv.ru
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for i in 1:n_nodes
            ui = Vec{3}(@view u_e[ru[3i-2:3i]])
            Δa₁ += ui * scv.dNdξ[i, qp][1]; Δa₂ += ui * scv.dNdξ[i, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, _ = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
                   scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            @views re[ru[3I-2:3I]] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent_KL!(ke, scv, u_e, mat)

Kirchhoff–Love membrane tangent. `u_e = u[celldofs(cell)]`; its displacement dofs are read at `scv.ru`.
"""
function membrane_tangent_KL!(ke, scv::ShellCellValues, u_e::AbstractVector{T}, mat) where T
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru      = scv.ru
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for i in 1:n_nodes
            ui = Vec{3}(@view u_e[ru[3i-2:3i]])
            Δa₁ += ui * scv.dNdξ[i, qp][1]; Δa₂ += ui * scv.dNdξ[i, qp][2]
        end
        a₁ = scv.A₁[qp] + Δa₁; a₂ = scv.A₂[qp] + Δa₂
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        N, C = membrane_stress_and_tangent(mat, c_ms, scv.A_metric[qp],
                   scv.A₁[qp], scv.A₂[qp], scv.G₃[qp])
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            for J in 1:n_nodes
                ∂NJ1, ∂NJ2 = scv.dNdξ[J, qp]
                geo_scalar = ∂NI1*(N[1,1]*∂NJ1 + N[1,2]*∂NJ2) +
                             ∂NI2*(N[2,1]*∂NJ1 + N[2,2]*∂NJ2)
                Kgeo = geo_scalar * one(SymmetricTensor{2,3})
                H1 = SymmetricTensor{2,2}((∂NJ1, 0.5∂NJ2, 0.0))
                H2 = SymmetricTensor{2,2}((0.0, 0.5∂NJ1, ∂NJ2))
                D1 = C ⊡ H1
                D2 = C ⊡ H2
                Kmat = zero(Tensor{2,3})
                for (α,∂NIα) in enumerate((∂NI1, ∂NI2)), (β,aβ) in enumerate((a₁, a₂))
                    v = D1[α,β]*a₁ + D2[α,β]*a₂
                    Kmat += ∂NIα * (aβ ⊗ v)
                end
                @views ke[ru[3I-2:3I], ru[3J-2:3J]] .+= (Kgeo + Kmat) * dΩ
            end
        end
    end
end


# Per-QP bending energy for KL, dispatched on material type.
# For LinearElastic: ½ κ:D:κ  (D evaluated at reference metric — safe, A_metric is plain Float64).
@inline function bending_kl_qp_energy(mat::LinearElastic, c_ms, κ, A_metric, A₁, A₂, G₃)
    D, _ = bending_and_shear_stiffness(mat, c_ms, A_metric)
    0.5 * (κ ⊡ D ⊡ κ)
end

# Hyperelastic: through-thickness Gauss quadrature in the physical Cartesian frame.
# Subtracts midsurface term to isolate the bending contribution (KL has no shear).
@inline function bending_kl_qp_energy(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                                       κ, A_metric, A₁, A₂, G₃) where T
    det_A = det(A_metric)
    Jinv  = inv(J_ref(A₁, A₂, G₃))
    ξ3s, w3s = gauss3_thickness(mat.thickness)
    W_ms = W_phys(mat, c_ms, det_A, Jinv)
    W = zero(T)
    for (ξ3, w3) in zip(ξ3s, w3s)
        W += W_phys(mat, c_ms + 2*ξ3*κ, det_A, Jinv) * w3
    end
    return W - mat.thickness * W_ms
end

"""
    bending_energy_KL(u_flat, scv::ShellCellValues, mat)

Computes the Kirchhoff–Love bending energy through the curvature change ``\\kappa_{\\alpha\\beta} = b_{\\alpha\\beta} - B_{\\alpha\\beta}``
(current minus reference second fundamental form).

**Arguments:**
* `u_flat`: the element vector `u[celldofs(cell)]`; its displacement dofs are read at `scv.ru`.
* `scv`: a [`ShellCellValues`](@ref) object containing the precomputed shape function values and derivatives.
* `mat`: an instance of an `AbstractMaterial` describing the material properties.

**Note:**
To capture the full curvature change ``\\kappa``, quadratic elements are required (linear element only captures twist ``\\kappa_{12}``).
"""
function bending_energy_KL(u_flat, scv::ShellCellValues, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    ru      = scv.ru
    u_e     = [Vec{3}(@view u_flat[ru[3i-2:3i]]) for i in 1:n_nodes]
    W  = zero(T)
    for qp in 1:getnquadpoints(scv)
        A₁q = scv.A₁[qp]; A₂q = scv.A₂[qp]
        a₁  = A₁q; a₂  = A₂q
        a₁₁ = Vec{3,T}(Tuple(scv.A₁₁[qp])); a₁₂ = Vec{3,T}(Tuple(scv.A₁₂[qp])); a₂₂ = Vec{3,T}(Tuple(scv.A₂₂[qp]))
        for I in 1:n_nodes
            UI  = u_e[I]; dN  = scv.dNdξ[I, qp]; d2N = scv.d2Ndξ2[I, qp]
            a₁  += UI * dN[1];     a₂  += UI * dN[2]
            a₁₁ += UI * d2N[1,1]; a₁₂ += UI * d2N[1,2]; a₂₂ += UI * d2N[2,2]
        end
        n_n  = (a₁ × a₂) / norm(a₁ × a₂)
        b    = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ    = b - scv.B[qp]
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        W   += bending_kl_qp_energy(mat, c_ms, κ, scv.A_metric[qp], A₁q, A₂q, scv.G₃[qp]) * scv.detJdV[qp]
    end
    return W
end

"""
    bending_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love bending residual via `ForwardDiff.gradient` of [`bending_energy_KL`](@ref).
`u_e = u[celldofs(cell)]`; its displacement dofs are read at `scv.ru`.
"""
function bending_residuals_KL!(re, scv::ShellCellValues, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
    bending_tangent_KL!(ke, scv, u_e, mat)

Kirchhoff–Love bending tangent via `ForwardDiff.hessian` of [`bending_energy_KL`](@ref).
`u_e = u[celldofs(cell)]`; its displacement dofs are read at `scv.ru`.
"""
function bending_tangent_KL!(ke, scv::ShellCellValues, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, mat), u_e)
end