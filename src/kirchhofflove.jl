# Kirchhoff-Love bending; the membrane part is shared with RM (membrane.jl).

# Per-QP bending energy for KL, dispatched on material type.
# LinearElastic: ½ κ:D:κ, with D from the reference metric only (independent of u).
@inline function bending_kl_qp_energy(mat::LinearElastic, c_ms, κ, A_metric, A₁, A₂, G₃)
    _, C = membrane_stress_and_tangent(mat, A_metric, A_metric)
    D = bending_stiffness(mat, C)
    0.5 * (κ ⊡ D ⊡ κ)
end

# Hyperelastic: through-thickness Gauss quadrature in the physical Cartesian frame.
# Subtracts midsurface term to isolate the bending contribution (KL has no shear).
@inline function bending_kl_qp_energy(mat::Hyperelastic, c_ms::SymmetricTensor{2,2,T},
                                       κ, A_metric, A₁, A₂, G₃) where T
    det_A, Jinv = reference_frame(A_metric, A₁, A₂, G₃)
    ξ3s, w3s = gauss3_thickness(mat.thickness)
    W_ms = W_phys(mat, c_ms, det_A, Jinv)
    W = zero(T)
    for (ξ3, w3) in zip(ξ3s, w3s)
        W += W_phys(mat, c_ms + 2*ξ3*κ, det_A, Jinv) * w3
    end
    return W - mat.thickness * W_ms
end

# Deformed second covariant derivatives (a₁₁, a₁₂, a₂₂) at qp; displacement dofs at scv.ru.
@inline function covariant_basis_derivatives(scv, qp, u_e::AbstractVector{T}) where T
    ru = scv.ru
    Δa₁₁ = zero(Vec{3,T}); Δa₁₂ = zero(Vec{3,T}); Δa₂₂ = zero(Vec{3,T})
    for I in 1:getnbasefunctions(scv)
        u_I = Vec{3}(@view u_e[ru[3I-2:3I]]); d2N = scv.d2Ndξ2[I, qp]
        Δa₁₁ += u_I * d2N[1,1]; Δa₁₂ += u_I * d2N[1,2]; Δa₂₂ += u_I * d2N[2,2]
    end
    scv.A₁₁[qp] + Δa₁₁, scv.A₁₂[qp] + Δa₁₂, scv.A₂₂[qp] + Δa₂₂
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
    T = eltype(u_flat)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂        = covariant_basis(scv, qp, u_flat)
        a₁₁, a₁₂, a₂₂ = covariant_basis_derivatives(scv, qp, u_flat)
        n    = a₁ × a₂; n /= norm(n)
        κ    = SymmetricTensor{2,2,T}((dot(a₁₁,n), dot(a₁₂,n), dot(a₂₂,n))) - scv.B[qp]
        c_ms = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        W   += bending_kl_qp_energy(mat, c_ms, κ, scv.A_metric[qp], scv.A₁[qp], scv.A₂[qp], scv.G₃[qp]) * scv.detJdV[qp]
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
