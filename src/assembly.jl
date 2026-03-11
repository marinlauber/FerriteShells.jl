
using ForwardDiff

"""
    membrane_residuals_KL!(re, scv, u_e, mat)

Kirchhoff–Love membrane residual. `u_e` is a flat vector of length 3·n_nodes: [u₁,u₂,u₃, …].
"""
function membrane_residuals_KL!(re, scv, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
        E = 0.5 * (a_metric - A_metric)
        N = contravariant_elasticity(mat, A_metric) ⊡ E
        dΩ = scv.detJdV[qp]
        for I in 1:n_nodes
            ∂NI1, ∂NI2 = scv.dNdξ[I, qp]
            v = ∂NI1 * (N[1,1]*a₁ + N[1,2]*a₂) +
                ∂NI2 * (N[2,1]*a₁ + N[2,2]*a₂)
            @views re[3I-2:3I] .+= v * dΩ
        end
    end
end

"""
    membrane_tangent_KL!(ke, scv, u_e, mat)

Kirchhoff–Love membrane tangent. `u_e` is a flat vector of length 3·n_nodes.
"""
function membrane_tangent_KL!(ke, scv, u_e, mat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    for qp in 1:getnquadpoints(scv)
        a₁, a₂, A_metric, a_metric = kinematics(scv, qp, u_e)
        E = 0.5 * (a_metric - A_metric)
        C = contravariant_elasticity(mat, A_metric)
        N = C ⊡ E
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
                @views ke[3I-2:3I, 3J-2:3J] .+= (Kgeo + Kmat) * dΩ
            end
        end
    end
end

"""
Kirchhoff–Love bending strain energy.
Curvature change κ_{αβ} = b_{αβ} - B_{αβ} (current minus reference second fundamental form).
Requires Q2+ elements for full κ (Q4 only captures twist κ₁₂).
`u_e` is a flat vector of length 3·n_nodes.
"""
function bending_energy_KL(u_flat, scv, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    u_e     = [Vec{3,T}((u_flat[3i-2], u_flat[3i-1], u_flat[3i])) for i in 1:n_nodes]
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        a₁  = Vec{3,T}(Tuple(scv.A₁[qp]));  a₂  = Vec{3,T}(Tuple(scv.A₂[qp]))
        a₁₁ = Vec{3,T}(Tuple(scv.A₁₁[qp])); a₁₂ = Vec{3,T}(Tuple(scv.A₁₂[qp])); a₂₂ = Vec{3,T}(Tuple(scv.A₂₂[qp]))
        for I in 1:n_nodes
            UI  = u_e[I]
            dN  = scv.dNdξ[I, qp]
            d2N = scv.d2Ndξ2[I, qp]
            a₁  += UI * dN[1];     a₂  += UI * dN[2]
            a₁₁ += UI * d2N[1,1]; a₁₂ += UI * d2N[1,2]; a₂₂ += UI * d2N[2,2]
        end
        n_n = (a₁ × a₂) / norm(a₁ × a₂)
        b   = SymmetricTensor{2,2,T}((dot(a₁₁,n_n), dot(a₁₂,n_n), dot(a₂₂,n_n)))
        κ   = b - scv.B[qp]
        D   = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        W  += 0.5 * (κ ⊡ D ⊡ κ) * scv.detJdV[qp]
    end
    return W
end

function bending_residuals_KL!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> bending_energy_KL(u, scv, mat), u_e)
end

function bending_tangent_KL!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> bending_energy_KL(u, scv, mat), u_e)
end

"""
Reissner–Mindlin membrane strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).
Only the displacement DOFs (indices 5I-4:5I-2) contribute to membrane energy.
"""
function rm_membrane_energy(u_flat, scv, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁       = scv.A₁[qp] + Δa₁
        a₂       = scv.A₂[qp] + Δa₂
        a_metric = SymmetricTensor{2,2,T}((dot(a₁,a₁), dot(a₁,a₂), dot(a₂,a₂)))
        E        = 0.5 * (a_metric - scv.A_metric[qp])
        C        = contravariant_elasticity(mat, scv.A_metric[qp])
        W       += 0.5 * (E ⊡ C ⊡ E) * scv.detJdV[qp]
    end
    return W
end

"""
Reissner–Mindlin bending + transverse shear strain energy.
DOF layout: 5 DOFs per node — [u₁, u₂, u₃, φ₁, φ₂, …] (flat vector of length 5·n_nodes).

Director parametrization: d_I = G₃ + φ₁_I·T₁ + φ₂_I·T₂
where G₃ is the reference unit normal and T₁, T₂ are reference tangents from scv.

Bending strain:  κ_{αβ} = ½(a_α·d,β + a_β·d,α) - B_{αβ}
Transverse shear: γ_α = a_α·d

Shear correction factor κ_s = 5/6 is applied.
"""
function rm_bending_shear_energy(u_flat, scv, mat)
    T       = eltype(u_flat)
    n_nodes = getnbasefunctions(scv.ip_shape)
    W = zero(T)
    for qp in 1:getnquadpoints(scv)
        Δa₁ = zero(Vec{3,T}); Δa₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            dN  = scv.dNdξ[I, qp]
            u_I = Vec{3,T}((u_flat[5I-4], u_flat[5I-3], u_flat[5I-2]))
            Δa₁ += u_I * dN[1]; Δa₂ += u_I * dN[2]
        end
        a₁ = scv.A₁[qp] + Δa₁
        a₂ = scv.A₂[qp] + Δa₂
        G₃ = scv.G₃[qp]; T₁ = scv.T₁[qp]; T₂ = scv.T₂[qp]
        d  = zero(Vec{3,T}); d₁ = zero(Vec{3,T}); d₂ = zero(Vec{3,T})
        for I in 1:n_nodes
            φ₁  = u_flat[5I-1]; φ₂ = u_flat[5I]
            d_I = Vec{3,T}((G₃[1] + φ₁*T₁[1] + φ₂*T₂[1],
                            G₃[2] + φ₁*T₁[2] + φ₂*T₂[2],
                            G₃[3] + φ₁*T₁[3] + φ₂*T₂[3]))
            d  += scv.N[I, qp]      * d_I
            d₁ += scv.dNdξ[I, qp][1] * d_I
            d₂ += scv.dNdξ[I, qp][2] * d_I
        end
        B   = scv.B[qp]
        κ₁₁ = dot(a₁, d₁) - B[1,1]
        κ₁₂ = 0.5 * (dot(a₁, d₂) + dot(a₂, d₁)) - B[1,2]
        κ₂₂ = dot(a₂, d₂) - B[2,2]
        κ   = SymmetricTensor{2,2,T}((κ₁₁, κ₁₂, κ₂₂))
        γ₁  = dot(a₁, d)
        γ₂  = dot(a₂, d)
        D    = contravariant_bending_stiffness(mat, scv.A_metric[qp])
        Aup  = inv(scv.A_metric[qp])
        G_sh = mat.E / (2*(1 + mat.ν))
        κ_s  = 5.0/6.0
        W_bend  = 0.5 * (κ ⊡ D ⊡ κ)
        W_shear = 0.5 * κ_s * G_sh * mat.thickness *
                  (Aup[1,1]*γ₁^2 + 2*Aup[1,2]*γ₁*γ₂ + Aup[2,2]*γ₂^2)
        W += (W_bend + W_shear) * scv.detJdV[qp]
    end
    return W
end

"""
    membrane_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin membrane residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_membrane_energy(u, scv, mat), u_e)
end

"""
    membrane_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin membrane tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function membrane_tangent_RM!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_membrane_energy(u, scv, mat), u_e)
end

"""
    bending_residuals_RM!(re, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear residual. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_residuals_RM!(re, scv, u_e, mat)
    re .+= ForwardDiff.gradient(u -> rm_bending_shear_energy(u, scv, mat), u_e)
end

"""
    bending_tangent_RM!(ke, scv, u_e, mat)

Reissner–Mindlin bending + transverse shear tangent. `u_e` is a flat vector of length 5·n_nodes.
"""
function bending_tangent_RM!(ke, scv, u_e, mat)
    ke .+= ForwardDiff.hessian(u -> rm_bending_shear_energy(u, scv, mat), u_e)
end
